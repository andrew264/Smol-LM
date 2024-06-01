from copy import deepcopy
from typing import List, Union, Dict

import numpy as np
import torch
import torch.nn as nn
from torchaudio.functional import resample
from torchaudio.transforms import MelSpectrogram

from .block import Block
from .config import ModelConfig


class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class AudioHead(torch.nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.audio_preprocessor = AudioFeatureExtractor()
        self.encoder_dim = 384

        self.conv_layers = nn.ModuleList([
            ConvNorm(80, self.encoder_dim, 3, 1),
            ConvNorm(self.encoder_dim, self.encoder_dim, 3, 2),
        ])
        self.proj = nn.Linear(self.encoder_dim, config.hidden_size)
        config = deepcopy(config)
        config.hidden_size = self.encoder_dim
        config.intermediate_size = self.encoder_dim * 4
        config.num_hidden_layers = 4
        config.num_attention_heads = 6
        config.num_key_value_heads = 6
        self.attention_blocks = nn.ModuleList([
            Block(config, causal=False) for _ in range(config.num_hidden_layers)
        ])

    def forward(self, inputs: Dict[str, Union[torch.Tensor, np.ndarray, int]]):
        x = self.audio_preprocessor(inputs)
        for conv in self.conv_layers:
            x = conv(x)
        x = x.permute(0, 2, 1)
        position_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        for block in self.attention_blocks:
            x = block(x, position_ids=position_ids)
        return self.proj(x)


class AudioFeatureExtractor(nn.Module):
    def __init__(self, sample_rate: int = 16000, n_fft: int = 400, hop_length: int = 160, num_mel_bins: int = 80):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_samples = 30 * sample_rate

        self.mel_spectrogram = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm='slaney',
            onesided=True,
            n_mels=num_mel_bins,
            mel_scale="slaney"
        )
        self.fix_applied = False

    def fix_low_precision_training(self):
        self.mel_spectrogram.mel_scale = self.mel_spectrogram.mel_scale.to(dtype=torch.float32)

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, inputs: Dict[str, List[Union[torch.Tensor, np.ndarray, int]]]):
        if not self.fix_applied:
            self.fix_low_precision_training()
            self.fix_applied = True
        waveforms = inputs.get("audio")
        sampling_rate = inputs.get("sampling_rate")
        for i in range(len(waveforms)):
            sr = sampling_rate[i]
            audio = waveforms[i]
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio).to(device="cuda", dtype=torch.float32)
            if sr != self.sample_rate:
                audio = resample(audio, orig_freq=sr, new_freq=self.sample_rate, lowpass_filter_width=8)
            waveforms[i] = audio

        # pad waveforms to n_samples or truncate if longer
        waveforms = [torch.nn.functional.pad(w, (0, max(0, self.n_samples - w.shape[-1])))[:self.n_samples] for w in
                     waveforms]
        waveforms = torch.stack(waveforms)
        x = self.mel_spectrogram(waveforms)
        x = torch.clamp(x, min=1e-8).log10()

        # normalize
        x = torch.maximum(x, x.max(dim=-1, keepdim=True)[0] - 8.0)
        x = (x + 4.0) / 4.0

        return x.to(dtype=torch.bfloat16)
