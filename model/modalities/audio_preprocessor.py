from typing import Union, Dict

import numpy as np
import torch
from torch import nn
from torchaudio.functional import resample
from torchaudio.transforms import MelSpectrogram


class AudioFeatureExtractor(nn.Module):
    def __init__(self, output_dtype: torch.dtype = torch.bfloat16,
                 sample_rate: int = 16000, n_fft: int = 400, hop_length: int = 160, num_mel_bins: int = 80):
        super().__init__()
        self.output_dtype = output_dtype
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

    @torch.no_grad()
    def forward(self, inputs: Dict[str, Union[np.ndarray, int]]) -> torch.Tensor:
        waveform = inputs.get("audio")
        sampling_rate = inputs.get("sampling_rate")

        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).to(dtype=torch.float32)
        if sampling_rate != self.sample_rate:
            waveform = resample(waveform, orig_freq=sampling_rate, new_freq=self.sample_rate, lowpass_filter_width=8)

        # pad waveforms to n_samples or truncate if longer than n_samples
        if waveform.size(0) < self.n_samples:
            waveform = torch.cat([waveform, torch.zeros(self.n_samples - waveform.size(0))])
        else:
            waveform = waveform[:self.n_samples]
        x = self.mel_spectrogram(waveform.unsqueeze(0)).squeeze(0)
        x = torch.clamp(x, min=1e-8).log10()

        # normalize
        x = torch.maximum(x, x.max(dim=-1, keepdim=True)[0] - 8.0)
        x = (x + 4.0) / 4.0

        return x.to(self.output_dtype)
