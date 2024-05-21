import torch
import torch.nn as nn
import torchaudio.transforms as T

from .config import AudioConfig


class AudioHead(torch.nn.Module):
    def __init__(self, config: AudioConfig, hidden_size):
        super().__init__()
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm='slaney',
            n_mels=config.n_mels,
            mel_scale="htk",
        )
        self.conv1 = nn.Conv1d(config.n_mels, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1)
        self.act = nn.GELU()

    def _fix_low_precision_training(self):
        self.mel_spectrogram = self.mel_spectrogram.to(dtype=torch.float32)

    def forward(self, x):
        with torch.autocast(enabled=False, device_type=x.device.type):
            features = self.mel_spectrogram(x)
        if features.dtype == torch.float32:
            features = features.to(torch.bfloat16)
        features = self.act(self.conv1(features))
        features = self.act(self.conv2(features))
        features = features.permute(0, 2, 1)

        return features  # batch, seq_len, hidden_size
