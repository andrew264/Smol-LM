import torch
import torch.nn as nn
import torchaudio.transforms as T

from .config import AudioConfig
from .norm import get_rms_norm_class


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
        NORM_CLASS = get_rms_norm_class()
        self.conv1 = nn.Conv1d(config.n_mels, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1)
        self.act = nn.GELU()

        self.ln1 = NORM_CLASS(hidden_size)

    def _fix_low_precision_training(self):
        """
        no fp16 in spectrogram for now cuz complex32 is not fully supported
        """
        self.mel_spectrogram = self.mel_spectrogram.to(dtype=torch.float32)

    def forward(self, x):
        with torch.autocast(enabled=False, device_type=x.device.type):
            features = self.mel_spectrogram(x)
        if features.dtype in (torch.float16, torch.float32):
            features = features.to(torch.bfloat16)
        features = self.act(self.conv1(features))
        features = self.act(self.conv2(features))
        features = self.act(self.conv3(features))
        features = self.act(self.conv4(features))
        features = features.permute(0, 2, 1)

        return self.ln1(features)  # batch, seq_len, hidden_size
