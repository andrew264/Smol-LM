import torch
import torch.nn as nn
from torch import Tensor

from model.config import ModelConfig
from .simple_transformer import SimpleTransformer


class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1):
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
        self.encoder_dim = 384

        self.conv_layers = nn.ModuleList([
            ConvNorm(80, self.encoder_dim, 3, 1),
            ConvNorm(self.encoder_dim, self.encoder_dim, 3, 2),
        ])
        self.transformer = SimpleTransformer(self.encoder_dim, 6, 4)
        self.proj = nn.Linear(self.encoder_dim, config.hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        for conv in self.conv_layers:
            x = conv(x)
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        return self.proj(x)
