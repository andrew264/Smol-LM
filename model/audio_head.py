import torch
import torch.nn as nn

from .block import Block
from .config import ModelConfig


class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class AudioHead(torch.nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        hs = config.hidden_size
        conv_dim = (1, hs, hs, hs, hs, hs, hs, hs)
        conv_stride = (5, 2, 2, 2, 2, 2, 2)
        conv_kernel = (10, 3, 3, 3, 3, 3, 3)

        conv_layers = []
        for i in range(7):
            conv_layers.append(ConvNorm(conv_dim[i], conv_dim[i + 1], conv_kernel[i], conv_stride[i]))
        self.conv_layers = nn.ModuleList(conv_layers)

        self.attn_blocks = nn.ModuleList([Block(config, i) for i in range(3)])

    def forward(self, x):
        if x.dtype == torch.float32:
            x = x.to(dtype=torch.bfloat16)
        x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)
        x = x.permute(0, 2, 1)
        position_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        for block in self.attn_blocks:
            x = block(x, position_ids=position_ids)[0]
        return x
