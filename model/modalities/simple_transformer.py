import torch
from einops import rearrange
from torch import nn, Tensor

try:
    from flash_attn import flash_attn_qkvpacked_func
    from flash_attn.layers.rotary import RotaryEmbedding
except ImportError:
    flash_attn_qkvpacked_func = None
    RotaryEmbedding = None

from model.norm import get_rmsnorm_class


class SimpleAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8, ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.rotary_emb = RotaryEmbedding(dim=self.head_dim, )
        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        bsz, seqlen, _ = x.size()
        qkv_states = self.qkv_proj(x)
        qkv_states = rearrange(qkv_states, "... (three h d) -> ... three h d", three=3, d=self.head_dim)

        qkv_states = self.rotary_emb(qkv_states)

        attn_output = flash_attn_qkvpacked_func(qkv_states, causal=False)

        attn_output = attn_output.contiguous().view(bsz, seqlen, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output


class SimpleFeedForward(nn.Module):
    def __init__(self, hidden_size, ):
        super().__init__()
        intermediate_size = hidden_size * 4
        self.up_proj = nn.Linear(hidden_size, intermediate_size, )
        self.down_proj = nn.Linear(intermediate_size, hidden_size, )
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act(self.up_proj(x)))


class SimpleBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, ):
        super().__init__()
        self.attention_block = SimpleAttention(hidden_size, num_heads)
        self.feed_forward = SimpleFeedForward(hidden_size)

        NORM_CLASS = get_rmsnorm_class()
        self.input_layernorm = NORM_CLASS(hidden_size)
        self.post_attention_layernorm = NORM_CLASS(hidden_size)

    def forward(self, x):
        residual = x
        x = self.input_layernorm(x)
        x = self.attention_block(x)
        residual = residual + x

        x = self.post_attention_layernorm(residual)
        x = self.feed_forward(x)
        x = residual + x

        return x


class SimpleTransformer(nn.Module):
    def __init__(self, hidden_size, num_heads, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            SimpleBlock(hidden_size, num_heads)
            for _ in range(num_layers)
        )
        self.norm = get_rmsnorm_class()(hidden_size)

    def forward(self, x):
        for block in self.layers:
            x = block(x)
        return self.norm(x)


if __name__ == '__main__':
    a = SimpleAttention(256).cuda().bfloat16()
    a(torch.rand(1, 10, 256, device='cuda', dtype=torch.bfloat16))
