from typing import Tuple, Union, Optional

import torch
from torch import nn, Tensor
from transformers.activations import get_activation

from .config import ModelConfig
from .utils import LINEAR

_MAX_SQRT_GRADIENT = 1000.0


class SqrtBoundDerivative(torch.autograd.Function):
    """Computes a square root with a gradient clipped at `_MAX_SQRT_GRADIENT`."""

    @staticmethod
    def forward(ctx, x: Tensor, ) -> Tensor:
        """The forward pass, which is a normal `sqrt`."""
        ctx.save_for_backward(x)
        return torch.sqrt(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        """The backward pass, which clips the `sqrt` gradient."""
        (x,) = ctx.saved_tensors
        clipped_x_times_4 = torch.clip(4.0 * x, min=1 / (_MAX_SQRT_GRADIENT ** 2))
        return grad_output / torch.sqrt(clipped_x_times_4)


class RGLRU(nn.Module):
    """A Real-Gated Linear Recurrent Unit (RG-LRU) layer."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.block_width = config.lru_width // self.num_attention_heads

        self.recurrent_param = nn.Parameter(torch.empty([config.lru_width]))
        self.input_gate_weight = nn.Parameter(
            torch.empty([self.num_attention_heads, self.block_width, self.block_width])
        )
        self.input_gate_bias = nn.Parameter(torch.empty([self.num_attention_heads, self.block_width]))

        self.recurrent_gate_weight = nn.Parameter(
            torch.empty([self.num_attention_heads, self.block_width, self.block_width])
        )
        self.recurrent_gate_bias = nn.Parameter(torch.empty([self.num_attention_heads, self.block_width]))
        self.recurrent_states = None

    def forward(
            self,
            activations: Tensor,
            position_ids: Tensor,
    ) -> Tensor:
        batch_size, seq_len, lru_width = activations.shape
        reset = position_ids[:, :, None] == 0

        reshape_act = activations.reshape(batch_size * seq_len, self.num_attention_heads, self.block_width)
        reshape_act = reshape_act.permute(1, 0, 2)

        res = torch.baddbmm(self.input_gate_bias[:, None, :], reshape_act, self.input_gate_weight)
        input_gate = torch.sigmoid(res.transpose(0, 1).reshape(batch_size, seq_len, lru_width))

        res = torch.baddbmm(self.recurrent_gate_bias[:, None, :], reshape_act, self.recurrent_gate_weight)
        recurrent_gate = torch.sigmoid(res.transpose(0, 1).reshape(batch_size, seq_len, lru_width))

        # Compute the parameter `A` of the recurrence.
        log_recurrent_gate = -8.0 * recurrent_gate * nn.functional.softplus(self.recurrent_param)
        recurrent_gate = torch.exp(log_recurrent_gate)
        a_square = torch.exp(2 * log_recurrent_gate)

        # Gate the input.
        gated_inputs = activations * input_gate

        # Apply gamma normalization to the input. We need to clip the derivatives of
        # `sqrt` in order to prevent NaNs during training in bfloat16. TODO a bit annoying
        multiplier = 1
        tracing = isinstance(activations, torch.fx.Proxy) or (
                hasattr(torch, "_dynamo") and torch._dynamo.is_compiling()
        )
        if not torch.jit.is_tracing() and not tracing:
            multiplier = SqrtBoundDerivative.apply(1 - a_square)
        multiplier = reset + ~reset * multiplier
        normalized_x = gated_inputs * multiplier.type(activations.dtype)

        hidden_states, recurrent_states = self._rnn_scan(
            hidden_states=normalized_x,
            recurrent_gate=recurrent_gate,
            reset=reset,
            recurrent_states=self.recurrent_states,
        )
        if not self.training:
            self.recurrent_states = recurrent_states
        return hidden_states

    # TODO refactor
    @staticmethod
    def _rnn_scan(
            hidden_states: Tensor,
            recurrent_gate: Tensor,
            reset: Tensor | bool,
            recurrent_states: Union[Tensor, None],
            acc_dtype: torch.dtype = torch.float32,
    ) -> Tuple[Tensor, Tensor]:
        """Runs the recurrence of a linear RNN.

        Args:
        hidden_states: The input sequence.
        recurrent_gate: The diagonal of the recurrence matrix `A`.
        reset: Indicator of document boundaries, e.g. when to reset the hidden state
            of the RNN.
        recurrent_states: The initial hidden state.
        acc_dtype: The data type for the accumulation.

        Returns:
        The output of the linear recurrence.
        """
        # Multiply `a` by the reset.
        recurrent_gate = recurrent_gate * ~reset

        if hidden_states.shape[1] == 1:
            # Using scan in sampling mode.
            if recurrent_states is None:  # same here, when decoding you always have cache
                return hidden_states, hidden_states[:, 0].type(acc_dtype)

            else:
                contextualized_states = recurrent_gate.type(acc_dtype) * recurrent_states[:, None]
                contextualized_states += hidden_states.type(acc_dtype)
                return contextualized_states.type(hidden_states.dtype), contextualized_states[:, -1]

        else:
            # Using scan in linear mode.
            if recurrent_states is None:
                recurrent_states = torch.zeros(hidden_states[:, 0].shape, dtype=acc_dtype, device=hidden_states.device)

            contextualized_states = torch.zeros_like(hidden_states)
            for t in range(hidden_states.shape[1]):
                recurrent_states = recurrent_gate[:, t].type(acc_dtype) * recurrent_states
                recurrent_states = recurrent_states + hidden_states[:, t].type(acc_dtype)
                contextualized_states[:, t] = recurrent_states.type(hidden_states.dtype)

        return contextualized_states, recurrent_states


class RecurrentBlock(nn.Module):
    """Griffin and Hawk's recurrent block."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.lru_width = config.lru_width
        self.hidden_size = config.hidden_size
        self.linear_y: LINEAR = nn.Linear(in_features=config.hidden_size, out_features=config.lru_width)
        self.linear_x: LINEAR = nn.Linear(in_features=config.hidden_size, out_features=config.lru_width)
        self.linear_out: LINEAR = nn.Linear(in_features=config.lru_width, out_features=config.hidden_size)
        self.conv1d_width = config.conv1d_width
        self.conv_1d = nn.Conv1d(
            config.lru_width,
            config.lru_width,
            kernel_size=config.conv1d_width,
            groups=config.lru_width,
            padding=config.conv1d_width - 1,
        )
        self.rg_lru = RGLRU(config)
        self.act_fn = get_activation(config.hidden_act)

        self.conv1d_state = None
        self._cache_length = 0

    def forward(
            self,
            input_states: Tensor,
            attention_mask: Optional[Tensor],
            position_ids: Tensor,
            cache_position: Optional[Tensor] = None,
    ) -> Tensor:
        _, seq_len, _ = input_states.shape

        y_branch = self.linear_y(input_states)
        y_branch = self.act_fn(y_branch)

        x_branch = self.linear_x(input_states)
        x_branch = x_branch.transpose(1, 2)

        if cache_position is not None and not self.training:
            self._cache_length = cache_position[-1] + 1
            if cache_position.shape[0] != 1:  # prefill
                self.conv1d_state = nn.functional.pad(x_branch, (self.conv1d_width - x_branch.shape[-1] - 1, 0))
                x_branch = self.conv_1d(x_branch)[..., :seq_len]
            else:  # decoding
                conv_state = torch.cat((self.conv1d_state, x_branch), -1)
                x_branch = torch.sum(conv_state * self.conv_1d.weight[:, 0, :], dim=-1) + self.conv_1d.bias
                x_branch = x_branch.unsqueeze(-1)
                self.conv1d_state = conv_state[:, :, 1:]
        else:
            x_branch = self.conv_1d(x_branch)[..., :seq_len]

        x_branch = self.rg_lru(x_branch.transpose(1, 2), position_ids)

        hidden_states = x_branch * y_branch
        hidden_states = self.linear_out(hidden_states)
        return hidden_states

    def setup_cache(self, dtype: torch.dtype, device: torch.device):
        # recurrent_states always computed in full precision
        self.rg_lru.recurrent_states = torch.zeros((self.config.max_batch_size, self.lru_width),
                                                   device=device, dtype=torch.float32)
        self.conv1d_state = torch.zeros(
            (self.config.max_batch_size,
             self.hidden_size,
             self.conv1d_width - 1),
            device=device, dtype=dtype)

    @torch.no_grad()
    def reorder_cache(self, beam_idx: Tensor) -> None:
        assert self.rg_lru.recurrent_states is not None, "Cache is not initialized, call setup_cache() first."
        self.rg_lru.recurrent_states = self.rg_lru.recurrent_states.index_select(
            0,
            beam_idx.to(self.rg_lru.recurrent_states.device)
        )
        self.conv1d_state = self.conv1d_state.index_select(
            0, beam_idx.to(self.conv1d_state.device)
        )

    def get_cache_length(self) -> int | Tensor:
        return self._cache_length
