# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Qwen3.5 (qwen3_next) model for ExecuTorch export.

Hybrid architecture: Gated DeltaNet (linear attention) + Gated Attention layers.
Supports XNNPACK and Metal backends.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Qwen3_5Config:
    hidden_size: int = 5120
    num_hidden_layers: int = 64
    num_attention_heads: int = 24
    num_key_value_heads: int = 4
    head_dim: int = 256
    intermediate_size: int = 17408
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000000.0
    partial_rotary_factor: float = 0.25
    max_position_embeddings: int = 262144

    # DeltaNet parameters
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    full_attention_interval: int = 4

    # Derived
    layer_types: List[str] = field(default_factory=list)

    # Runtime
    max_seq_len: int = 2048
    use_metal_attention: bool = False

    def __post_init__(self):
        if not self.layer_types:
            self.layer_types = [
                "linear_attention"
                if bool((i + 1) % self.full_attention_interval)
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)

    @classmethod
    def from_json(cls, path: str) -> "Qwen3_5Config":
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Qwen3Next-style RMSNorm: uses (1 + weight) scaling."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        normed = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * (1.0 + self.weight.float())).type_as(x)


class RMSNormGated(nn.Module):
    """RMSNorm followed by SiLU gating."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        normed = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        normed = self.weight * normed.type_as(x)
        return normed * F.silu(gate.float()).type_as(x)


# ---------------------------------------------------------------------------
# Rotary Position Embedding (Partial)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        dim = config.rotary_dim
        inv_freq = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(0, dim, 2, dtype=torch.float32) / dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> tuple:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple:
    """Apply rotary embeddings to the first rotary_dim dimensions only."""
    rotary_dim = cos.shape[-1]

    q_rot = q[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]
    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]

    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, rotary_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    return (
        torch.cat([q_embed, q_pass], dim=-1),
        torch.cat([k_embed, k_pass], dim=-1),
    )


# ---------------------------------------------------------------------------
# KV Cache (XNNPACK style)
# ---------------------------------------------------------------------------

class KVCache(nn.Module):
    """Static KV cache with [B, S, H, D] layout for XNNPACK."""

    def __init__(self, max_seq_len: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        self.register_buffer(
            "k_cache",
            torch.zeros(1, max_seq_len, n_kv_heads, head_dim),
        )
        self.register_buffer(
            "v_cache",
            torch.zeros(1, max_seq_len, n_kv_heads, head_dim),
        )

    def update(
        self, input_pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple:
        """k, v: [B, S, H, D]. Stores and returns full cache."""
        k = k.to(self.k_cache.dtype)
        v = v.to(self.v_cache.dtype)
        self.k_cache[:, input_pos] = k
        self.v_cache[:, input_pos] = v
        return self.k_cache, self.v_cache

    def reset(self):
        self.k_cache.zero_()
        self.v_cache.zero_()


# ---------------------------------------------------------------------------
# KV Cache (Metal style)
# ---------------------------------------------------------------------------

class StaticKVCache(nn.Module):
    """Static KV cache with [B, H, S, D] layout for Metal (index_copy_)."""

    def __init__(self, max_seq_len: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        self.register_buffer(
            "k_cache",
            torch.zeros(1, n_kv_heads, max_seq_len, head_dim),
        )
        self.register_buffer(
            "v_cache",
            torch.zeros(1, n_kv_heads, max_seq_len, head_dim),
        )

    def update(
        self, input_pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple:
        """k, v: [B, H, S, D]. Uses index_copy_ for Metal compatibility."""
        k = k.to(self.k_cache.dtype)
        v = v.to(self.v_cache.dtype)
        self.k_cache.index_copy_(2, input_pos, k)
        self.v_cache.index_copy_(2, input_pos, v)
        return self.k_cache, self.v_cache

    def reset(self):
        self.k_cache.zero_()
        self.v_cache.zero_()


# ---------------------------------------------------------------------------
# Attention Mask Builders
# ---------------------------------------------------------------------------

def build_causal_mask_metal(
    input_pos: torch.Tensor, max_seq_len: int, device: torch.device
) -> torch.Tensor:
    """Build causal mask without bool tensors (Metal AOTI requirement)."""
    k_pos = torch.arange(max_seq_len, device=device)
    diff = input_pos.unsqueeze(1) - k_pos.unsqueeze(0) + 1
    valid = torch.clamp(diff, min=0, max=1)
    return (valid.float() - 1.0) * 1e9


# ---------------------------------------------------------------------------
# SDPA Variants
# ---------------------------------------------------------------------------

class SDPA(nn.Module):
    """Standard SDPA for XNNPACK -- uses F.scaled_dot_product_attention."""

    def __init__(self, n_heads: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = head_dim
        self.dim = n_heads * head_dim

    def forward(
        self,
        input_pos: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bsz: int,
        seqlen: int,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # q: [B, n_heads, S, D], k/v from cache: [B, S_total, n_kv_heads, D]
        # Expand KV for GQA
        k = k.transpose(1, 2)  # [B, n_kv_heads, S_total, D]
        v = v.transpose(1, 2)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * scale
        if mask is not None:
            attn_weights = attn_weights + mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        y = torch.matmul(attn_weights, v)

        return y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)


class DecomposedSDPA(nn.Module):
    """Decomposed SDPA for Metal -- avoids the head_dim=256 SDPA kernel limitation.

    Uses matmul+softmax+matmul instead of the MPS SDPA op, since the Metal
    SDPA kernel only supports head_dim 64/96/128.
    """

    def __init__(self, n_heads: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = head_dim
        self.dim = n_heads * head_dim

    def forward(
        self,
        input_pos: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bsz: int,
        seqlen: int,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # k, v from StaticKVCache: [B, H, S_total, D]
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * scale
        if mask is not None:
            attn_weights = attn_weights + mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        y = torch.matmul(attn_weights, v)

        return y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)


# ---------------------------------------------------------------------------
# Gated Attention Block (Full Attention with Sigmoid Gating)
# ---------------------------------------------------------------------------

class GatedAttention(nn.Module):
    def __init__(self, config: Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads

        # Q is doubled for gating: first half is query, second half is gate
        self.q_proj = nn.Linear(
            config.hidden_size, self.n_heads * self.head_dim * 2, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, config.hidden_size, bias=False
        )

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        if config.use_metal_attention:
            self.kv_cache = StaticKVCache(
                config.max_seq_len, self.n_kv_heads, self.head_dim
            )
            self.sdpa = DecomposedSDPA(
                self.n_heads, self.n_kv_heads, self.head_dim
            )
        else:
            self.kv_cache = KVCache(
                config.max_seq_len, self.n_kv_heads, self.head_dim
            )
            self.sdpa = SDPA(self.n_heads, self.n_kv_heads, self.head_dim)

        self.use_metal = config.use_metal_attention

    def forward(
        self,
        x: torch.Tensor,
        input_pos: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        # Project Q (doubled), K, V
        qg = self.q_proj(x)
        # Split into query and gate
        qg = qg.view(bsz, seqlen, self.n_heads, self.head_dim * 2)
        q, gate = qg[..., : self.head_dim], qg[..., self.head_dim :]
        gate = gate.reshape(bsz, seqlen, -1)  # [B, S, n_heads * head_dim]

        # Per-head QK norms
        q = self.q_norm(q)  # [B, S, n_heads, head_dim]
        k = self.k_norm(
            self.k_proj(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        )
        v = self.v_proj(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # Partial RoPE
        q = q.transpose(1, 2)  # [B, n_heads, S, head_dim]
        k_for_rope = k.transpose(1, 2)  # [B, n_kv_heads, S, head_dim]
        cos = freqs_cos[input_pos]
        sin = freqs_sin[input_pos]
        q, k_for_rope = apply_partial_rotary_emb(q, k_for_rope, cos, sin)

        if self.use_metal:
            # Metal: [B, H, S, D] layout
            v_t = v.transpose(1, 2)  # [B, H, S, D]
            k_cache, v_cache = self.kv_cache.update(input_pos, k_for_rope, v_t)
            mask = build_causal_mask_metal(
                input_pos, k_cache.shape[2], q.device
            )
            mask = mask.unsqueeze(0).unsqueeze(0)
            attn_out = self.sdpa(input_pos, q, k_cache, v_cache, bsz, seqlen, mask)
        else:
            # XNNPACK: [B, S, H, D] layout
            k_for_cache = k_for_rope.transpose(1, 2)
            k_cache, v_cache = self.kv_cache.update(input_pos, k_for_cache, v)
            # Build causal mask
            max_sl = k_cache.shape[1]
            positions = torch.arange(max_sl, device=q.device)
            mask = (input_pos.unsqueeze(1) >= positions.unsqueeze(0)).float()
            mask = (1.0 - mask.unsqueeze(0).unsqueeze(0)) * torch.finfo(q.dtype).min
            attn_out = self.sdpa(input_pos, q, k_cache, v_cache, bsz, seqlen, mask)

        # Sigmoid gating
        attn_out = attn_out * torch.sigmoid(gate)
        return self.o_proj(attn_out)


# ---------------------------------------------------------------------------
# L2 Norm Utility
# ---------------------------------------------------------------------------

def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


# ---------------------------------------------------------------------------
# Gated DeltaNet Block (Linear Attention with Recurrent State)
# ---------------------------------------------------------------------------

class GatedDeltaNet(nn.Module):
    """Export-compatible Gated DeltaNet layer.

    Uses registered buffers for conv_state and recurrent_state.
    Supports decode mode only (seq_len=1 per call for recurrence,
    or variable seq_len for initial prefill via token-by-token iteration).
    """

    def __init__(self, config: Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.gqa_factor = self.num_v_heads // self.num_k_heads

        self.conv_dim = self.key_dim * 2 + self.value_dim

        # Projections
        proj_size_qkvz = self.key_dim * 2 + self.value_dim * 2
        proj_size_ba = self.num_v_heads * 2
        self.in_proj_qkvz = nn.Linear(self.hidden_size, proj_size_qkvz, bias=False)
        self.in_proj_ba = nn.Linear(self.hidden_size, proj_size_ba, bias=False)

        # Depthwise conv1d (no padding -- manual state management)
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=self.conv_kernel_size,
            padding=0,
            groups=self.conv_dim,
            bias=False,
        )

        # Recurrence parameters
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))

        # Output
        self.norm = RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        # State buffers (decode mode)
        self.register_buffer(
            "conv_state",
            torch.zeros(1, self.conv_dim, self.conv_kernel_size - 1),
        )
        self.register_buffer(
            "recurrent_state",
            torch.zeros(1, self.num_v_heads, self.head_k_dim, self.head_v_dim),
        )

    def _unpack_qkvzba(
        self, hidden_states: torch.Tensor
    ) -> tuple:
        """Project and unpack Q, K, V, Z (gate), beta, alpha."""
        bsz, seq_len, _ = hidden_states.shape

        mixed_qkvz = self.in_proj_qkvz(hidden_states)
        mixed_ba = self.in_proj_ba(hidden_states)

        # Reshape for grouped heads
        mixed_qkvz = mixed_qkvz.view(
            bsz,
            seq_len,
            self.num_k_heads,
            2 * self.head_k_dim
            + 2 * self.head_v_dim * self.gqa_factor,
        )
        mixed_ba = mixed_ba.view(
            bsz, seq_len, self.num_k_heads, 2 * self.gqa_factor
        )

        # Split
        split_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            self.gqa_factor * self.head_v_dim,
            self.gqa_factor * self.head_v_dim,
        ]
        split_ba = [self.gqa_factor, self.gqa_factor]

        q, k, v, z = torch.split(mixed_qkvz, split_qkvz, dim=3)
        b, a = torch.split(mixed_ba, split_ba, dim=3)

        # Reshape v, z from grouped to per-head
        v = v.reshape(bsz, seq_len, self.num_v_heads, self.head_v_dim)
        z = z.reshape(bsz, seq_len, self.num_v_heads, self.head_v_dim)
        b = b.reshape(bsz, seq_len, self.num_v_heads)
        a = a.reshape(bsz, seq_len, self.num_v_heads)

        return q, k, v, z, b, a

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        q, k, v, z, b, a = self._unpack_qkvzba(hidden_states)

        # Flatten Q, K, V for conv1d
        q_flat = q.reshape(bsz, seq_len, -1)  # [B, S, key_dim]
        k_flat = k.reshape(bsz, seq_len, -1)
        v_flat = v.reshape(bsz, seq_len, -1)  # [B, S, value_dim]

        mixed_qkv = torch.cat([q_flat, k_flat, v_flat], dim=-1)  # [B, S, conv_dim]
        mixed_qkv = mixed_qkv.transpose(1, 2)  # [B, conv_dim, S]

        # Conv1d with state management (like LFM2 ShortConv)
        mixed_qkv = torch.cat([self.conv_state, mixed_qkv], dim=-1)
        new_conv_state = mixed_qkv[..., -(self.conv_kernel_size - 1) :]
        with torch.no_grad():
            self.conv_state.copy_(new_conv_state)

        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[..., :seq_len])
        mixed_qkv = mixed_qkv.transpose(1, 2)  # [B, S, conv_dim]

        # Split back to Q, K, V
        q_conv, k_conv, v_conv = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )

        # Reshape to heads
        q_h = q_conv.reshape(bsz, seq_len, self.num_k_heads, self.head_k_dim)
        k_h = k_conv.reshape(bsz, seq_len, self.num_k_heads, self.head_k_dim)
        v_h = v_conv.reshape(bsz, seq_len, self.num_v_heads, self.head_v_dim)

        # Compute gates
        beta = b.sigmoid()   # [B, S, num_v_heads]
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        # GQA expansion for DeltaNet
        if self.gqa_factor > 1:
            q_h = q_h.repeat_interleave(self.gqa_factor, dim=2)
            k_h = k_h.repeat_interleave(self.gqa_factor, dim=2)

        # L2 normalize Q, K
        q_h = l2norm(q_h, dim=-1)
        k_h = l2norm(k_h, dim=-1)

        # Transpose to [B, heads, S, dim] for recurrence
        q_h = q_h.transpose(1, 2).contiguous().float()
        k_h = k_h.transpose(1, 2).contiguous().float()
        v_h = v_h.transpose(1, 2).contiguous().float()
        beta = beta.transpose(1, 2).contiguous().float()  # [B, heads, S]
        g = g.transpose(1, 2).contiguous().float()

        scale = 1.0 / math.sqrt(self.head_k_dim)

        # Recurrent delta rule (works for any seq_len, step by step)
        rec_state = self.recurrent_state.float()
        outputs = []

        for i in range(seq_len):
            q_t = q_h[:, :, i] * scale    # [B, heads, k_dim]
            k_t = k_h[:, :, i]            # [B, heads, k_dim]
            v_t = v_h[:, :, i]            # [B, heads, v_dim]
            g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)  # [B, heads, 1, 1]
            beta_t = beta[:, :, i].unsqueeze(-1)                # [B, heads, 1]

            # Decay
            rec_state = rec_state * g_t
            # Read
            kv_mem = (rec_state * k_t.unsqueeze(-1)).sum(dim=-2)
            # Delta update
            delta = (v_t - kv_mem) * beta_t
            # Write
            rec_state = rec_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            # Query
            out_t = (rec_state * q_t.unsqueeze(-1)).sum(dim=-2)
            outputs.append(out_t)

        with torch.no_grad():
            self.recurrent_state.copy_(rec_state.to(self.recurrent_state.dtype))

        # Stack outputs: [B, heads, S, v_dim]
        core_out = torch.stack(outputs, dim=2)
        core_out = core_out.transpose(1, 2).contiguous().to(hidden_states.dtype)
        # [B, S, heads, v_dim]

        # RMSNormGated + output projection
        z_shape = z.shape
        core_out = core_out.reshape(-1, core_out.shape[-1])
        z_flat = z.reshape(-1, z.shape[-1])
        core_out = self.norm(core_out, z_flat)
        core_out = core_out.reshape(z_shape)
        core_out = core_out.reshape(z_shape[0], z_shape[1], -1)

        return self.out_proj(core_out)

    def reset_cache(self):
        self.conv_state.zero_()
        self.recurrent_state.zero_()


# ---------------------------------------------------------------------------
# Feed-Forward Network (SwiGLU)
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Decoder Layer (dispatches to DeltaNet or Attention based on layer_type)
# ---------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]

        if self.layer_type == "linear_attention":
            self.token_mixer = GatedDeltaNet(config, layer_idx)
        else:
            self.token_mixer = GatedAttention(config, layer_idx)

        self.ffn = FeedForward(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(
        self,
        x: torch.Tensor,
        input_pos: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> torch.Tensor:
        # Token mixer
        normed = self.input_layernorm(x)
        if self.layer_type == "linear_attention":
            h = self.token_mixer(normed)
        else:
            h = self.token_mixer(normed, input_pos, freqs_cos, freqs_sin)
        x = x + h

        # FFN
        x = x + self.ffn(self.post_attention_layernorm(x))
        return x


# ---------------------------------------------------------------------------
# Full Qwen3.5 Model
# ---------------------------------------------------------------------------

class Qwen3_5Model(nn.Module):
    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rope = RotaryEmbedding(config)

        # Precompute RoPE frequencies
        freqs_cos, freqs_sin = self.rope(config.max_seq_len, torch.device("cpu"))
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self, tokens: torch.Tensor, input_pos: torch.Tensor
    ) -> torch.Tensor:
        x = self.tok_embeddings(tokens)

        for layer in self.layers:
            x = layer(x, input_pos, self.freqs_cos, self.freqs_sin)

        x = self.norm(x)
        return self.output(x)

    def reset_caches(self):
        for layer in self.layers:
            if hasattr(layer.token_mixer, "reset_cache"):
                layer.token_mixer.reset_cache()
            if hasattr(layer.token_mixer, "kv_cache"):
                layer.token_mixer.kv_cache.reset()


# ---------------------------------------------------------------------------
# Model Loading from HuggingFace Safetensors
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_dir: str,
    config: Qwen3_5Config,
    device: str = "cpu",
) -> Qwen3_5Model:
    """Load Qwen3.5 model from HuggingFace safetensors checkpoint."""
    from safetensors import safe_open

    checkpoint_path = Path(checkpoint_dir)

    # Build model on meta device to avoid memory doubling
    with torch.device("meta"):
        model = Qwen3_5Model(config)

    # Find safetensors files
    st_files = sorted(checkpoint_path.glob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(
            f"No safetensors files found in {checkpoint_dir}"
        )

    # Import weight conversion
    from executorch.examples.models.qwen3_5.convert_weights import (
        convert_weights,
    )

    # Load and convert weights
    state_dict = {}
    for st_file in st_files:
        with safe_open(str(st_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key).to(torch.float32)

    converted = convert_weights(state_dict, config)

    # Load with assign=True to avoid doubling memory
    model.load_state_dict(converted, strict=False, assign=True)

    # Materialize any remaining meta tensors (buffers like caches)
    for name, buf in model.named_buffers():
        if buf.device == torch.device("meta"):
            model_part = model
            parts = name.split(".")
            for part in parts[:-1]:
                model_part = getattr(model_part, part)
            setattr(
                model_part,
                parts[-1],
                torch.zeros_like(buf, device="cpu"),
            )

    # Recompute RoPE
    freqs_cos, freqs_sin = model.rope(config.max_seq_len, torch.device("cpu"))
    model.freqs_cos = freqs_cos
    model.freqs_sin = freqs_sin

    return model.to(device)
