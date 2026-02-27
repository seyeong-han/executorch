# Qwen3.5 ExecuTorch Export -- Research Context

## Model Identity

Qwen3.5 is registered as `qwen3_next` in HuggingFace Transformers (v5.0.0rc1). The `model_type` is `"qwen3_next"`, NOT `"qwen3_5"`. The model classes are `Qwen3NextForCausalLM`, `Qwen3NextModel`, etc.

## Architecture Overview

Qwen3.5 uses a **hybrid architecture** combining two distinct layer types:

1. **Gated DeltaNet** (linear attention) -- a recurrent linear attention mechanism
2. **Gated Attention** (full attention) -- standard multi-head attention with sigmoid output gating

The layers are interleaved in a fixed repeating pattern: every 4th layer is full attention, the other 3 are Gated DeltaNet. This is controlled by `full_attention_interval=4` in the config.

### Qwen3.5-27B (Dense)

- **27B parameters**, hidden_size=5120, 64 layers, vocab=248320
- Layout: `16 x (3 x GatedDeltaNet + FFN → 1 x GatedAttention + FFN)` = 48 DeltaNet + 16 Attention
- Dense FFN (intermediate_size=17408) for all layers -- no MoE
- Context length: 262,144 tokens natively

### Qwen3.5-35B-A3B (MoE)

- **35B total / 3B active**, hidden_size=2048, 40 layers, vocab=248320
- Layout: `10 x (3 x GatedDeltaNet + MoE → 1 x GatedAttention + MoE)`
- MoE: 256 experts, 8 routed + 1 shared active, expert_intermediate=512
- All 35B params must reside in memory (MoE does not reduce memory footprint)

### Why 27B for 36GB M3 Pro

| Variant | INT4 Size | Fits? | Headroom |
|---------|----------|-------|----------|
| 27B dense INT4 | ~14 GB | Yes | ~18 GB for KV cache + system |
| 35B-A3B INT4 | ~18 GB | Yes | ~14 GB, but MoE adds export complexity |

The 27B dense model is chosen because:
- No MoE routing (data-dependent expert loops break `torch.export`)
- Simpler architecture to export
- More memory headroom on 36GB

## Full Attention Layer Details

Each of the 16 attention layers uses:
- **Q projection**: hidden_size → num_attention_heads * head_dim * 2 (doubled for gate)
  - The output is split into `query_states` and `gate` via `torch.chunk(..., 2, dim=-1)`
- **K projection**: hidden_size → num_key_value_heads * head_dim
- **V projection**: hidden_size → num_key_value_heads * head_dim
- **O projection**: num_attention_heads * head_dim → hidden_size

With 27B defaults: 24 Q heads, 4 KV heads, head_dim=256, hidden=5120

Key features:
- **QK Norms**: Per-head RMSNorm on Q and K (note: Qwen3Next RMSNorm uses `(1 + weight)` formulation)
- **Partial RoPE**: `partial_rotary_factor=0.25`, so only 64 of 256 head dims get rotary embeddings
- **Sigmoid gating**: `attn_output = attn_output * sigmoid(gate)` before `o_proj`
- **GQA**: 24 Q heads / 4 KV heads = 6x GQA ratio

### Partial RoPE Implementation

```python
rotary_dim = cos.shape[-1]  # = head_dim * partial_rotary_factor = 256 * 0.25 = 64
q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
q_embed = torch.cat([q_embed, q_pass], dim=-1)
k_embed = torch.cat([k_embed, k_pass], dim=-1)
```

## Gated DeltaNet Layer Details

Each of the 48 DeltaNet layers uses:
- **in_proj_qkvz**: hidden_size → key_dim*2 + value_dim*2 (Q, K, V, and Z gate)
  - With defaults: 5120 → 128*16*2 + 128*32*2 = 4096 + 8192 = 12288
- **in_proj_ba**: hidden_size → num_v_heads * 2 (beta and alpha)
  - With defaults: 5120 → 32*2 = 64
- **conv1d**: Depthwise Conv1d, in/out=conv_dim=key_dim*2+value_dim, kernel=4, groups=conv_dim
  - With defaults: conv_dim = 4096 + 4096 = 8192, kernel=4
- **dt_bias**: Parameter shape (num_v_heads,) = (32,)
- **A_log**: Parameter shape (num_v_heads,) = (32,)
- **norm**: RMSNormGated (hidden_size=head_v_dim=128)
- **out_proj**: value_dim → hidden_size (4096 → 5120)

### State Shapes

- **conv_state**: `(batch, conv_dim, conv_kernel_size)` = `(1, 8192, 4)` -- last 4 timesteps for conv1d
- **recurrent_state**: `(batch, num_v_heads, key_head_dim, value_head_dim)` = `(1, 32, 128, 128)` -- the "KV cache" equivalent

Total state per DeltaNet layer: 8192*4*4 + 32*128*128*4 ≈ 131KB + 2MB ≈ 2.1 MB
Total state for 48 layers: ~101 MB (very manageable)

### Decode Step (seq_len=1) -- Core Recurrence

```python
# Single-step recurrence (from torch_recurrent_gated_delta_rule)
q_t, k_t, v_t = query[:,:,0], key[:,:,0], value[:,:,0]
g_t = g[:,:,0].exp().unsqueeze(-1).unsqueeze(-1)  # decay gate
beta_t = beta[:,:,0].unsqueeze(-1)                  # update gate

# 1. Decay the recurrent state
recurrent_state = recurrent_state * g_t

# 2. Read from memory
kv_mem = (recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)

# 3. Compute delta (error-corrected update)
delta = (v_t - kv_mem) * beta_t

# 4. Write to memory
recurrent_state = recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)

# 5. Query the memory
output = (recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
```

### Prefill (variable seq_len) -- Chunked Computation

The prefill uses `torch_chunk_gated_delta_rule` which:
1. Pads sequence to multiple of chunk_size=64
2. Computes cumulative decay within chunks
3. Iterates over chunks with Python for-loop (NOT exportable)
4. Returns final recurrent state

**Export strategy**: We implement decode-only mode (seq_len=1 per call). Prefill is done by iterating one token at a time. This is slower than chunked prefill but fully exportable.

## FFN Layer

Standard SwiGLU: `down_proj(silu(gate_proj(x)) * up_proj(x))`
- gate_proj: hidden_size → intermediate_size (5120 → 17408)
- up_proj: hidden_size → intermediate_size (5120 → 17408)
- down_proj: intermediate_size → hidden_size (17408 → 5120)

## RMSNorm Variants

Qwen3Next has a unique RMSNorm that uses `(1 + weight)` instead of just `weight`:
```python
output = rms_norm(x) * (1.0 + weight)
```

RMSNormGated combines RMSNorm with SiLU gating:
```python
output = rms_norm(x) * silu(gate)
```

## Metal Backend Constraints

### SDPA head_dim Limitation

The Metal SDPA kernel (`op_sdpa.mm`) only supports head_dim 64, 96, 128.
Qwen3.5-27B uses head_dim=256 for full attention -- **NOT supported**.

Mitigation strategies:
1. Let AOTI decompose SDPA into matmul+softmax+matmul (Metal handles mm_out natively)
2. Implement decomposed attention explicitly in the model
3. Extend the Metal SDPA kernel to support head_dim=256 (best performance)

### No Bool Tensors on Metal

Metal AOTI cannot allocate bool tensors on MPS. Attention masks must use integer/float arithmetic.
Pattern from voxtral_realtime:
```python
diff = input_pos.unsqueeze(1) - k_pos.unsqueeze(0) + 1
valid = torch.clamp(diff, min=0, max=1)
mask = (valid.float() - 1.0) * 1e9
```

### Linear Bias Decomposition

All Metal exports must decompose `aten.linear.default` with bias to avoid `reinterpret_tensor_wrapper` producing 0-stride tensors:
```python
def _linear_bias_decomposition(input, weight, bias=None):
    weight_t = torch.ops.aten.t.default(weight)
    out = torch.ops.aten.matmul.default(input, weight_t)
    if bias is not None:
        return torch.ops.aten.add.Tensor(out, bias)
    return out
```

### Metal Quantization

- `fpa4w`: Floating-point activation, 4-bit weight (UIntxWeightOnlyConfig with HQQ)
- Requires torchao built with `TORCHAO_BUILD_EXPERIMENTAL_MPS=1`
- Every linear layer's weight K dimension must be divisible by group_size

## Precedent Models Studied

### LFM2 (Hybrid Architecture)

- Mixes `conv` and `full_attention` layers via `layer_types` config
- `ShortConvBlock` uses `register_buffer("conv_state", ...)` for recurrent state
- State updated with `copy_()` inside `torch.no_grad()` -- proven exportable
- Same `forward(x, freqs_cos, freqs_sin, attn_options)` signature as TransformerBlock

### Voxtral Realtime (Metal Export)

- Multi-method export: audio_encoder, text_decoder, token_embedding
- `StaticKVCache` with `[B,H,S,D]` layout and `index_copy_` for Metal
- `MetalSDPA` calling `torch.ops.aten._scaled_dot_product_attention_math_for_mps`
- Meta-device loading + safetensors lazy access for memory efficiency
- `_linear_bias_decomposition` for Metal AOTI compatibility

### Parakeet (Metal Export)

- 4-method export with preprocessor on CPU, encoder/decoder/joint on Metal
- Same `_linear_bias_decomposition` and `MetalPartitioner` patterns
- BFloat16 support for Metal export

## Weight Name Mapping (HF → Custom)

Key weight names in HuggingFace format:
```
model.embed_tokens.weight
model.layers.{i}.input_layernorm.weight
model.layers.{i}.post_attention_layernorm.weight

# Full attention layers:
model.layers.{i}.self_attn.q_proj.weight    # (num_heads * head_dim * 2, hidden)
model.layers.{i}.self_attn.k_proj.weight    # (num_kv_heads * head_dim, hidden)
model.layers.{i}.self_attn.v_proj.weight    # (num_kv_heads * head_dim, hidden)
model.layers.{i}.self_attn.o_proj.weight    # (hidden, num_heads * head_dim)
model.layers.{i}.self_attn.q_norm.weight    # (head_dim,)
model.layers.{i}.self_attn.k_norm.weight    # (head_dim,)

# DeltaNet layers:
model.layers.{i}.linear_attn.in_proj_qkvz.weight  # (key_dim*2 + value_dim*2, hidden)
model.layers.{i}.linear_attn.in_proj_ba.weight     # (num_v_heads*2, hidden)
model.layers.{i}.linear_attn.conv1d.weight         # (conv_dim, 1, kernel)
model.layers.{i}.linear_attn.dt_bias               # (num_v_heads,)
model.layers.{i}.linear_attn.A_log                 # (num_v_heads,)
model.layers.{i}.linear_attn.norm.weight            # (head_v_dim,)
model.layers.{i}.linear_attn.out_proj.weight        # (hidden, value_dim)

# FFN (dense):
model.layers.{i}.mlp.gate_proj.weight
model.layers.{i}.mlp.up_proj.weight
model.layers.{i}.mlp.down_proj.weight

model.norm.weight
lm_head.weight  # (vocab, hidden) -- NOT tied to embed_tokens
```

## Export Architecture Decision

We use a **custom export script** (like voxtral_realtime/parakeet) rather than the native LLM pipeline or optimum-executorch because:

1. `Qwen3NextDynamicCache.is_compileable = False` blocks HF's `TorchExportableModuleForDecoderOnlyLM`
2. The hybrid DeltaNet+Attention architecture needs custom state management
3. We need full control over Metal-specific attention implementations
4. The native pipeline's `llama_transformer.py` would need significant modifications

The model is built from scratch with export-compatible patterns:
- Registered buffers for all state (KV cache, conv_state, recurrent_state)
- In-place `copy_()` updates inside `torch.no_grad()`
- Decode-only mode (seq_len=1) to avoid chunked recurrence loops
- Backend-switchable attention (XNNPACK vs Metal patterns)
