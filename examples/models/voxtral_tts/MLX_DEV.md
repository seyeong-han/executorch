# Voxtral TTS — MLX Backend Enablement Handover

**Branch:** `voxtral-tts-dev`
**Target device:** Apple Silicon Mac (M1/M2/M3/M4 Pro / Max / Ultra)
**Reference implementations:** `examples/models/voxtral_realtime/` (MLX + Metal both done) and `examples/models/qwen3_5_moe/` (MLX done)
**Expected performance:** M3 Max with 4w quant — LM RTF ~1.5–2x (fast, not yet sub-real-time); M4 Max could approach 1x.

---

## How MLX differs from Metal and CUDA

| | CUDA | Metal | MLX |
|---|---|---|---|
| Runtime | AOTI Triton kernel | AOTI Metal kernel | MLX delegate |
| SDPA op | `torch.ops.triton.sdpa(q, k, v, mask, ...)` | `aten._scaled_dot_product_attention_math_for_mps(q, k, v, mask, ...)` | `torch.ops.mlx.custom_sdpa(q, k, v, start_pos=, is_causal=True, scale=)` |
| KV cache | `StaticKVCache` (bf16, BHSD, `index_copy_`) | `StaticKVCache` (fp32, BHSD, `index_copy_`) | `MLXStaticKVCache` wrapping `executorch.backends.mlx.llm.cache.KVCache` |
| Causal mask | Explicit `[1, 1, T_q, max_seq_len]` bool mask — YOU build it | Additive `(valid - 1) * 1e9` fp mask — YOU build it | **Not needed** — `custom_sdpa` handles causal masking via `start_pos` |
| RoPE | `apply_rotary_emb(q, k, freqs_cos, freqs_sin)` (any backend) | Same | `torch.ops.mlx.rope(q, dim, start_pos, traditional=True, base=theta)` — on-the-fly, no table lookup |
| Partitioner | `CudaPartitioner` | `MetalPartitioner` | `MLXPartitioner` from `executorch.backends.mlx.partitioner` |
| Quantization | `--qlinear 4w` (tile_packed_to_4d) | `--qlinear fpa4w` (Apple native 4-bit) | `--qlinear 4w` (same as CUDA) |
| macOS only | No | Yes (`condition: Darwin`) | Yes (`condition: Darwin`) |

---

## Phase checklist

```
Phase M-1  model.py: MLXStaticKVCache + MLXSDPA + mlx.rope branch
               └─ eager parity gate (CPU fp32 vs MLX fp32)
Phase M-2  export_voxtral_tts.py: --backend mlx lowering
Phase M-3  CMakePresets.json: voxtral-tts-mlx preset
Phase M-4  Makefile: voxtral_tts-mlx target
Phase M-5  run_mlx_e2e.sh: one-shot script
Phase M-6  Parity gate: XNNPACK fp32 vs MLX .pte
Phase M-7  BENCHMARK.md: Apple Silicon numbers
```

---

## Phase M-1 — `model.py` changes

File: `examples/models/voxtral_tts/model.py`

### M-1a: Add `MLXStaticKVCache` after `StaticKVCache`

Paste this class after the `StaticKVCache` class (currently around line 305):

```python
class MLXStaticKVCache(nn.Module):
    """KV cache wrapping executorch.backends.mlx.llm.cache.KVCache.

    MLX's KVCache uses BHSD layout. The model's QKV projections arrive in
    BSHD, so this wrapper transposes on entry and returns BHSD for MLXSDPA.
    """

    def __init__(
        self, max_seq_len: int, n_kv_heads: int, head_dim: int, dtype: torch.dtype
    ):
        super().__init__()
        from executorch.backends.mlx.llm.cache import KVCache as MLXKVCacheImpl
        self.cache = MLXKVCacheImpl(
            max_batch_size=1,
            max_context_length=max_seq_len,
            n_heads=n_kv_heads,
            head_dim=head_dim,
            enable_dynamic_shape=True,
            dtype=dtype,
        )

    def update(
        self, input_pos: torch.Tensor, k_val: torch.Tensor, v_val: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k_val = k_val.transpose(1, 2)  # BSHD → BHSD
        v_val = v_val.transpose(1, 2)
        return self.cache.update(input_pos, k_val, v_val)
```

### M-1b: Add `MLXSDPA` after `StandardSDPA`

Paste after the `StandardSDPA` class (currently around line 395):

```python
class MLXSDPA(nn.Module):
    """SDPA using MLX's custom op for Apple Silicon GPU acceleration.

    torch.ops.mlx.custom_sdpa handles GQA expansion, causal masking via
    start_pos, and on-device execution. No explicit mask needed.
    KV cache is in BHSD layout; queries arrive in BSHD.
    """

    def __init__(self, n_heads: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dim = n_heads * head_dim
        self.scale = head_dim**-0.5

    def forward(
        self,
        input_pos: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bsz: int,
        seqlen: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        start_pos = input_pos[0].item()
        q = q.transpose(1, 2)          # BSHD → BHSD for custom_sdpa
        y = torch.ops.mlx.custom_sdpa(
            q, k, v,
            start_pos=start_pos,
            is_causal=True,
            scale=self.scale,
        )
        return y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
```

### M-1c: Add the `mlx` branch in `LMAttention.__init__`

In `LMAttention.__init__`, after the existing `elif self.config.backend == "cuda":` block, add:

```python
elif self.config.backend == "mlx":
    self.kv_cache = MLXStaticKVCache(
        config.max_seq_len, self.n_kv_heads, self.head_dim, dtype=torch.float16
    )
    self.sdpa = MLXSDPA(self.n_heads, self.n_kv_heads, self.head_dim)
```

Note: MLX uses fp16 (not bf16) for the KV cache by default. If the model is loaded with `dtype=torch.float16`, this matches automatically. Check the checkpoint dtype — Mistral 4B TTS ships as bfloat16, so pass `dtype=torch.bfloat16` if you keep the model in bf16 throughout.

### M-1d: Add the `mlx` RoPE branch in `LMAttention.forward`

In `LMAttention.forward`, the RoPE computation currently calls `apply_rotary_emb(q, k, freqs_cos, freqs_sin)`. Add an MLX branch before that call:

```python
if self.config.backend == "mlx":
    start_pos = input_pos[0].item()
    q = torch.ops.mlx.rope(
        q.transpose(1, 2),          # BSHD → BHSD for mlx.rope
        self.head_dim,
        start_pos,
        traditional=True,
        base=self.config.rope_theta,
    ).transpose(1, 2)              # back to BSHD
    k = torch.ops.mlx.rope(
        k.transpose(1, 2),
        self.head_dim,
        start_pos,
        traditional=True,
        base=self.config.rope_theta,
    ).transpose(1, 2)
else:
    freqs_cos = freqs_cos[input_pos]
    freqs_sin = freqs_sin[input_pos]
    q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)
```

The `traditional=True` flag matches Mistral's interleaved RoPE format. Using `mlx.rope` avoids storing the full `[max_seq_len, head_dim/2]` freqs_cos/freqs_sin buffers and computes positions on-the-fly from `start_pos`.

### M-1e: Update `MistralDecoder.forward` — no mask needed for MLX

The CUDA path builds `_build_causal_mask_bool`. For MLX, `custom_sdpa` handles causality via `start_pos`, so pass `attn_mask=None`:

```python
# In MistralDecoder.forward, update the existing mask logic:
attn_mask = None
if self.config.backend == "cuda":
    attn_mask = _build_causal_mask_bool(
        input_pos, self.config.max_seq_len, input_embeds.device
    )
elif self.config.backend == "metal":
    attn_mask = _build_attn_mask(
        input_pos, self.config.max_seq_len,
        input_embeds.device, input_embeds.dtype
    )
# else: mlx, xnnpack, portable — mask stays None
```

### M-1f: Add `"mlx"` to `load_model` backend validation

In `load_model`, update any backend validation to include `"mlx"`:
```python
# Wherever backend choices are validated, add "mlx"
assert backend in ("portable", "xnnpack", "cuda", "metal", "mlx"), ...
```

---

## Phase M-2 — `export_voxtral_tts.py` changes

File: `examples/models/voxtral_tts/export_voxtral_tts.py`

### M-2a: Add `mlx` to argparse backend choices

```python
parser.add_argument(
    "--backend",
    choices=["portable", "xnnpack", "cuda", "cuda-windows", "metal", "mlx"],  # add "mlx"
    ...
)
```

### M-2b: Add MLX lowering branch in `lower_to_executorch`

In the `lower_to_executorch` function, add after the existing Metal branch:

```python
elif backend == "mlx":
    from executorch.backends.mlx.partitioner import MLXPartitioner
    from executorch.backends.mlx.passes import get_default_passes

    print(f"\nLowering to ExecuTorch with MLX ({len(programs)} methods)...")
    partitioner = {key: [MLXPartitioner()] for key in programs}
    transform_passes = get_default_passes()
```

And update the `lower_to_executorch` call site to pass `transform_passes` through when MLX is selected. See `examples/models/qwen3_5_moe/export.py:_export_mlx` for the exact pattern.

### M-2c: Add Darwin condition for MLX export

Add a validation at the top of `main()` next to the CUDA arg defaults:

```python
if backend_for_export == "mlx" and sys.platform != "darwin":
    parser.error("--backend=mlx requires macOS (Apple Silicon)")
```

### M-2d: Sample tensors for MLX export must be on CPU

Unlike Metal (where `Dim.AUTO` is used to avoid bounds issues), MLX works fine with explicit sequence-length bounds. Keep the existing export_model dynamic shapes as-is — they should work for MLX unchanged.

### M-2e: Codec export for MLX

The conv-as-matmul rewrite from the CUDA work (`_conv1d_as_matmul`, `_conv_transpose1d_as_matmul`) lowers cleanly to MLX matmul ops. No additional changes needed for the codec.

Add the MLX codec export path in `_export_codec_pte`:

```python
# In _export_codec_pte, update the triton_mode logic:
codec_triton_mode = "OFF" if args.backend in ("cuda", "cuda-windows", "metal") else "ON"
# For mlx, triton_mode is irrelevant — MLXPartitioner handles it.
codec_backend = args.backend
```

### M-2f: Recommended export command (with 4w quantization)

```bash
python export_voxtral_tts.py \
    --model-path ~/models/Voxtral-4B-TTS-2603 \
    --backend mlx \
    --dtype bf16 \
    --qlinear 4w \
    --qembedding 8w \
    --output-dir ./voxtral_tts_exports_mlx
```

**Why these flags:**
- `--dtype bf16` — MLX runs faster in bf16; Apple Silicon memory bandwidth is precious
- `--qlinear 4w` — halves the linear layer weight footprint with minimal quality loss
- `--qembedding 8w` — reduces token embedding table (131K × 3072 = 1.5 GB) to 8-bit; embedding lookup is MLX-friendly at 8-bit

**Expected output sizes (4w + 8w embedding):**
- `model.pte` ≈ 5 MB (stub)
- `model.ptd` ≈ 2.5–3 GB (weights + MLX delegate data)
- `codec_decoder.pte` ≈ 5 MB
- `codec_decoder.ptd` ≈ 280 MB

---

## Phase M-3 — `CMakePresets.json` additions

File: `examples/models/voxtral_tts/CMakePresets.json`

Add these blocks following the existing `voxtral-tts-cuda` preset pattern:

```json
{
    "name": "voxtral-tts-mlx",
    "displayName": "Voxtral TTS runner (MLX)",
    "inherits": ["voxtral-tts-base"],
    "cacheVariables": {
        "EXECUTORCH_BUILD_MLX": "ON"
    },
    "condition": {
        "lhs": "${hostSystemName}",
        "type": "equals",
        "rhs": "Darwin"
    }
}
```

In `buildPresets`:
```json
{
    "name": "voxtral-tts-mlx",
    "displayName": "Build Voxtral TTS runner (MLX)",
    "configurePreset": "voxtral-tts-mlx",
    "configuration": "Release",
    "targets": ["voxtral_tts_runner"]
}
```

In `workflowPresets`:
```json
{
    "name": "voxtral-tts-mlx",
    "displayName": "Configure and build Voxtral TTS runner (MLX)",
    "steps": [
        {"type": "configure", "name": "voxtral-tts-mlx"},
        {"type": "build",     "name": "voxtral-tts-mlx"}
    ]
}
```

---

## Phase M-4 — `Makefile` targets

File: `Makefile` (repo root)

Add after `voxtral_tts-cuda`:

```makefile
voxtral_tts-mlx:
	@echo "==> Building and installing ExecuTorch with MLX..."
	cmake --workflow --preset mlx-release
	@echo "==> Building Voxtral TTS runner with MLX..."
	cd examples/models/voxtral_tts && cmake --workflow --preset voxtral-tts-mlx
	@echo ""
	@echo "✓ Build complete!"
	@echo "  Binary: cmake-out/examples/models/voxtral_tts/voxtral_tts_runner"
```

Add `voxtral_tts-mlx` to the `.PHONY` line and the `help` echo block.

---

## Phase M-5 — `run_mlx_e2e.sh`

Create `examples/models/voxtral_tts/run_mlx_e2e.sh` following the pattern of `run_cuda_e2e.sh`, but:
- Replace `llm-release-cuda` with `mlx-release`
- Replace `voxtral-tts-cuda` with `voxtral-tts-mlx`
- Remove the `unset CPATH` / CUDA env var guards (not needed on macOS)
- Export defaults to `--backend mlx --dtype bf16 --qlinear 4w --qembedding 8w`
- Runner does not need `--data_path` if the `.pte` embeds weights (check whether `et_prog._tensor_data` is populated for MLX exports and handle `--codec_data_path` the same way as the CUDA path)

---

## Phase M-6 — Tests

### M-6a: Eager parity (runs on-device before export)

```bash
# On the Mac, from the executorch repo root:
export VOXTRAL_TTS_MODEL_DIR=~/models/Voxtral-4B-TTS-2603
pytest -xvs examples/models/voxtral_tts/test_metal_parity.py
```

`test_metal_parity.py` already has the prefill cosine and semantic argmax tests. The
`models()` fixture will skip with a clear message until `load_model(..., backend="mlx")`
is wired in. Gate: cosine ≥ 0.998, argmax identical.

### M-6b: Export smoke test

```bash
python examples/models/voxtral_tts/export_voxtral_tts.py \
    --model-path ~/models/Voxtral-4B-TTS-2603 \
    --backend mlx \
    --dtype bf16 \
    --qlinear 4w \
    --export-target model \        # test LM only first, codec later
    --output-dir /tmp/voxtral_tts_mlx_test

# Verify file size is sane (not 0 bytes or absurdly small)
ls -lh /tmp/voxtral_tts_mlx_test/
```

### M-6c: Runner smoke test

```bash
make voxtral_tts-mlx

cmake-out/examples/models/voxtral_tts/voxtral_tts_runner \
    --model /tmp/voxtral_tts_mlx_test/model.pte \
    --codec /tmp/voxtral_tts_mlx_test/codec_decoder.pte \
    --tokenizer ~/models/Voxtral-4B-TTS-2603/tekken.json \
    --voice ~/models/Voxtral-4B-TTS-2603/voice_embedding/neutral_female.pt \
    --text "Hello, how are you today?" \
    --output /tmp/mlx_test.wav \
    --seed 42 \
    --max_new_tokens 100
```

Gate: `Generated N frames` in output (not END_AUDIO at frame 0), audio RMS > 0.01, wall clock reasonable for device.

### M-6d: Parity vs XNNPACK baseline

```bash
# Run both and compare frame codes
VOXTRAL_DUMP_CODES=/tmp/xnnpack_codes.txt cmake-out/.../voxtral_tts_runner \
    --model ./voxtral_tts_exports/model.pte \
    --codec ./voxtral_tts_exports/codec_decoder.pte \
    ...

VOXTRAL_DUMP_CODES=/tmp/mlx_codes.txt cmake-out/.../voxtral_tts_runner \
    --model ./voxtral_tts_exports_mlx/model.pte \
    --codec ./voxtral_tts_exports_mlx/codec_decoder.pte \
    ...

diff /tmp/xnnpack_codes.txt /tmp/mlx_codes.txt
```

Acceptable: first semantic code matches; subsequent codes may drift due to bf16 sampling noise. Frame count should be within ±5 of the XNNPACK fp32 baseline (40 frames for "Hello, how are you today?").

---

## Gotchas and MLX-specific pitfalls

### 1. `mlx.rope` uses `traditional=True` — this matters

Mistral uses interleaved (traditional) RoPE layout. `traditional=False` would use the PyTorch split-rotate layout and produce garbage outputs. Always pass `traditional=True` when calling `torch.ops.mlx.rope`.

### 2. MLX KV cache dtype

`MLXStaticKVCache` wraps `executorch.backends.mlx.llm.cache.KVCache`. The `dtype` passed to it must match what the model is running in (bf16 or fp16). If you export with `--dtype bf16`, pass `dtype=torch.bfloat16`. If you see shape mismatch errors at KV cache update time, this is likely the culprit.

### 3. `enable_dynamic_shape=True` is required

Without it, the MLX KV cache pre-allocates exactly one shape and errors on any other sequence length during prefill. Always set it.

### 4. `get_default_passes()` from `executorch.backends.mlx.passes` is mandatory

The MLX backend requires a set of pre-lowering graph passes (reshape/transpose canonicalization) before `MLXPartitioner` runs. Always call `transform_passes=get_default_passes()` in `to_edge_transform_and_lower`.

### 5. Dynamic shapes and `Dim.AUTO` on MLX

MLX handles explicit sequence-length bounds (`Dim("seq_len", min=1, max=max_seq_len)`) correctly. Do NOT use `Dim.AUTO` for MLX — it's only needed for Metal to avoid a Metal AOTI bounds-inference bug. The existing TTS export code already uses explicit bounds; leave them as-is.

### 6. `.pte` may or may not embed weights

For CUDA, `et_prog._tensor_data` is populated with the `.ptd`. For MLX, check at export time whether weights are embedded in the `.pte` or written separately. If `et_prog._tensor_data` is non-empty, write it out and support `--data_path` in the runner. If empty, the `.pte` is self-contained (common for small MLX exports).

### 7. Linear-with-bias decomposition (Metal needs it, MLX may not)

Metal AOTI errors on linear ops with bias (stride-0 reinterpret issue). MLX typically does not need `_linear_bias_decomposition`. Test the export without it first; only add if you see a lowering error mentioning "linear with bias".

### 8. Flow matching head (`predict_velocity`) — 7 ODE steps × 2 calls per frame

The predict_velocity method is called 14 times per audio frame. On MLX, each dispatch has a small overhead. If the method is exported as a separate .pte method (as in the current 5-method setup), each call incurs MLX delegate launch cost. Options:
- Keep 5-method setup and accept the overhead (simplest, correctness-first)
- Fuse predict_velocity loop into a single exported method that runs all 7 steps (advanced optimization, significant model.py surgery)

Start with option 1; revisit if profiling shows MLX dispatch is the bottleneck.

---

## Best performance strategy for MLX

Based on the realtime and qwen MLX work, ranked by impact:

### 1. bf16 + 4w linear quantization (biggest win)

```bash
--dtype bf16 --qlinear 4w --qembedding 8w
```

MLX's GPU matrix multiply is fastest in bf16. 4w quantization halves the linear weight memory bandwidth, which is the dominant cost at `seqlen=1` decode steps. This alone accounts for ~2–3× speedup vs fp32 unquantized.

### 2. `mlx.rope` instead of precomputed freqs table

On-the-fly RoPE via `torch.ops.mlx.rope` avoids fetching a `[max_seq_len, dim]` buffer from metal memory every step. The op is fused on the MLX GPU. Small win (~5%) but zero implementation cost.

### 3. `mlx.custom_sdpa` with `start_pos` (correctness + performance)

Using `custom_sdpa` with `start_pos` instead of building an explicit mask means the MLX kernel only computes attention over the populated cache prefix — it doesn't compute and discard attention on unwritten zero slots. For long sequences this matters more than short.

### 4. Single-method decode loop (advanced, post-validation only)

The runner currently calls:
```
audio_token_embedding → text_decoder → semantic_head → predict_velocity (×14) → ...
```
That's 17 cross-boundary dispatch calls per frame. If MLX dispatch overhead is measurable (profile with `executorch.extension.profiler`), consider exporting a `generate_frame` method that runs one full LM + ODE step. This is a significant refactor — do not attempt until parity is confirmed.

### 5. Codec on MLX via conv-as-matmul

The `_conv1d_as_matmul` / `_conv_transpose1d_as_matmul` rewrite from the CUDA work lowers cleanly to MLX matmul. MLX's `mlx.core.matmul` is highly optimized on Apple Silicon. The codec should be fast. If it's slow, profile to confirm which layers are bottlenecks; the `ConvTranspose1d` up-samplers are the likely candidates.

### 6. Memory bandwidth is king on Apple Silicon

Apple Silicon has unified memory — CPU and GPU share the same pool. The 4B model in bf16 = ~8 GB. M3 Pro (18 GB), M3 Max (36–128 GB), M4 Max (48–128 GB). With 4w quant the model fits even in 18 GB alongside the OS. Fitting in cache matters more than arithmetic throughput for decode-heavy workloads like TTS.

---

## Expected benchmark targets

Fill these in after you have a working MLX runner on device:

| Device | Config | LM time (3s audio) | Codec time | Total | E2E RTF |
|---|---|---|---|---|---|
| M3 Pro 18GB | bf16 + 4w | TBD | TBD | TBD | TBD |
| M3 Max 36GB | bf16 + 4w | TBD | TBD | TBD | TBD |
| M4 Max 48GB | bf16 + 4w | TBD | TBD | TBD | TBD |

For reference: CUDA A100 fp32 LM = 11.5 s, CUDA A100 4w LM = 2.1 s (RTF 0.88x total).
Apple Silicon M3 Max is comparable to an A10G per FP16 TFLOPS.
Conservative estimate for M3 Max 4w: LM ~4–6 s → E2E RTF ~2–3x (not sub-realtime, but usable).
M4 Max could push toward RTF ~1.5x.

---

## File change summary

| File | Status | Change |
|---|---|---|
| `model.py` | Modify | Add `MLXStaticKVCache`, `MLXSDPA`; add `mlx` branches in `LMAttention.__init__` and `LMAttention.forward`; update `MistralDecoder.forward` mask logic |
| `export_voxtral_tts.py` | Modify | Add `mlx` to `--backend` choices; add MLX lowering branch in `lower_to_executorch`; update `_apply_cuda_arg_defaults` to not error on MLX |
| `CMakePresets.json` | Modify | Add `voxtral-tts-mlx` configure/build/workflow presets |
| `Makefile` | Modify | Add `voxtral_tts-mlx` target and `.PHONY` + help entries |
| `run_mlx_e2e.sh` | Create | One-shot Mac export + build + run script |
| `test_metal_parity.py` | Modify | `models()` fixture already stubs the MLX path (`backend="mlx"`); fill in test bodies once M-1 lands |

---

## Order of operations

```
On the Mac:
  1. pip install -e . --no-build-isolation (from the executorch repo root)
  2. conda install mlx  (or pip install mlx if not using conda)
  3. Implement M-1 (model.py) → run test_metal_parity.py eager tests
  4. Implement M-2 (export) → test_metal_parity.py still passes? Run export smoke test
  5. Implement M-3 (CMake) + M-4 (Makefile)
  6. make voxtral_tts-mlx
  7. M-6c runner smoke test → M-6d XNNPACK parity check
  8. M-5 run_mlx_e2e.sh
  9. M-7 fill in BENCHMARK.md
```

Stop at each gate. Don't proceed to the next phase if tests fail.

---

## Reference reading

- `examples/models/voxtral_realtime/model.py` — `MLXStaticKVCache`, `MLXSDPA`, `MLXMaskedSDPA`, `mlx.rope` usage
- `examples/models/voxtral_realtime/export_voxtral_rt.py` — `lower_to_executorch` MLX branch, `_linear_bias_decomposition`, `get_default_passes()`
- `examples/models/qwen3_5_moe/export.py` — `_export_mlx`, `_prepare_and_quantize_mlx`
- `examples/models/qwen3_5_moe/mlx_source_transformations.py` — pattern for MLX source transforms if any ops need replacing
- `executorch/backends/mlx/partitioner.py` — `MLXPartitioner` internals
- `executorch/backends/mlx/llm/cache.py` — `KVCache`, `RingBufferKVCache` implementations
- `executorch/backends/mlx/passes.py` — `get_default_passes()` contents (important for understanding what graph transforms run before lowering)
