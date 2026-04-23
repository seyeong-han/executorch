# Voxtral TTS — Metal backend development

This branch (`voxtral-tts-dev`) holds in-progress work bringing the Apple Metal
backend to voxtral_tts. The CUDA work shipped on the parent branch
(`voxtral-tts`) — Metal mirrors most of that pattern but needs to be built
and validated on a real Mac (M-series Apple Silicon).

## What's here

- `METAL_DEV.md` (this file)         — plan + reference notes
- `test_metal_parity.py`             — gate test (skips on non-Mac); will fail until Metal lands
- `run_cuda_4w.txt`                  — paste-and-run cheatsheet from the CUDA work; useful for cross-backend comparison once Metal lands

## Reference: how voxtral_realtime does Metal

`examples/models/voxtral_realtime` already supports Metal end-to-end. The
relevant pieces, by file:

| File | Symbol | Purpose |
|---|---|---|
| `voxtral_realtime/model.py` | `MetalSDPA` | Calls `torch.ops.aten._scaled_dot_product_attention_math_for_mps` directly. Avoids the `custom_sdpa` op which is XNNPACK-only. |
| `voxtral_realtime/model.py` | `_build_attn_mask` | **Additive** fp causal mask (`(valid - 1) * 1e9`) — Metal MPS doesn't support bool mask allocation in some versions. |
| `voxtral_realtime/model.py` | `LMAttention.__init__` | `elif self.backend == "metal": self.kv_cache = StaticKVCache(...); self.sdpa = MetalSDPA(...)`. Same `StaticKVCache` we already use for CUDA. |
| `voxtral_realtime/export_voxtral_rt.py` | `lower_to_executorch` `metal` branch | Imports `MetalBackend`, `MetalPartitioner`. Runs `_linear_bias_decomposition` decomp because Metal AOTI errors on linears with bias due to 0-stride reinterpret_tensor. |
| `voxtral_realtime/CMakePresets.json` | `voxtral-realtime-metal` | `EXECUTORCH_BUILD_METAL=ON`, `condition: hostSystemName == "Darwin"`. |

## Plan

### Phase M-1 — Model: backend-aware SDPA for Metal

`examples/models/voxtral_tts/model.py`:

1. Port `MetalSDPA` from realtime verbatim.
2. Port `_build_attn_mask` (additive fp variant). Keep `_build_causal_mask_bool` for CUDA.
3. In `MistralDecoder.forward`, branch:
   ```python
   if self.config.backend == "cuda":
       attn_mask = _build_causal_mask_bool(...)
   elif self.config.backend == "metal":
       attn_mask = _build_attn_mask(input_pos, max_seq_len,
                                    input_embeds.device, input_embeds.dtype)
   else:
       attn_mask = None
   ```
4. In `LMAttention.__init__`, add `metal` branch using `StaticKVCache` (same
   class CUDA uses — BHSD layout, `index_copy_` updates) and `MetalSDPA`.
5. Codec is already CUDA-friendly via the conv-as-matmul rewrite. **Metal
   should be able to lower the same code** — verify in Phase M-3 below.

### Phase M-2 — Export: `--backend metal`

`examples/models/voxtral_tts/export_voxtral_tts.py`:

1. Add `metal` to `--backend` choices.
2. In `lower_to_executorch`, add a `metal` branch:
   ```python
   elif backend == "metal":
       from executorch.backends.apple.metal.metal_backend import MetalBackend
       from executorch.backends.apple.metal.metal_partitioner import MetalPartitioner
       updated_programs = {}
       decomp_table = torch.export.default_decompositions()
       decomp_table[torch.ops.aten.linear.default] = _linear_bias_decomposition
       for key, ep in programs.items():
           updated_programs[key] = ep.run_decompositions(decomp_table)
       programs = updated_programs
       partitioner = {
           key: [MetalPartitioner([MetalBackend.generate_method_name_compile_spec(key)])]
           for key in programs
       }
   ```
3. Port `_linear_bias_decomposition` from `voxtral_realtime/export_voxtral_rt.py:373-379`.
4. Codec export: same Metal partitioner (no `triton_kernel_mode=OFF` knob — that's CUDA-only).

### Phase M-3 — Build: Metal CMake preset

`examples/models/voxtral_tts/CMakePresets.json`:

```json
{
    "name": "voxtral-tts-metal",
    "displayName": "Voxtral TTS runner (Metal)",
    "inherits": ["voxtral-tts-base"],
    "cacheVariables": { "EXECUTORCH_BUILD_METAL": "ON" },
    "condition": {
        "lhs": "${hostSystemName}",
        "type": "equals",
        "rhs": "Darwin"
    }
}
```

Plus matching `buildPresets` and `workflowPresets` entries (mirror the CUDA pattern).

### Phase M-4 — Runner

The runner is already backend-transparent (`Module(model_path, data_path, …)`).
For Metal, the `.ptd` filename convention should match — check what
`MetalBackend` writes (CUDA writes `aoti_cuda_blob.ptd`; Metal probably writes
`aoti_metal_blob.ptd`).

Codec collision: same workaround we used on CUDA (`codec_aoti_metal_blob.ptd`)
will likely apply.

### Phase M-5 — Parity test

`examples/models/voxtral_tts/test_metal_parity.py` (already in this branch as
a stub) should be updated to exercise:

- `MetalSDPA` math vs `aten.scaled_dot_product_attention`
- Prefill hidden cosine vs XNNPACK baseline (≥ 0.998)
- First-frame semantic argmax match
- Codec parity (same conv-as-matmul rewrite as CUDA)

The test must skip cleanly on Linux (no Metal) — see `test_cuda_parity.py` for
the skip pattern.

### Phase M-6 — Benchmark

Update `BENCHMARK.md` with a Metal section. Targets to capture:
- Apple Silicon LM RTF
- Codec time (Metal SDPA on the codec's ALiBi attention may need investigation)
- Total wall clock vs the CUDA baseline (3.7 s on A100)

## Known gotchas to watch for

1. **Metal AOTI doesn't accept `Dim.AUTO` for some shapes** — `voxtral_realtime/export_voxtral_rt.py:209-227` uses `Dim.AUTO` for the audio encoder on Metal but explicit bounds for XNNPACK. Apply the same pattern for any Metal-specific dynamic shapes.
2. **Linear with bias** — Metal AOTI errors on 0-stride reinterpret. Always run `_linear_bias_decomposition` in the lowering pass.
3. **bool causal masks** — some Metal MPS versions can't allocate `torch.bool` on MPS device. Use the additive fp mask form (`_build_attn_mask`).
4. **Codec ALiBi attention** — uses an additive position-bias mask. Matches Metal's preferred mask form. Should JustWork(tm).
5. **Quantization** — Metal supports `--qlinear fpa4w` (Apple-specific 4-bit weight format). See realtime's `--qlinear-encoder fpa4w --qlinear fpa4w` usage.

## Order of operations for the developer with a Mac

```
Phase M-1 (model.py)       → eager parity test on Mac (CPU vs MPS, both fp32)
Phase M-2 (export)          → produce model.pte + codec_decoder.pte for Metal
Phase M-3 (CMake preset)    → build voxtral_tts_runner with Metal
Phase M-4 (runner)          → first end-to-end run
Phase M-5 (parity test)     → fill in test_metal_parity.py and confirm green
Phase M-6 (benchmark)       → update BENCHMARK.md
```

Each phase has a single test gate. Don't pipeline.
