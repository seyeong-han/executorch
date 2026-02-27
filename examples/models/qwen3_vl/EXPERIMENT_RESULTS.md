# Qwen3-VL-2B ExecuTorch Export — Experiment Results

Systematic evaluation of exporting Qwen3-VL-2B-Instruct to ExecuTorch via
`optimum-executorch` across two backends (XNNPACK and CoreML) with incremental
optimizations. All measurements taken on Apple Silicon (M-series Mac),
generating 50 tokens in response to "What is in this image?" on a 1000x667
lake/dock photograph.

## Summary Table

| Config | Backend | Recipe | Model Size | Decode Rate | Prefill (667 tok) | Vision Encoder |
|--------|---------|--------|-----------|-------------|-------------------|----------------|
| XNNPACK quantized | XNNPACK | `xnnpack` | 1.4 GB | 25-35 tok/s | 3.4s (bulk) | HF eager (8.3s) |
| CoreML fp16 baseline | CoreML | `coreml_fp16` | 4.4 GB | 17.9 tok/s | 69.6s (1-by-1) | PTE (28.1s) |
| CoreML fp16 + int4 | CoreML | `coreml_fp16_int4` | **1.7 GB** | **29.8 tok/s** | **34.4s** (1-by-1) | PTE (6.0s) |
| CoreML fp16 + int4 + ANE | CoreML | `coreml_fp16_int4_ne` | 1.7 GB | fails on macOS | -- | PTE (12.4s, VE ok) |

## Experiment 1: XNNPACK Export (Baseline)

Quantized XNNPACK export using `optimum-executorch` with custom SDPA and KV
cache, 8da4w decoder quantization, and 8w embeddings.

**Export command:**
```bash
optimum-cli export executorch \
  --model "Qwen/Qwen3-VL-2B-Instruct" \
  --task "multimodal-text-to-text" \
  --recipe "xnnpack" \
  --use_custom_sdpa \
  --use_custom_kv_cache \
  --qlinear "8da4w" \
  --qlinear_group_size 32 \
  --qlinear_encoder "8da4w,8da8w" \
  --qlinear_encoder_group_size 32 \
  --qembedding "8w" \
  --qembedding_encoder "8w" \
  --dtype "float32" \
  --output_dir="qwen3/Qwen3-VL-2B-Instruct-xnnpack"
```

**Results:**
- Model size: 1.4 GB
- Decode rate: ~25-35 tok/s
- Prefill: 3.4s (bulk, dynamic shapes)
- Vision encoder: runs via HF eager PyTorch (Conv3d unsupported in portable runtime)
- Response: accurate, coherent image description

**Key limitation:** The Qwen3-VL vision encoder uses Conv3d for 3D patch
embedding. XNNPACK's portable runtime `aten::convolution.out` kernel does not
support 5D inputs, so the vision encoder cannot run from the PTE. The runtime
script falls back to loading the full HF model and running the vision encoder
in PyTorch eager mode.

## Experiment 2: CoreML Export — fp16 Baseline

Unquantized CoreML export with fp16 compute precision and `ComputeUnit.ALL`.

**Export command:**
```bash
optimum-cli export executorch \
  --model "Qwen/Qwen3-VL-2B-Instruct" \
  --task "multimodal-text-to-text" \
  --recipe "coreml_fp16" \
  --disable_dynamic_shapes \
  --dtype "float32" \
  --output_dir="qwen3/Qwen3-VL-2B-Instruct-coreml"
```

**Results:**
- Model size: 4.4 GB (fp32 weights, fp16 compute)
- Decode rate: 17.9 tok/s
- Prefill: 69.6s (token-by-token, static shapes)
- Vision encoder: runs from PTE natively (CoreML delegates Conv3d)
- Response: accurate, coherent image description

**Key findings:**
- CoreML successfully delegates the Conv3d patch embedding — no HF eager fallback needed.
- `--disable_dynamic_shapes` is required because coremltools cannot handle `SymInt` placeholders from dynamic `torch.export` shapes.
- The static seq_len=1 export means prefill must process each token individually, which is ~20x slower than XNNPACK's bulk prefill.
- First run compiles CoreML models (~5 min timeout per method). Subsequent runs use the cached compiled models.

**Challenges overcome:**
1. `index_put` crash in coremltools: the standard static KV cache uses `index_put` for cache updates, but coremltools fails converting this op (rank mismatch bug). Fixed by using `--disable_dynamic_shapes` which simplifies the cache update pattern.
2. `SymInt` placeholder error: dynamic shapes produce symbolic integers that coremltools cannot process. Fixed with `--disable_dynamic_shapes`.
3. Partition count mismatch: CoreML's default `POSITIONAL` weight sharing strategy requires equal partitions across methods. Fixed by setting `MULTIMETHOD_WEIGHT_SHARING_STRATEGY.DISABLED`.

## Experiment 3: CoreML + int4 Quantization

CoreML-native int4 weight quantization via `OpLinearQuantizerConfig`, applied
post-conversion by coremltools. Quantizes all linear layer weights to int4
symmetric per-block (block_size=32), automatically skipping embeddings (gather ops).

**Export command:**
```bash
optimum-cli export executorch \
  --model "Qwen/Qwen3-VL-2B-Instruct" \
  --task "multimodal-text-to-text" \
  --recipe "coreml_fp16_int4" \
  --disable_dynamic_shapes \
  --dtype "float32" \
  --output_dir="qwen3/Qwen3-VL-2B-Instruct-coreml-int4"
```

**Results:**
- Model size: **1.7 GB** (2.6x smaller than fp16 baseline)
- Decode rate: **29.8 tok/s** (1.66x faster)
- Prefill: **34.4s** (2.0x faster)
- Vision encoder: **6.0s** (4.7x faster, cached run)
- Response: accurate, coherent — no visible quality degradation

**Improvement over CoreML baseline:**

| Metric | fp16 Baseline | fp16 + int4 | Improvement |
|--------|--------------|-------------|-------------|
| Model size | 4.4 GB | 1.7 GB | **-61%** |
| Decode rate | 17.9 tok/s | 29.8 tok/s | **+66%** |
| Prefill time | 69.6s | 34.4s | **-51%** |
| Vision encoder | 28.1s | 6.0s | **-79%** |

The int4 quantization was the single highest-impact optimization. Smaller
weights mean less memory bandwidth pressure, which is the primary bottleneck
for LLM inference on Apple Silicon.

## Experiment 4: CoreML + int4 + ANE Targeting

Same int4 quantization but with `ComputeUnit.CPU_AND_NE` to force ops onto
the Apple Neural Engine.

**Export command:**
```bash
optimum-cli export executorch \
  --model "Qwen/Qwen3-VL-2B-Instruct" \
  --task "multimodal-text-to-text" \
  --recipe "coreml_fp16_int4_ne" \
  --disable_dynamic_shapes \
  --dtype "float32" \
  --output_dir="qwen3/Qwen3-VL-2B-Instruct-coreml-int4-ne"
```

**Results:**
- Export succeeds (1.7 GB model)
- Vision encoder executes successfully (12.4s)
- Text decoder fails at runtime with error 0x23

**Analysis:** `CPU_AND_NE` restricts compute to CPU and Neural Engine only
(no GPU). Some ops in the text decoder require GPU and have no viable compute
target under this restriction. On actual iOS devices with a dedicated ANE,
this configuration may work and provide better sustained throughput. The `ALL`
compute unit (which lets CoreML choose the best target per op) is the correct
default for macOS.

## Changes Made to optimum-executorch

All changes are in the `optimum-executorch` repository:

### `optimum/exporters/executorch/recipes/coreml.py`
- Added `MultiModalTextToTextExportableModule` to imports and type annotation
- Rewrote `_lower_to_executorch` to produce a single multi-method `.pte` (matching XNNPACK/Metal pattern) instead of separate files per method
- Used per-method `CoreMLPartitioner` dicts with `MULTIMETHOD_WEIGHT_SHARING_STRATEGY.DISABLED`
- Threaded `op_linear_quantizer_config` through to `CoreMLBackend.generate_compile_specs`
- Registered `coreml_fp16_int4[_cpu|_gpu|_ne]` recipe variants with int4 weight quantization

### `optimum/exporters/executorch/integrations.py`
- Added `disable_dynamic_shapes` parameter to `MultiModalTextToTextExportableModule`
- Modified `_prepare_decoder_only_export_inputs` and `_prepare_text_embedding_export_inputs` to support static shapes (seq_len=1) when the flag is set

### `optimum/exporters/executorch/tasks/multimodal_text_to_text.py`
- Wired `disable_dynamic_shapes` kwarg through to `MultiModalTextToTextExportableModule`

## Recommended Configuration

For Apple devices (iPhone/iPad/Mac):
```bash
optimum-cli export executorch \
  --model "Qwen/Qwen3-VL-2B-Instruct" \
  --task "multimodal-text-to-text" \
  --recipe "coreml_fp16_int4" \
  --disable_dynamic_shapes \
  --dtype "float32" \
  --output_dir="qwen3/Qwen3-VL-2B-Instruct-coreml-int4"
```

For cross-platform (Android/Linux/any CPU):
```bash
optimum-cli export executorch \
  --model "Qwen/Qwen3-VL-2B-Instruct" \
  --task "multimodal-text-to-text" \
  --recipe "xnnpack" \
  --use_custom_sdpa \
  --use_custom_kv_cache \
  --qlinear "8da4w" \
  --qlinear_group_size 32 \
  --qlinear_encoder "8da4w,8da8w" \
  --qlinear_encoder_group_size 32 \
  --qembedding "8w" \
  --qembedding_encoder "8w" \
  --dtype "float32" \
  --output_dir="qwen3/Qwen3-VL-2B-Instruct-xnnpack"
```

## Future Optimization Opportunities

1. **Enumerated shapes for prefill**: Export with multiple fixed sequence lengths (1, 16, 64, 256) to enable chunked prefill instead of token-by-token. Requires fixing the `SymInt` / `index_put` issues with dynamic shapes in CoreML.

2. **Pre-compiled model caching**: Set `model_type=CoreMLBackend.MODEL_TYPE.COMPILED_MODEL` to embed pre-compiled `.mlmodelc` in the `.pte`, eliminating the multi-minute first-run compilation.

3. **torchao pre-export quantization**: Apply int4/int8 quantization before export (at the PyTorch level) for finer control over per-component quantization settings, potentially combined with CoreML-native quantization.
