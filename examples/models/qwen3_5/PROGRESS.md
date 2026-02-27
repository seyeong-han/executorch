# Qwen3.5 ExecuTorch Export -- Progress Tracker

## Target

- **Model**: Qwen3.5-27B (Dense) -- `Qwen/Qwen3.5-27B`
- **HF model_type**: `qwen3_next`
- **Hardware**: M3 Pro 36GB MBP
- **Backends**: XNNPACK (Phase 1), Metal (Phase 2)

## Phase 1: XNNPACK Backend

| Task | Status | Notes |
|------|--------|-------|
| Research & CONTEXT.md | Done | Architecture, constraints, precedents documented |
| model.py (GatedDeltaNetBlock) | Done | Decode-only recurrence with registered buffers |
| model.py (GatedAttentionBlock) | Done | Partial RoPE, sigmoid gating, QK norms |
| model.py (Qwen3_5Model) | Done | 64-layer hybrid model |
| convert_weights.py | Done | HF qwen3_next weight mapping |
| export_qwen3_5.py (XNNPACK) | Done | 8da4w quantization, XnnpackPartitioner |
| config/qwen3_5_27b_config.json | Done | Architecture parameters |
| XNNPACK export test | Done | Tiny model: export + XNNPACK lowering + PTE generation all pass |

## Phase 2: Metal Backend

| Task | Status | Notes |
|------|--------|-------|
| Metal attention (StaticKVCache) | Done | index_copy_ based, [B,H,S,D] layout |
| head_dim=256 SDPA handling | Done | Decomposed attention (matmul+softmax+matmul) |
| export_qwen3_5.py (Metal) | Done | fpa4w quant, MetalPartitioner, _linear_bias_decomposition |
| CMakeLists.txt | Done | Metal linking |
| CMakePresets.json | Done | cpu + metal presets |
| Metal export test | Blocked | Export + partitioning works; AOTI compile needs torchao with TORCHAO_BUILD_EXPERIMENTAL_MPS=1 |

## Known Issues & Risks

1. **head_dim=256 on Metal**: The Metal SDPA kernel only supports 64/96/128. We use decomposed attention (matmul+softmax+matmul) as the initial approach. A kernel extension for head_dim=256 would give better performance.

2. **Prefill speed**: Decode-only mode (token-by-token) is slower than chunked prefill. The chunked prefill has data-dependent loops that break `torch.export`. Future optimization: implement a fixed-iteration prefill method.

3. **Model size**: 27B at INT4 = ~14 GB. Export process needs meta-device loading to fit in memory.

4. **fpa4w quantization**: Requires torchao built with experimental MPS ops. All linear weight K dimensions must be divisible by group_size (32).

## Export Commands

### XNNPACK
```bash
python examples/models/qwen3_5/export_qwen3_5.py \
    --checkpoint_dir /path/to/Qwen3.5-27B \
    --backend xnnpack \
    --quantize 8da4w \
    --max_seq_len 2048 \
    --output_dir /path/to/output
```

### Metal
```bash
python examples/models/qwen3_5/export_qwen3_5.py \
    --checkpoint_dir /path/to/Qwen3.5-27B \
    --backend metal \
    --quantize fpa4w \
    --max_seq_len 2048 \
    --output_dir /path/to/output
```

## Test Results

### Environment: `metal-backend` conda env

- PyTorch 2.11.0.dev20260215
- ExecuTorch with `EXECUTORCH_BUILD_KERNELS_TORCHAO=1 TORCHAO_BUILD_EXPERIMENTAL_MPS=1`
- macOS 26.2, M3 Pro 36GB

### Tiny Model Validation (8 layers, hidden=64, vocab=100)

| Test | Result | Notes |
|------|--------|-------|
| Model instantiation | Pass | 380,752 params |
| Forward pass (single token) | Pass | |
| Multi-step decode (5 tokens) | Pass | State accumulates correctly |
| torch.export (static seq_len=1) | Pass | 1313 graph nodes (XNNPACK), 1309 (Metal) |
| XNNPACK lowering + PTE | Pass | 1764.9 KB |
| XNNPACK inference | Pass | Prefill: 181.5 tok/s, Decode: 200.4 tok/s |
| Metal AOTI compile + PTE | Pass | 3407.5 KB |
| Metal inference (Python) | Skip | Metal backend not in portable_lib -- needs C++ runner |

### Full Model Export (27B)

_(Requires downloading Qwen3.5-27B checkpoint -- ~54 GB at fp16)_
