# Qwen3.5 ExecuTorch Export

Export [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) (dense, 27B params) to ExecuTorch `.pte` format with XNNPACK (CPU) or Metal (GPU) backend.

Qwen3.5 uses a hybrid architecture combining **Gated DeltaNet** (linear attention) with **Gated Attention** layers. In HuggingFace Transformers, the model type is `qwen3_next`.

## Architecture

- 64 layers: 48 Gated DeltaNet + 16 Gated Attention (3:1 ratio)
- DeltaNet layers maintain recurrent state (not KV cache) -- constant memory regardless of context length
- Attention layers use standard KV cache with GQA (24 Q heads, 4 KV heads, head_dim=256)
- Partial RoPE: only 25% of head dimensions get rotary embeddings
- Dense SwiGLU FFN (intermediate=17408)

See [CONTEXT.md](CONTEXT.md) for full architecture details.

## Requirements

- ExecuTorch (built from source)
- PyTorch 2.6+
- `safetensors` (`pip install safetensors`)
- HuggingFace model checkpoint (`Qwen/Qwen3.5-27B`)
- For Metal: `torchao` built with `TORCHAO_BUILD_EXPERIMENTAL_MPS=1`

### Download the Model

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3.5-27B --local-dir ./Qwen3.5-27B
```

## Export

### XNNPACK (CPU)

```bash
python examples/models/qwen3_5/export_qwen3_5.py \
    --checkpoint-dir ./Qwen3.5-27B \
    --backend xnnpack \
    --quantize 8da4w \
    --max-seq-len 2048 \
    --output-dir ./qwen3_5_exports
```

Quantization options for XNNPACK:
- `8da4w` -- 8-bit dynamic activation, 4-bit weight (recommended)
- `8da8w` -- 8-bit dynamic activation, 8-bit weight
- `8w` -- 8-bit weight-only
- `4w` -- 4-bit weight-only

### Metal (GPU)

```bash
python examples/models/qwen3_5/export_qwen3_5.py \
    --checkpoint-dir ./Qwen3.5-27B \
    --backend metal \
    --quantize fpa4w \
    --max-seq-len 2048 \
    --output-dir ./qwen3_5_exports
```

Metal uses `fpa4w` quantization (floating-point activation, 4-bit weight via torchao HQQ).

### Portable (no acceleration)

```bash
python examples/models/qwen3_5/export_qwen3_5.py \
    --checkpoint-dir ./Qwen3.5-27B \
    --backend portable \
    --max-seq-len 2048 \
    --output-dir ./qwen3_5_exports
```

## Build C++ Runner

### CPU (XNNPACK)

First build ExecuTorch with XNNPACK:
```bash
cmake --workflow --preset llm-xnnpack
```

Then build the runner:
```bash
cd examples/models/qwen3_5 && cmake --workflow --preset qwen3-5-cpu
```

### Metal (macOS)

First build ExecuTorch with Metal:
```bash
cmake --workflow --preset llm-metal-stats
```

Then build the runner:
```bash
cd examples/models/qwen3_5 && cmake --workflow --preset qwen3-5-metal
```

## Design Decisions

### Decode-Only Export

The model is exported with static `seq_len=1` (decode-only mode). The Gated DeltaNet recurrence uses a Python for-loop that prevents dynamic sequence length export via `torch.export`. For prefill, the runner calls the model token-by-token.

### head_dim=256 on Metal

The Metal SDPA kernel only supports head_dim 64/96/128. Since Qwen3.5 uses head_dim=256, we use decomposed attention (matmul + softmax + matmul) for the Metal path instead of the native MPS SDPA op. This avoids runtime errors while allowing full Metal GPU acceleration for all other operations.

### Memory on 36GB M3 Pro

At INT4 quantization, the 27B model uses ~14 GB for weights. With ~4 GB for state (KV cache + DeltaNet recurrent/conv states) and system overhead, the total fits comfortably in 36 GB unified memory.

## Files

| File | Description |
|------|-------------|
| `model.py` | Qwen3.5 model with GatedDeltaNet, GatedAttention, KV cache variants |
| `convert_weights.py` | HF `qwen3_next` weight name mapping |
| `export_qwen3_5.py` | Export script (XNNPACK / Metal / Portable) |
| `config/qwen3_5_27b_config.json` | Model architecture parameters |
| `CMakeLists.txt` | C++ build configuration |
| `CMakePresets.json` | CMake presets (cpu, metal) |
| `CONTEXT.md` | Full research documentation |
| `PROGRESS.md` | Export progress and test results |
