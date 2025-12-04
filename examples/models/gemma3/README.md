# Summary

This example demonstrates how to export and run Google's [Gemma 3](https://huggingface.co/google/gemma-3-4b-it) models locally on ExecuTorch with multiple backend support.

Gemma 3 comes in two variants:
- **Multimodal (Vision-Language)**: `google/gemma-3-4b-it` - requires image input
- **Text-only**: `google/gemma-3-1b-it` - text generation only

# Setting up Optimum ExecuTorch

To export models, we use [Optimum ExecuTorch](https://github.com/huggingface/optimum-executorch), which enables exporting models directly from HuggingFace's Transformers.

Install through pip:
```bash
pip install optimum-executorch
```

Or install from source:
```bash
git clone https://github.com/huggingface/optimum-executorch.git
cd optimum-executorch
pip install -e .
```

# Obtaining the Tokenizer

Download the `tokenizer.json` file from HuggingFace:
```bash
# Using huggingface-cli
huggingface-cli download google/gemma-3-1b-it tokenizer.json --local-dir ./gemma3_tokenizer

# Or using curl
curl -L https://huggingface.co/google/gemma-3-1b-it/resolve/main/tokenizer.json -o tokenizer.json
```

---

# Text-Only Model (gemma-3-1b-it)

For text generation without image input, use the text-only `gemma-3-1b-it` model with the `llama_main` runner.

## Exporting the Text-Only Model

### Export with Vulkan Backend (GPU acceleration)
```bash
optimum-cli export executorch \
  --model "google/gemma-3-1b-it" \
  --task "text-generation" \
  --recipe "vulkan" \
  --dtype float16 \
  --output_dir="gemma3_vulkan"
```

### Export with XNNPACK Backend (CPU)
```bash
optimum-cli export executorch \
  --model "google/gemma-3-1b-it" \
  --task "text-generation" \
  --recipe "xnnpack" \
  --output_dir="gemma3_xnnpack"
```

This generates:
- `model.pte` - The exported model file

## Building the llama_main Runner

The `llama_main` runner is a generic text-only LLM runner that works with Gemma 3 and other text models.

### Prerequisites

First, build ExecutorTorch from the repository root:
```bash
cd executorch

# Install dependencies and build ExecuTorch with desired backends
# For Vulkan support:
./install_executorch.sh --use-pt-pinned-commit

cmake -DEXECUTORCH_BUILD_VULKAN=ON \
      -DEXECUTORCH_BUILD_XNNPACK=ON \
      -Bcmake-out .

cmake --build cmake-out -j$(nproc)
```

### Building llama_main

```bash
# Create build directory for llama runner
mkdir -p cmake-out/examples/models/llama
cd cmake-out/examples/models/llama

# Configure with paths to the main build
cmake ../../../../examples/models/llama \
    -DCMAKE_PREFIX_PATH=$(pwd)/../../.. \
    -Dgflags_DIR=$(pwd)/../../../third-party/gflags

# Build
cmake --build . --target llama_main -j$(nproc)
```

The binary will be at: `cmake-out/examples/models/llama/llama_main`

## Running the Text-Only Model

### Command Line Options
```
--model_path      Path to the exported model.pte file
--tokenizer_path  Path to tokenizer.json
--prompt          Text prompt for generation
--max_new_tokens  Maximum number of tokens to generate (default: 128)
--temperature     Sampling temperature (0 = greedy, default: 0.8)
--cpu_threads     Number of CPU threads (-1 = auto)
```

### Example Usage (Vulkan)

> **Note**: Vulkan requires a device with Vulkan support (Android, Linux with Vulkan GPU). macOS requires MoltenVK.

```bash
./cmake-out/examples/models/llama/llama_main \
  --model_path=gemma3_vulkan/model.pte \
  --tokenizer_path=tokenizer.json \
  --prompt="What is the capital of France?" \
  --max_new_tokens=100 \
  --temperature=0
```

### Example Usage (XNNPACK/CPU)
```bash
./cmake-out/examples/models/llama/llama_main \
  --model_path=gemma3_xnnpack/model.pte \
  --tokenizer_path=tokenizer.json \
  --prompt="Explain quantum computing in simple terms." \
  --max_new_tokens=200 \
  --temperature=0.7
```

---

# Multimodal Model (gemma-3-4b-it)

For vision-language tasks requiring image input, use the multimodal `gemma-3-4b-it` model with the `gemma3_e2e_runner`.

## Exporting the Multimodal Model

### Export with CUDA Backend
```bash
optimum-cli export executorch \
  --model "google/gemma-3-4b-it" \
  --task "multimodal-text-to-text" \
  --recipe "cuda" \
  --dtype bfloat16 \
  --device cuda \
  --output_dir="gemma3_cuda"
```

This generates:
- `model.pte` - The exported model
- `aoti_cuda_blob.ptd` - The CUDA kernel blob required for runtime

### Export with INT4 Quantization (Tile Packed)

For improved performance and reduced memory footprint:
```bash
optimum-cli export executorch \
  --model "google/gemma-3-4b-it" \
  --task "multimodal-text-to-text" \
  --recipe "cuda" \
  --dtype bfloat16 \
  --device cuda \
  --qlinear 4w \
  --qlinear_encoder 4w \
  --qlinear_packing_format tile_packed_to_4d \
  --qlinear_encoder_packing_format tile_packed_to_4d \
  --output_dir="gemma3_cuda_int4"
```

### Export with Vulkan Backend
```bash
optimum-cli export executorch \
  --model "google/gemma-3-4b-it" \
  --task "multimodal-text-to-text" \
  --recipe "vulkan" \
  --dtype float16 \
  --output_dir="gemma3_vulkan_multimodal"
```

## Building the Gemma3 Multimodal Runner

### Prerequisites
Ensure you have a CUDA-capable GPU and CUDA toolkit installed for CUDA backend.

### Building
```bash
# Build with CUDA support
make gemma3-cuda

# Build with CPU support
make gemma3-cpu

# Build with Vulkan support
make gemma3-vulkan
```

## Running the Multimodal Model

The multimodal runner requires:
- `model.pte` - The exported model file
- `aoti_cuda_blob.ptd` - The CUDA kernel blob (CUDA backend only)
- `tokenizer.json` - The tokenizer file
- An image file (PNG, JPG, etc.)

### Example Usage (CUDA)
```bash
./cmake-out/examples/models/gemma3/gemma3_e2e_runner \
  --model_path gemma3_cuda/model.pte \
  --data_path gemma3_cuda/aoti_cuda_blob.ptd \
  --tokenizer_path tokenizer.json \
  --image_path docs/source/_static/img/et-logo.png \
  --prompt "What is in this image?" \
  --temperature 0
```

### Example Usage (Vulkan)
```bash
./cmake-out/examples/models/gemma3/gemma3_e2e_runner \
  --model_path gemma3_vulkan_multimodal/model.pte \
  --tokenizer_path tokenizer.json \
  --image_path docs/source/_static/img/et-logo.png \
  --prompt "Describe this image in detail." \
  --temperature 0
```

---

# Example Output

### Text-Only Model
```
$ ./llama_main --model_path=gemma3_xnnpack/model.pte --tokenizer_path=tokenizer.json \
    --prompt="What is the capital of France?" --max_new_tokens=50 --temperature=0

The capital of France is Paris. Paris is not only the capital but also the largest city
in France, serving as the country's political, economic, and cultural center.
```

### Multimodal Model
```
Okay, let's break down what's in the image!

It appears to be a stylized graphic combining:

*   **A Microchip:** The core shape is a representation of a microchip (the integrated circuit).
*   **An "On" Symbol:**  There's an "On" symbol (often represented as a circle with a vertical line) incorporated into the microchip design.
*   **Color Scheme:** The microchip is colored in gray, and

PyTorchObserver {"prompt_tokens":271,"generated_tokens":99,"model_load_start_ms":0,"model_load_end_ms":0,"inference_start_ms":1761118126790,"inference_end_ms":1761118128385,"prompt_eval_end_ms":1761118127175,"first_token_ms":1761118127175,"aggregate_sampling_time_ms":86,"SCALING_FACTOR_UNITS_PER_SECOND":1000}
```

---

# Backend Comparison

| Backend | Device | Model Type | Use Case |
|---------|--------|------------|----------|
| **XNNPACK** | CPU | Text-only, Multimodal | Cross-platform, no GPU required |
| **Vulkan** | GPU | Text-only, Multimodal | Android, Linux (requires Vulkan support) |
| **CUDA** | NVIDIA GPU | Text-only, Multimodal | High-performance inference on NVIDIA GPUs |

# Troubleshooting

### Vulkan Runtime Error on macOS
```
Pytorch Vulkan Runtime: The global runtime could not be retrieved because it failed to initialize.
```
macOS doesn't have native Vulkan support. Options:
1. Install MoltenVK: `brew install molten-vk`
2. Use XNNPACK backend instead for CPU inference
3. Run on a device with native Vulkan support (Android, Linux)

### Tokenizer Loading Issues
Ensure you're using the correct tokenizer format. The runner supports:
- HuggingFace JSON tokenizers (`tokenizer.json`)
- TikToken tokenizers
- SentencePiece tokenizers
