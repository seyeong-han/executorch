#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export Qwen3.5-27B to ExecuTorch with XNNPACK or Metal backend.

Usage:
    # XNNPACK (CPU)
    python examples/models/qwen3_5/export_qwen3_5.py \
        --checkpoint-dir /path/to/Qwen3.5-27B \
        --backend xnnpack \
        --quantize 8da4w

    # Metal (GPU)
    python examples/models/qwen3_5/export_qwen3_5.py \
        --checkpoint-dir /path/to/Qwen3.5-27B \
        --backend metal \
        --quantize fpa4w
"""

import argparse
import os
import time

import torch
import torch.export

from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.passes import MemoryPlanningPass

from executorch.examples.models.qwen3_5.model import (
    Qwen3_5Config,
    Qwen3_5Model,
    load_model,
)


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------


def quantize_model(model: Qwen3_5Model, config: str, group_size: int = 32):
    """Apply quantization to the model using torchao."""
    from executorch.extension.llm.export.quantize import quantize_model_

    print(f"  Applying {config} quantization (group_size={group_size})...")
    quantize_model_(
        model,
        qlinear_config=config,
        qlinear_group_size=group_size,
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_model(
    model: Qwen3_5Model,
    config: Qwen3_5Config,
    backend: str,
) -> dict:
    """Export the model to ExportedProgram(s).

    Exports with static seq_len=1 (decode-only mode) because the Gated DeltaNet
    recurrence uses a Python for-loop that prevents dynamic sequence length export.
    For prefill, the runner calls the model token-by-token.
    """
    model.eval()
    model.requires_grad_(False)

    # Static seq_len=1 for decode mode
    sample_tokens = torch.zeros(1, 1, dtype=torch.long)
    sample_pos = torch.zeros(1, dtype=torch.long)

    print("  Exporting model with torch.export (static seq_len=1, decode mode)...")
    t0 = time.time()

    ep = torch.export.export(
        model,
        (sample_tokens, sample_pos),
        strict=False,
    )

    print(f"  Export completed in {time.time() - t0:.1f}s")
    return {"forward": ep}


# ---------------------------------------------------------------------------
# Metal decomposition
# ---------------------------------------------------------------------------


def _linear_bias_decomposition(input, weight, bias=None):
    """Decompose linear with bias into matmul + add.

    Avoids reinterpret_tensor_wrapper producing 0-stride tensors,
    which ExecuTorch's Metal AOTI backend does not support.
    """
    weight_t = torch.ops.aten.t.default(weight)
    out = torch.ops.aten.matmul.default(input, weight_t)
    if bias is not None:
        return torch.ops.aten.add.Tensor(out, bias)
    return out


# ---------------------------------------------------------------------------
# Lowering
# ---------------------------------------------------------------------------


def lower_to_executorch(programs: dict, metadata: dict, backend: str):
    """Lower exported programs to ExecuTorch with the specified backend."""

    if backend == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackDynamicallyQuantizedPartitioner,
            XnnpackPartitioner,
        )

        print("\nLowering to ExecuTorch with XNNPACK...")
        partitioner = {
            key: [XnnpackDynamicallyQuantizedPartitioner(), XnnpackPartitioner()]
            for key in programs
        }

    elif backend == "metal":
        from executorch.backends.apple.metal.metal_backend import MetalBackend
        from executorch.backends.apple.metal.metal_partitioner import MetalPartitioner

        print("\nLowering to ExecuTorch with Metal...")

        # Run decompositions for Metal backend
        updated_programs = {}
        for key, ep in programs.items():
            updated_programs[key] = ep.run_decompositions(
                {torch.ops.aten.linear.default: _linear_bias_decomposition}
            )
        programs = updated_programs

        partitioner = {}
        for key in programs:
            compile_specs = [MetalBackend.generate_method_name_compile_spec(key)]
            partitioner[key] = [MetalPartitioner(compile_specs)]

    else:
        print("\nLowering to ExecuTorch (portable)...")
        partitioner = {key: [] for key in programs}

    et_prog = to_edge_transform_and_lower(
        programs,
        partitioner=partitioner,
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        constant_methods=metadata,
    )

    return et_prog.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            do_quant_fusion_and_const_prop=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        ),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Export Qwen3.5 to ExecuTorch"
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Directory with HuggingFace safetensors checkpoint",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Model config JSON (default: auto-detect from checkpoint_dir/config.json "
        "or use built-in 27B config)",
    )
    parser.add_argument(
        "--backend",
        default="xnnpack",
        choices=["portable", "xnnpack", "metal"],
        help="Backend for acceleration (default: xnnpack)",
    )
    parser.add_argument(
        "--output-dir",
        default="./qwen3_5_exports",
        help="Output directory (default: ./qwen3_5_exports)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Maximum sequence length / KV cache size (default: 2048)",
    )
    parser.add_argument(
        "--quantize",
        default=None,
        choices=["4w", "8w", "8da4w", "8da8w", "fpa4w"],
        help="Quantize linear layers",
    )
    parser.add_argument(
        "--quantize-group-size",
        type=int,
        default=32,
        help="Group size for quantization (default: 32)",
    )
    args = parser.parse_args()

    # Validate
    if args.quantize == "fpa4w" and args.backend != "metal":
        parser.error("--quantize=fpa4w can only be used with --backend=metal")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    config_path = args.config
    if config_path is None:
        hf_config = os.path.join(args.checkpoint_dir, "config.json")
        builtin_config = os.path.join(
            os.path.dirname(__file__), "config", "qwen3_5_27b_config.json"
        )
        if os.path.exists(hf_config):
            config_path = hf_config
        else:
            config_path = builtin_config

    print(f"Loading config from {config_path}...")

    # Parse the config, handling HF format vs our format
    import json
    with open(config_path) as f:
        raw_config = json.load(f)

    # Map HF config fields to our config fields
    field_map = {
        "rope_theta": None,  # handled specially
    }
    config_kwargs = {}
    for key in Qwen3_5Config.__dataclass_fields__:
        if key in raw_config:
            config_kwargs[key] = raw_config[key]

    # Handle HF rope_parameters format
    if "rope_parameters" in raw_config and isinstance(raw_config["rope_parameters"], dict):
        rope_params = raw_config["rope_parameters"]
        if "rope_theta" in rope_params:
            config_kwargs["rope_theta"] = rope_params["rope_theta"]
        if "partial_rotary_factor" in rope_params:
            config_kwargs["partial_rotary_factor"] = rope_params["partial_rotary_factor"]

    config_kwargs["max_seq_len"] = args.max_seq_len
    config_kwargs["use_metal_attention"] = args.backend == "metal"

    config = Qwen3_5Config(**config_kwargs)
    print(f"  {config.num_hidden_layers} layers, hidden={config.hidden_size}, "
          f"vocab={config.vocab_size}")
    n_attn = sum(1 for t in config.layer_types if t == "full_attention")
    n_delta = sum(1 for t in config.layer_types if t == "linear_attention")
    print(f"  {n_attn} attention layers, {n_delta} DeltaNet layers")

    # Load model
    print(f"\nLoading model from {args.checkpoint_dir}...")
    t0 = time.time()
    model = load_model(args.checkpoint_dir, config)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # Quantize
    if args.quantize:
        print(f"\nQuantizing ({args.quantize})...")
        quantize_model(model, args.quantize, args.quantize_group_size)

    # Export
    print("\nExporting...")
    programs = export_model(model, config, args.backend)

    # Metadata for runner
    metadata = {
        "get_vocab_size": config.vocab_size,
        "get_max_seq_len": config.max_seq_len,
        "get_n_layers": config.num_hidden_layers,
        "get_hidden_size": config.hidden_size,
    }

    # Lower and save
    et = lower_to_executorch(programs, metadata, backend=args.backend)

    pte_path = os.path.join(args.output_dir, "qwen3_5.pte")
    print(f"\nSaving to {pte_path}...")
    with open(pte_path, "wb") as f:
        et.write_to_file(f)
    size_mb = os.path.getsize(pte_path) / (1024 * 1024)
    print(f"Saved {size_mb:.1f} MB")

    print("\nDone!")


if __name__ == "__main__":
    main()
