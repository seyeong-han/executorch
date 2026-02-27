#!/usr/bin/env python3
"""
End-to-end test: export a tiny Qwen3.5 model and run inference.
Tests both XNNPACK and Metal backends.
"""

import os
import sys
import time

import torch

# Add executorch to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from examples.models.qwen3_5.model import Qwen3_5Config, Qwen3_5Model

TINY_CONFIG = Qwen3_5Config(
    hidden_size=64,
    num_hidden_layers=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    intermediate_size=128,
    vocab_size=100,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    max_seq_len=64,
)


def export_tiny(backend: str, output_path: str):
    """Export a tiny model with the given backend."""
    config = Qwen3_5Config(**{
        k: getattr(TINY_CONFIG, k)
        for k in TINY_CONFIG.__dataclass_fields__
    })
    config.use_metal_attention = (backend == "metal")

    model = Qwen3_5Model(config)
    model.eval()
    model.requires_grad_(False)

    tokens = torch.zeros(1, 1, dtype=torch.long)
    pos = torch.zeros(1, dtype=torch.long)

    print(f"\n{'='*60}")
    print(f"Exporting tiny Qwen3.5 with {backend.upper()} backend")
    print(f"{'='*60}")

    t0 = time.time()
    ep = torch.export.export(model, (tokens, pos), strict=False)
    print(f"  torch.export: {time.time() - t0:.1f}s ({len(ep.graph.nodes)} nodes)")

    from executorch.exir import (
        EdgeCompileConfig,
        ExecutorchBackendConfig,
        to_edge_transform_and_lower,
    )
    from executorch.exir.passes import MemoryPlanningPass

    programs = {"forward": ep}
    metadata = {"get_vocab_size": config.vocab_size, "get_max_seq_len": config.max_seq_len}

    if backend == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackDynamicallyQuantizedPartitioner,
            XnnpackPartitioner,
        )
        partitioner = {"forward": [XnnpackDynamicallyQuantizedPartitioner(), XnnpackPartitioner()]}

    elif backend == "metal":
        from executorch.backends.apple.metal.metal_backend import MetalBackend
        from executorch.backends.apple.metal.metal_partitioner import MetalPartitioner

        def _linear_bias_decomp(inp, weight, bias=None):
            wt = torch.ops.aten.t.default(weight)
            out = torch.ops.aten.matmul.default(inp, wt)
            if bias is not None:
                return torch.ops.aten.add.Tensor(out, bias)
            return out

        programs = {
            k: v.run_decompositions({torch.ops.aten.linear.default: _linear_bias_decomp})
            for k, v in programs.items()
        }
        compile_specs = [MetalBackend.generate_method_name_compile_spec("forward")]
        partitioner = {"forward": [MetalPartitioner(compile_specs)]}

    else:
        partitioner = {"forward": []}

    t0 = time.time()
    et_prog = to_edge_transform_and_lower(
        programs,
        partitioner=partitioner,
        compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True),
        constant_methods=metadata,
    )
    et = et_prog.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            do_quant_fusion_and_const_prop=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        ),
    )
    print(f"  Lowering: {time.time() - t0:.1f}s")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        et.write_to_file(f)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"  Saved: {output_path} ({size_kb:.1f} KB)")
    return output_path


def run_inference(pte_path: str, backend: str):
    """Run inference on an exported model."""
    from executorch.extension.pybindings.portable_lib import _load_for_executorch

    print(f"\n--- Running inference ({backend.upper()}) ---")

    if backend == "metal":
        from executorch.extension.pybindings.portable_lib import _get_operator_names
        metal_ops = [op for op in _get_operator_names() if "metal" in op.lower()]
        if not metal_ops:
            print("  SKIP: Metal backend not linked into Python portable_lib.")
            print("  The PTE file was exported successfully -- use a C++ runner")
            print("  built with metal_backend linked to run it.")
            return True

    t0 = time.perf_counter()
    module = _load_for_executorch(pte_path)
    print(f"  PTE loaded in {time.perf_counter() - t0:.3f}s")

    # Print method info
    try:
        meta = module.method_meta("forward")
        n_in = meta.num_inputs()
        n_out = meta.num_outputs()
        print(f"  Method 'forward': {n_in} inputs, {n_out} outputs")
        for i in range(min(n_in, 5)):
            try:
                tm = meta.input_tensor_meta(i)
                print(f"    input[{i}]: {list(tm.sizes())} {tm.scalar_type()}")
            except Exception:
                pass
    except Exception as e:
        print(f"  Metadata error: {e}")

    prompt_tokens = [42, 17, 88, 3, 55, 12, 67, 91, 23, 45]
    n_generate = 20

    # Prefill
    print(f"  Prefilling {len(prompt_tokens)} tokens...")
    t_prefill_start = time.perf_counter()
    logits = None
    pos = 0
    for tid in prompt_tokens:
        tokens = torch.tensor([[tid]], dtype=torch.long)
        input_pos = torch.tensor([pos], dtype=torch.long)
        logits = module.run_method("forward", [tokens, input_pos])[0]
        pos += 1
    t_prefill = time.perf_counter() - t_prefill_start
    print(f"  Prefill: {len(prompt_tokens)} tokens in {t_prefill:.3f}s "
          f"({len(prompt_tokens)/t_prefill:.1f} tok/s)")

    # Decode
    next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
    generated = [next_token]

    t_decode_start = time.perf_counter()
    for _ in range(n_generate - 1):
        tokens = torch.tensor([[next_token]], dtype=torch.long)
        input_pos = torch.tensor([pos], dtype=torch.long)
        logits = module.run_method("forward", [tokens, input_pos])[0]
        next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
        generated.append(next_token)
        pos += 1
    t_decode = time.perf_counter() - t_decode_start

    print(f"  Decode: {len(generated)} tokens in {t_decode:.3f}s "
          f"({len(generated)/t_decode:.1f} tok/s)")
    print(f"  Generated token IDs: {generated[:10]}...")
    print(f"  Total: {time.perf_counter() - t_prefill_start:.3f}s")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", nargs="+", default=["xnnpack"],
                        choices=["portable", "xnnpack", "metal"],
                        help="Backend(s) to test")
    parser.add_argument("--output-dir", default="/tmp/qwen3_5_test",
                        help="Output directory for PTE files")
    args = parser.parse_args()

    results = {}
    for backend in args.backend:
        pte_path = os.path.join(args.output_dir, f"qwen3_5_tiny_{backend}.pte")
        try:
            export_tiny(backend, pte_path)
            run_inference(pte_path, backend)
            results[backend] = "PASS"
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[backend] = f"FAIL: {e}"

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    for backend, result in results.items():
        print(f"  {backend:10s}: {result}")

    return 0 if all(r == "PASS" for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
