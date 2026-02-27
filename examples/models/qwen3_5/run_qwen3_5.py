#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Qwen3.5 Inference with ExecuTorch

Runs text generation using an exported .pte model (XNNPACK or Metal).
Since Qwen3.5 is exported with static seq_len=1 (decode-only), all input
processing is done token-by-token.

Example usage:
    python examples/models/qwen3_5/run_qwen3_5.py \
        --model_path ./qwen3_5_exports/qwen3_5.pte \
        --prompt "Hello, how are you?" \
        --tokenizer Qwen/Qwen3.5-27B
"""

import argparse
import sys
import time

import torch

try:
    import executorch.kernels.quantized  # noqa: F401
except Exception:
    pass

import torch as _torch
from executorch.extension.pybindings.portable_lib import _get_operator_names

if not any("quantized_decomposed" in op for op in _get_operator_names()):
    from pathlib import Path
    import site

    for sp in site.getsitepackages():
        candidates = list(
            Path(sp).glob("executorch/kernels/quantized/*quantized_ops_aot_lib*")
        )
        if candidates:
            _torch.ops.load_library(candidates[0])
            break

try:
    from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401
except Exception:
    pass

from executorch.extension.pybindings.portable_lib import _load_for_executorch


def prefill(module, token_ids, start_pos):
    """Prefill by feeding tokens one at a time (static seq_len=1 model)."""
    logits = None
    pos = start_pos
    for tid in token_ids:
        tokens = torch.tensor([[tid]], dtype=torch.long)
        input_pos = torch.tensor([pos], dtype=torch.long)
        logits = module.run_method("forward", [tokens, input_pos])[0]
        pos += 1
    return logits, pos


def decode_one(module, token_id, pos):
    """Generate one token."""
    tokens = torch.tensor([[token_id]], dtype=torch.long)
    input_pos = torch.tensor([pos], dtype=torch.long)
    logits = module.run_method("forward", [tokens, input_pos])[0]
    return logits, pos + 1


def main():
    parser = argparse.ArgumentParser(
        description="Run Qwen3.5 inference with ExecuTorch",
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to exported .pte file")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?",
                        help="Input prompt")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="HF tokenizer name or path (default: use simple encoding)")
    args = parser.parse_args()

    # --- Load PTE ---
    print(f"Loading PTE from: {args.model_path}")
    t_load = time.perf_counter()
    module = _load_for_executorch(args.model_path)
    print(f"PTE loaded in {time.perf_counter() - t_load:.2f}s")

    # Print method metadata
    try:
        meta = module.method_meta("forward")
        n_inputs = meta.num_inputs()
        print(f"Method 'forward': {n_inputs} inputs")
        for i in range(min(n_inputs, 5)):
            try:
                tm = meta.input_tensor_meta(i)
                print(f"  input[{i}]: shape={list(tm.sizes())}, dtype={tm.scalar_type()}")
            except Exception:
                pass
        n_outputs = meta.num_outputs()
        for i in range(min(n_outputs, 3)):
            try:
                tm = meta.output_tensor_meta(i)
                print(f"  output[{i}]: shape={list(tm.sizes())}, dtype={tm.scalar_type()}")
            except Exception:
                pass
    except Exception as e:
        print(f"Could not read method metadata: {e}")

    # --- Tokenizer ---
    tokenizer = None
    eos_ids = set()

    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
            if tokenizer.eos_token_id is not None:
                eos_ids.add(tokenizer.eos_token_id)
            for tok_str in ["<|endoftext|>", "<|im_end|>", "<|end|>"]:
                try:
                    tid = tokenizer.convert_tokens_to_ids(tok_str)
                    if tid != tokenizer.unk_token_id:
                        eos_ids.add(tid)
                except Exception:
                    pass
            print(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")
        except Exception as e:
            print(f"Could not load tokenizer: {e}")
            tokenizer = None

    if tokenizer is not None:
        input_ids = tokenizer.encode(args.prompt)
    else:
        input_ids = [ord(c) % 100 for c in args.prompt]
        eos_ids = {0}
        print(f"No tokenizer -- using dummy encoding ({len(input_ids)} tokens)")

    print(f"Prompt tokens: {len(input_ids)}")

    # --- Prefill ---
    print("\nPrefilling...")
    t_start = time.perf_counter()
    logits, pos = prefill(module, input_ids, 0)
    t_prefill = time.perf_counter()
    prefill_time = t_prefill - t_start
    print(f"Prefill done: {pos} tokens in {prefill_time:.3f}s "
          f"({pos / prefill_time:.1f} tok/s)")

    # --- Decode ---
    next_token = torch.argmax(logits[:, -1, :], dim=-1).item()

    print(f"\nPrompt: {args.prompt}")
    print("-" * 50)
    print("Response: ", end="", flush=True)

    generated_tokens = []
    generated_count = 0

    for _ in range(args.max_new_tokens):
        if next_token in eos_ids:
            break

        generated_tokens.append(next_token)
        generated_count += 1

        if tokenizer:
            text = tokenizer.decode([next_token], skip_special_tokens=False)
            print(text, end="", flush=True)

        logits, pos = decode_one(module, next_token, pos)

        if args.temperature <= 0:
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
        else:
            probs = torch.softmax(logits[:, -1, :] / args.temperature, dim=-1)
            next_token = torch.multinomial(probs.squeeze(0), 1).item()

    if not tokenizer:
        print(f"[{generated_count} tokens generated]", end="")

    t_end = time.perf_counter()
    gen_time = t_end - t_prefill

    print()
    print("-" * 50)
    print(f"Prompt tokens:    {len(input_ids)}")
    print(f"Generated tokens: {generated_count}")
    print(f"Prefill time:     {prefill_time:.3f}s")
    print(f"Prefill rate:     {len(input_ids) / prefill_time:.2f} tokens/sec")
    if gen_time > 0 and generated_count > 0:
        print(f"Decode time:      {gen_time:.3f}s")
        print(f"Decode rate:      {generated_count / gen_time:.2f} tokens/sec")
    print(f"Total time:       {t_end - t_start:.3f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
