# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Weight conversion from HuggingFace qwen3_next format to custom Qwen3_5Model format.
"""

from typing import Dict

import torch

from executorch.examples.models.qwen3_5.model import Qwen3_5Config


# HF weight name -> our model weight name
_ATTENTION_MAP = {
    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.token_mixer.q_proj.weight",
    "model.layers.{}.self_attn.k_proj.weight": "layers.{}.token_mixer.k_proj.weight",
    "model.layers.{}.self_attn.v_proj.weight": "layers.{}.token_mixer.v_proj.weight",
    "model.layers.{}.self_attn.o_proj.weight": "layers.{}.token_mixer.o_proj.weight",
    "model.layers.{}.self_attn.q_norm.weight": "layers.{}.token_mixer.q_norm.weight",
    "model.layers.{}.self_attn.k_norm.weight": "layers.{}.token_mixer.k_norm.weight",
}

_DELTANET_MAP = {
    "model.layers.{}.linear_attn.in_proj_qkvz.weight": "layers.{}.token_mixer.in_proj_qkvz.weight",
    "model.layers.{}.linear_attn.in_proj_ba.weight": "layers.{}.token_mixer.in_proj_ba.weight",
    "model.layers.{}.linear_attn.conv1d.weight": "layers.{}.token_mixer.conv1d.weight",
    "model.layers.{}.linear_attn.dt_bias": "layers.{}.token_mixer.dt_bias",
    "model.layers.{}.linear_attn.A_log": "layers.{}.token_mixer.A_log",
    "model.layers.{}.linear_attn.norm.weight": "layers.{}.token_mixer.norm.weight",
    "model.layers.{}.linear_attn.out_proj.weight": "layers.{}.token_mixer.out_proj.weight",
}

_LAYER_MAP = {
    "model.layers.{}.input_layernorm.weight": "layers.{}.input_layernorm.weight",
    "model.layers.{}.post_attention_layernorm.weight": "layers.{}.post_attention_layernorm.weight",
    "model.layers.{}.mlp.gate_proj.weight": "layers.{}.ffn.gate_proj.weight",
    "model.layers.{}.mlp.up_proj.weight": "layers.{}.ffn.up_proj.weight",
    "model.layers.{}.mlp.down_proj.weight": "layers.{}.ffn.down_proj.weight",
}

_GLOBAL_MAP = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "output.weight",
}


def convert_weights(
    hf_state_dict: Dict[str, torch.Tensor],
    config: Qwen3_5Config,
) -> Dict[str, torch.Tensor]:
    """Convert HuggingFace qwen3_next weights to Qwen3_5Model format."""
    converted = {}
    n_layers = config.num_hidden_layers

    # Global weights
    for hf_key, our_key in _GLOBAL_MAP.items():
        if hf_key in hf_state_dict:
            converted[our_key] = hf_state_dict[hf_key]

    for i in range(n_layers):
        layer_type = config.layer_types[i]

        # Layer norms and FFN (common to all layer types)
        for hf_pattern, our_pattern in _LAYER_MAP.items():
            hf_key = hf_pattern.format(i)
            our_key = our_pattern.format(i)
            if hf_key in hf_state_dict:
                converted[our_key] = hf_state_dict[hf_key]

        if layer_type == "full_attention":
            for hf_pattern, our_pattern in _ATTENTION_MAP.items():
                hf_key = hf_pattern.format(i)
                our_key = our_pattern.format(i)
                if hf_key in hf_state_dict:
                    converted[our_key] = hf_state_dict[hf_key]

        elif layer_type == "linear_attention":
            for hf_pattern, our_pattern in _DELTANET_MAP.items():
                hf_key = hf_pattern.format(i)
                our_key = our_pattern.format(i)
                if hf_key in hf_state_dict:
                    converted[our_key] = hf_state_dict[hf_key]

    return converted
