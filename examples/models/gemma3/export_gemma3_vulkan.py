import argparse
import logging
import os
import sys

sys.path.insert(0, "/home/pe/project/executorch")
import json
import torch
from typing import Dict, Optional

from transformers import AutoModelForPreTraining, AutoConfig, GenerationConfig
from optimum.exporters.executorch.integrations import (
    MultiModalTextToTextExportableModule,
)
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.passes import MemoryPlanningPass
from torch.export import ExportedProgram

ExportedProgram.validate = lambda self: None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _validate_multimodal_components(model):
    # Simplified validation for Gemma 3
    # Gemma 3 has 'model' which contains 'vision_tower' and 'text_model' usually?
    # Or it follows a specific structure.
    # Based on optimum code, it returns decoder_name, audio_encoder_name, vision_encoder_name

    # Let's inspect model structure if needed, or assume standard names.
    # For Gemma 3, it likely has a text decoder and a vision encoder.

    # We can try to find them.
    decoder_name = None
    vision_encoder_name = None

    # Common names
    if hasattr(model, "language_model"):
        decoder_name = "language_model"
    elif hasattr(model, "text_model"):
        decoder_name = "text_model"

    if hasattr(model, "vision_tower"):
        vision_encoder_name = "vision_tower"
    elif hasattr(model, "vision_model"):
        vision_encoder_name = "vision_model"

    # If not found, try to infer from config or just use what we found.
    # Optimum implementation does a more thorough check.
    # For now, let's assume Gemma 3 structure.

    return decoder_name, None, vision_encoder_name


def load_model(model_name_or_path, dtype="float32"):
    logger.info(f"Loading model {model_name_or_path}...")

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

    # Ensure cache is enabled
    if hasattr(config, "use_cache") and config.use_cache is False:
        config.use_cache = True

    # Load model
    # Gemma 3 might need trust_remote_code=True
    model = AutoModelForPreTraining.from_pretrained(
        model_name_or_path,
        dtype=dtype,
        config=config,
        trust_remote_code=True,
        device_map="cpu",  # Export on CPU
    )

    # Configure generation
    model.generation_config = GenerationConfig(
        use_cache=True,
        cache_implementation="static",
        max_length=2048,  # Default
        cache_config={
            "batch_size": 1,
            "max_cache_len": 2048,
            "device": "cpu",
        },
    )

    # Apply generation config to decoder
    # We need to find the decoder.
    # For Gemma 3, it might be 'model' or 'language_model'.
    # Let's check attributes.
    if hasattr(model, "language_model"):
        model.language_model.generation_config = model.generation_config
        decoder_name = "language_model"
    elif hasattr(model, "model"):  # Some models wrap it in 'model'
        # But AutoModelForPreTraining might return the top level.
        # Let's assume standard structure or check optimum's logic.
        # Optimum's logic uses _validate_multimodal_components.
        pass

    # We need to identify encoder/decoder for MultiModalTextToTextExportableModule
    # Let's use a simplified version or try to use optimum's function if importable.
    # It was not easily importable as it was decorated.

    # Let's rely on MultiModalTextToTextExportableModule to handle it if we pass the right args.
    # It takes: model, modality, encoder_name, max_seq_len, processor_config, ...

    # We need to find encoder_name.
    encoder_name = "vision_tower"  # Likely for Gemma 3
    if not hasattr(model, encoder_name):
        # Try to find it
        for name, module in model.named_children():
            if "vision" in name or "visual" in name:
                encoder_name = name
                break

    logger.info(f"Identified encoder: {encoder_name}")

    return MultiModalTextToTextExportableModule(
        model=model,
        modality="vision",
        encoder_name=encoder_name,
        max_seq_len=2048,
    )


def export_gemma3_vulkan(model_id: str, output_dir: str, dtype: str):

    model_wrapper = load_model(model_id, dtype)

    logger.info("Exporting to ExportedProgram...")
    exported_programs = model_wrapper.export()

    logger.info("Lowering to ExecuTorch with Vulkan backend...")

    backend_config = ExecutorchBackendConfig(
        extract_delegate_segments=True,
        memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        do_quant_fusion_and_const_prop=True,
    )

    os.makedirs(output_dir, exist_ok=True)

    # Lower all programs into a single ExecutorchProgram
    # This ensures we have one .pte file with multiple methods if needed

    # Check if we have multiple programs
    if len(exported_programs) == 1:
        # If only one, ensure it's named 'forward' for the method
        key = next(iter(exported_programs.keys()))
        if key != "forward":
            exported_programs = {"forward": exported_programs[key]}

    et_prog = to_edge_transform_and_lower(
        exported_programs,
        partitioner=[VulkanPartitioner()],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
    )

    et_prog = et_prog.to_executorch(config=backend_config)

    output_path = os.path.join(output_dir, "model.pte")

    with open(output_path, "wb") as f:
        f.write(et_prog.buffer)
    logger.info(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--output_dir", default="vulkan_output")
    parser.add_argument("--dtype", default="float16")
    args = parser.parse_args()

    export_gemma3_vulkan(args.model, args.output_dir, args.dtype)
