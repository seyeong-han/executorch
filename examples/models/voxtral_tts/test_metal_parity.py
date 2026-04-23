"""Metal parity tests for Voxtral TTS — stub.

Mirrors test_cuda_parity.py for the Apple Metal (MPS) backend. Will SKIP on
non-Mac systems and SKIP if the Voxtral checkpoint isn't available, so this
file is safe to keep in the default test suite even before Metal lands.

The actual Metal model.py changes (MetalSDPA, _build_attn_mask additive form,
backend == "metal" branches in MistralDecoder/LMAttention) live in METAL_DEV.md
as a plan. Once those land, fill in the bodies marked `# TODO: when Metal
backend lands`.

Run:
    pytest -xvs examples/models/voxtral_tts/test_metal_parity.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

VOXTRAL_DIR_ENV = "VOXTRAL_TTS_MODEL_DIR"
DEFAULT_VOXTRAL_DIR = Path.home() / "models/mistralai/Voxtral-4B-TTS-2603"


def _voxtral_dir() -> Path | None:
    p = Path(os.environ.get(VOXTRAL_DIR_ENV, DEFAULT_VOXTRAL_DIR))
    return p if (p / "params.json").exists() else None


def _mps_available() -> bool:
    return (
        sys.platform == "darwin"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )


pytestmark = [
    pytest.mark.skipif(not _mps_available(), reason="Metal/MPS not available"),
]


# ---------------------------------------------------------------------------
# Sanity check — import the symbols Metal will need (skip if unimplemented)
# ---------------------------------------------------------------------------


def test_metal_symbols_present():
    """Metal helpers (MetalSDPA, _build_attn_mask) should exist in model.py.

    Until they're added, this skips with a clear message pointing to METAL_DEV.md.
    """
    import model  # noqa: E402

    missing = [
        sym for sym in ("MetalSDPA", "_build_attn_mask") if not hasattr(model, sym)
    ]
    if missing:
        pytest.skip(
            f"Metal helpers not yet implemented: {missing}. "
            f"See METAL_DEV.md for the porting plan."
        )


# ---------------------------------------------------------------------------
# Parity tests — only run once MetalSDPA lands. Modeled on test_cuda_parity.py
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def models():
    vdir = _voxtral_dir()
    if vdir is None:
        pytest.skip(
            f"Voxtral-4B-TTS-2603 checkpoint not found "
            f"(set ${VOXTRAL_DIR_ENV} or place at {DEFAULT_VOXTRAL_DIR})"
        )
    from model import load_model  # noqa: E402

    cpu = load_model(
        str(vdir), max_seq_len=4096, dtype=torch.float32, backend="xnnpack"
    )
    cpu.eval()
    try:
        metal_model = load_model(
            str(vdir), max_seq_len=4096, dtype=torch.float32, backend="metal"
        )
    except (ValueError, AttributeError) as e:
        pytest.skip(f"Metal backend not yet wired in load_model: {e}")
    metal_model.to("mps").eval()
    return cpu, metal_model


def test_prefill_hidden_parity(models):
    """Metal decoder prefill matches XNNPACK baseline.

    Cosine threshold 0.998 — set by Metal-MPS SDPA fp32 vs the LM's
    F.scaled_dot_product_attention path. Tighten if MetalSDPA is bit-exact.
    """
    import torch.nn.functional as F  # noqa: E402

    cpu, metal_model = models
    torch.manual_seed(42)
    embeds = torch.randn(1, 230, 3072, dtype=torch.float32)
    pos = torch.arange(230, dtype=torch.long)

    with torch.no_grad():
        h_cpu = cpu.decoder(embeds, pos)
        h_metal = metal_model.decoder(embeds.to("mps"), pos.to("mps")).cpu()

    cos = F.cosine_similarity(h_cpu[0, -1], h_metal[0, -1], dim=0).item()
    assert cos > 0.998, f"prefill hidden cosine = {cos:.6f} (expected > 0.998)"


def test_first_frame_semantic_argmax_match(models):
    cpu, metal_model = models
    torch.manual_seed(42)
    embeds = torch.randn(1, 230, 3072, dtype=torch.float32)
    pos = torch.arange(230, dtype=torch.long)

    with torch.no_grad():
        h_cpu = cpu.decoder(embeds, pos)[0, -1].unsqueeze(0)
        h_metal = metal_model.decoder(embeds.to("mps"), pos.to("mps"))[0, -1].unsqueeze(
            0
        )
        sem_cpu = cpu.flow_head.semantic_logits(h_cpu)
        sem_metal = metal_model.flow_head.semantic_logits(h_metal).cpu()

    argmax_cpu = sem_cpu[0].argmax().item()
    argmax_metal = sem_metal[0].argmax().item()
    assert (
        argmax_cpu == argmax_metal
    ), f"semantic argmax mismatch: cpu={argmax_cpu} metal={argmax_metal}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-xvs"]))
