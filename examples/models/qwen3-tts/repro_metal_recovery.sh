#!/usr/bin/env bash
set -euo pipefail

# Deterministic repro harness for qwen3-tts backend recovery work.
# Captures all commands and logs for side-by-side XNNPACK vs Metal validation.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXAMPLE_DIR="$ROOT_DIR/examples/models/qwen3-tts"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-TTS-12Hz-0.6B-Base}"
CONDA_ENV="${CONDA_ENV:-executorch}"
FIXED_CODES_LEN="${FIXED_CODES_LEN:-1200}"
QUANT_MODE="${QUANT_MODE:-8w}"
QUANT_GROUP_SIZE="${QUANT_GROUP_SIZE:-32}"
TEXT_PROMPT="${TEXT_PROMPT:-Metal recovery harness validation.}"
CODE_SOURCE="${CODE_SOURCE:-synthetic}" # synthetic|helper|path
INPUT_CODES_PATH="${INPUT_CODES_PATH:-}"
CODEGEN_SEED="${CODEGEN_SEED:-20260315}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-$EXAMPLE_DIR/repro_runs/$RUN_ID}"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"
NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$OUT_DIR/numba-cache}"
mkdir -p "$NUMBA_CACHE_DIR"

CODES_PATH="$OUT_DIR/repro_codes.bin"
CODE_META_PATH="$OUT_DIR/repro_codes.json"

XNN_EXPORT_DIR="$OUT_DIR/export_xnnpack_fp32"
METAL_EXPORT_DIR="$OUT_DIR/export_metal_fp32"
METAL_QUANT_EXPORT_DIR="$OUT_DIR/export_metal_${QUANT_MODE}"

XNN_WAV="$OUT_DIR/output_xnnpack_fp32.wav"
METAL_WAV="$OUT_DIR/output_metal_fp32.wav"
METAL_QUANT_WAV="$OUT_DIR/output_metal_${QUANT_MODE}.wav"

RUNNER_BIN="$ROOT_DIR/cmake-out/examples/models/qwen3-tts/qwen3_tts_runner"

run_logged() {
  local name="$1"
  shift
  local log_file="$LOG_DIR/${name}.log"
  echo "=== [$name] ===" | tee "$log_file"
  echo "$*" | tee -a "$log_file"
  "$@" 2>&1 | tee -a "$log_file"
}

run_logged_allow_fail() {
  local name="$1"
  shift
  local log_file="$LOG_DIR/${name}.log"
  echo "=== [$name] ===" | tee "$log_file"
  echo "$*" | tee -a "$log_file"
  set +e
  "$@" 2>&1 | tee -a "$log_file"
  local cmd_rc=${PIPESTATUS[0]}
  set -e
  echo "exit_code=$cmd_rc" | tee -a "$log_file"
  return "$cmd_rc"
}

echo "Repro output: $OUT_DIR"

case "$CODE_SOURCE" in
  path)
    if [[ -z "$INPUT_CODES_PATH" ]]; then
      echo "CODE_SOURCE=path requires INPUT_CODES_PATH." >&2
      exit 2
    fi
    cp "$INPUT_CODES_PATH" "$CODES_PATH"
    ;;
  helper)
    run_logged gen_codes_helper \
      conda run -n "$CONDA_ENV" env NUMBA_CACHE_DIR="$NUMBA_CACHE_DIR" python "$EXAMPLE_DIR/generate_codes.py" \
        --model-id-or-path "$MODEL_ID" \
        --text "$TEXT_PROMPT" \
        --language English \
        --output-codes "$CODES_PATH"
    if [[ -f "${CODES_PATH%.bin}.json" ]]; then
      cp "${CODES_PATH%.bin}.json" "$CODE_META_PATH"
    fi
    ;;
  synthetic)
    run_logged gen_codes_synthetic \
      conda run -n "$CONDA_ENV" env NUMBA_CACHE_DIR="$NUMBA_CACHE_DIR" python -c "
import json
import random
import struct
from pathlib import Path

meta = json.loads(Path('$EXAMPLE_DIR/qwen3_tts_artifacts/decoder_metadata.json').read_text())
codebook = int(meta['codebook_size'])
num_q = int(meta['num_quantizers'])
t_len = int('$FIXED_CODES_LEN')
rng = random.Random(int('$CODEGEN_SEED'))
vals = [rng.randrange(codebook) for _ in range(t_len * num_q)]
out = Path('$CODES_PATH')
out.parent.mkdir(parents=True, exist_ok=True)
with out.open('wb') as f:
    f.write(struct.pack('<ii', t_len, num_q))
    f.write(struct.pack(f'<{len(vals)}i', *vals))
"
    ;;
  *)
    echo "Unknown CODE_SOURCE='$CODE_SOURCE' (expected synthetic|helper|path)." >&2
    exit 2
    ;;
esac

run_logged export_xnnpack_fp32 \
  conda run -n "$CONDA_ENV" env NUMBA_CACHE_DIR="$NUMBA_CACHE_DIR" python "$EXAMPLE_DIR/export_qwen3_tts.py" \
    --converted-dir "$EXAMPLE_DIR/qwen3_tts_artifacts" \
    --backend xnnpack \
    --fixed-codes-len "$FIXED_CODES_LEN" \
    --output-dir "$XNN_EXPORT_DIR"

if run_logged_allow_fail export_metal_fp32 \
  conda run -n "$CONDA_ENV" env NUMBA_CACHE_DIR="$NUMBA_CACHE_DIR" python "$EXAMPLE_DIR/export_qwen3_tts.py" \
    --converted-dir "$EXAMPLE_DIR/qwen3_tts_artifacts" \
    --backend metal \
    --fixed-codes-len "$FIXED_CODES_LEN" \
    --output-dir "$METAL_EXPORT_DIR"; then
  METAL_EXPORT_RC=0
else
  METAL_EXPORT_RC=$?
fi

if run_logged_allow_fail export_metal_quant \
  conda run -n "$CONDA_ENV" env NUMBA_CACHE_DIR="$NUMBA_CACHE_DIR" python "$EXAMPLE_DIR/export_qwen3_tts.py" \
    --converted-dir "$EXAMPLE_DIR/qwen3_tts_artifacts" \
    --backend metal \
    --fixed-codes-len "$FIXED_CODES_LEN" \
    --qlinear "$QUANT_MODE" \
    --qlinear-group-size "$QUANT_GROUP_SIZE" \
    --output-dir "$METAL_QUANT_EXPORT_DIR"; then
  METAL_QUANT_EXPORT_RC=0
else
  METAL_QUANT_EXPORT_RC=$?
fi

run_logged run_xnnpack_fp32 \
  conda run -n "$CONDA_ENV" "$RUNNER_BIN" \
    --model_path "$XNN_EXPORT_DIR/model.pte" \
    --codes_path "$CODES_PATH" \
    --output_wav "$XNN_WAV"

if [[ -f "$METAL_EXPORT_DIR/model.pte" ]]; then
  if run_logged_allow_fail run_metal_fp32 \
    conda run -n "$CONDA_ENV" env ET_METAL_FLUSH_INTERVAL=0 ET_METAL_BUFFER_POOL_SIZE_MB=64 "$RUNNER_BIN" \
      --model_path "$METAL_EXPORT_DIR/model.pte" \
      --codes_path "$CODES_PATH" \
      --output_wav "$METAL_WAV"; then
    METAL_FP32_RC=0
  else
    METAL_FP32_RC=$?
  fi
else
  METAL_FP32_RC=99
fi

if [[ -f "$METAL_QUANT_EXPORT_DIR/model.pte" ]]; then
  if run_logged_allow_fail run_metal_quant \
    conda run -n "$CONDA_ENV" env ET_METAL_FLUSH_INTERVAL=0 ET_METAL_BUFFER_POOL_SIZE_MB=64 "$RUNNER_BIN" \
      --model_path "$METAL_QUANT_EXPORT_DIR/model.pte" \
      --codes_path "$CODES_PATH" \
      --output_wav "$METAL_QUANT_WAV"; then
    METAL_QUANT_RC=0
  else
    METAL_QUANT_RC=$?
  fi
else
  METAL_QUANT_RC=99
fi

{
  echo "run_id=$RUN_ID"
  echo "model_id=$MODEL_ID"
  echo "fixed_codes_len=$FIXED_CODES_LEN"
  echo "code_source=$CODE_SOURCE"
  echo "codegen_seed=$CODEGEN_SEED"
  echo "numba_cache_dir=$NUMBA_CACHE_DIR"
  echo "quant_mode=$QUANT_MODE"
  echo "quant_group_size=$QUANT_GROUP_SIZE"
  echo "metal_export_rc=$METAL_EXPORT_RC"
  echo "metal_quant_export_rc=$METAL_QUANT_EXPORT_RC"
  echo "xnn_wav=$XNN_WAV"
  echo "metal_wav=$METAL_WAV"
  echo "metal_quant_wav=$METAL_QUANT_WAV"
  echo "metal_fp32_rc=$METAL_FP32_RC"
  echo "metal_quant_rc=$METAL_QUANT_RC"
} >"$OUT_DIR/summary.txt"

echo "Done. Summary: $OUT_DIR/summary.txt"
if [[ "$METAL_EXPORT_RC" -ne 0 || "$METAL_QUANT_EXPORT_RC" -ne 0 || "$METAL_FP32_RC" -ne 0 || "$METAL_QUANT_RC" -ne 0 ]]; then
  echo "Metal path failures captured in logs under: $LOG_DIR"
  exit 1
fi
