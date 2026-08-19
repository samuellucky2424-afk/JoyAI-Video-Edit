#!/bin/bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$HERE/.env.local" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$HERE/.env.local"
  set +a
fi

JOYOMNI_CONDA_SH="${JOYOMNI_CONDA_SH:-}"
JOYOMNI_CONDA_ENV="${JOYOMNI_CONDA_ENV:-}"
if [ -n "$JOYOMNI_CONDA_ENV" ]; then
  : "${NVCC_PREPEND_FLAGS:=}" "${NVCC_APPEND_FLAGS:=}"
  export NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS
  if [ -n "$JOYOMNI_CONDA_SH" ]; then
    # shellcheck disable=SC1090
    source "$JOYOMNI_CONDA_SH"
  fi
  while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do conda deactivate; done
  conda activate "$JOYOMNI_CONDA_ENV"
  hash -r
fi

cd "$HERE"

CACHE_ROOT="${JOYOMNI_CACHE_ROOT:-$HERE/deps/cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$CACHE_ROOT/nv_compute}"
export TORCHINDUCTOR_FX_GRAPH_CACHE="${TORCHINDUCTOR_FX_GRAPH_CACHE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"

echo "JoyAI compile cache root: $CACHE_ROOT"
echo "JoyAI VAE compile: ${JOYOMNI_VAE_COMPILE:-1} (strict=${JOYOMNI_VAE_COMPILE_STRICT:-0})"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$HERE"

export JOYOMNI_FP8_IMG="${JOYOMNI_FP8_IMG:-1}"
export JOYOMNI_FP8_TXT="${JOYOMNI_FP8_TXT:-1}"
export JOYOMNI_CUDA_GRAPH="${JOYOMNI_CUDA_GRAPH:-1}"
export JOYOMNI_SAGE_ATTN="${JOYOMNI_SAGE_ATTN:-1}"
export JOYOMNI_TXT_PARALLEL="${JOYOMNI_TXT_PARALLEL:-1}"

RECORD_DIR="${JOYOMNI_RECORD_DIR:-$HERE/recordings}"
RECORD_ENABLED="${JOYOMNI_RECORD_ENABLED:-1}"
ONLINE_GATE_ENABLED="${JOYOMNI_ONLINE_GATE_ENABLED:-1}"
EXTRA_ARGS=()

case "${RECORD_ENABLED,,}" in
  1|true|yes|on)
    EXTRA_ARGS+=(--record-dir "$RECORD_DIR")
    ;;
  0|false|no|off)
    ;;
  *)
    echo "JOYOMNI_RECORD_ENABLED must be one of: 1, 0, true, false, yes, no, on, off" >&2
    exit 2
    ;;
esac

case "${ONLINE_GATE_ENABLED,,}" in
  1|true|yes|on)
    ;;
  0|false|no|off)
    EXTRA_ARGS+=(--no-online-gate)
    ;;
  *)
    echo "JOYOMNI_ONLINE_GATE_ENABLED must be one of: 1, 0, true, false, yes, no, on, off" >&2
    exit 2
    ;;
esac

CKPT_ROOT="${JOYOMNI_CKPT_ROOT:-$HERE/deps/checkpoints}"
DIT_CKPT="${JOYOMNI_DIT_CKPT:-$CKPT_ROOT/JoyAI-Video-Edit/dit/joyai_video_edit_dit_0811.pth}"
VAE_CKPT="${JOYOMNI_VAE_CKPT:-$CKPT_ROOT/JoyAI-Video-Edit/vae}"
TE_CKPT="${JOYOMNI_TEXT_ENCODER_CKPT:-$CKPT_ROOT/MiMo-VL-7B-RL-2508}"
FACE_ONNX="${JOYOMNI_FACE_ONNX:-$CKPT_ROOT/face_detection_yunet_2023mar.onnx}"
PERSON_ONNX="${JOYOMNI_PERSON_ONNX:-$CKPT_ROOT/yolov8n.onnx}"

DEVICE="${JOYOMNI_DEVICE:-cuda:0}"
HOST="${JOYOMNI_HOST:-0.0.0.0}"
PORT="${JOYOMNI_PORT:-8080}"

python xvideo/serving/serve_joyomni_streaming.py \
  --dit-ckpt          "$DIT_CKPT" \
  --vae-ckpt          "$VAE_CKPT" \
  --text-encoder-ckpt "$TE_CKPT" \
  --face-detector-onnx   "$FACE_ONNX" \
  --person-detector-onnx "$PERSON_ONNX" \
  --device "$DEVICE" \
  --vae-encode-device "$DEVICE" \
  --vae-decode-device "$DEVICE" \
  --vae-pseudo-device "$DEVICE" \
  --postprocess-device "$DEVICE" \
  --width "${JOYOMNI_WIDTH:-840}" --height "${JOYOMNI_HEIGHT:-480}" \
  --fps "${JOYOMNI_FPS:-24}" \
  --num-inference-steps "${JOYOMNI_NUM_INFERENCE_STEPS:-2}" \
  --host "$HOST" --port "$PORT" \
  "${EXTRA_ARGS[@]}" \
  "$@"
