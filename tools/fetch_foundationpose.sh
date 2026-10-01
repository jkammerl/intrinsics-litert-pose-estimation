#!/usr/bin/env bash
# Downloads NVIDIA's FoundationPose models from NGC and converts them to the
# TFLite models in models/ that the pipeline uses:
#
#   foundationpose_refine.tflite         refiner, batches of 40 candidates
#   foundationpose_score.tflite          scorer, all 280 candidates
#   foundationpose_score_b<N>.tflite     scorers for the IOC service's chunks:
#                                        128 and 24 (OMTS), 240 and 40 (the
#                                        service's default batch size)
#   *_fp16.tflite                        float16 variants
#
# The models are covered by NVIDIA's Deep Learning Models License Agreement,
# which doesn't allow distributing them stand-alone, so this repository
# doesn't contain them. Usage:
#
#   tools/fetch_foundationpose.sh --accept-nvidia-license
#
# Environment: PYTHON (default python3.12) creates the conversion venv at
# VENV (default ./venv) from tools/requirements.txt unless it exists; OUT
# (default ./models) is where the models go; BATCHES the scorer batch sizes.
# Takes about 25 minutes on 6 arm64 cores and 8 GB of memory.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$ROOT/models}"
PYTHON="${PYTHON:-python3.12}"
VENV="${VENV:-$ROOT/venv}"
BATCHES="${BATCHES:-280,128,24,240,40}"

# FoundationPose 1.0.0_onnx on NGC, with the checksums that OMTS pins
# (intrinsic-omts third_party/foundationpose/deps.bzl).
NGC="https://api.ngc.nvidia.com/v2/models/nvidia/isaac/foundationpose/versions/1.0.0_onnx/files"
REFINE_SHA256="06ad19f2c3598cb76733feec084d3f6802e7ff143882ec42ba368df7e38ae094"
SCORE_SHA256="49a4f5f094358913670733ec31e856b96271c869f9949aa3a0361cf7cf8f0be8"
LICENSE_URL="https://developer.download.nvidia.com/licenses/tao_toolkit_21-08_models_eula.pdf"

if [[ "${1:-}" != "--accept-nvidia-license" ]]; then
  cat >&2 <<EOF
The FoundationPose models (https://catalog.ngc.nvidia.com/orgs/nvidia/teams/isaac/models/foundationpose)
are covered by NVIDIA's Deep Learning Models License Agreement:
  $LICENSE_URL
Read it, then run: $0 --accept-nvidia-license
EOF
  exit 1
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fetch() {  # url file sha256
  echo "Downloading $1"
  curl -fsSL --retry 3 -o "$WORK/$2" "$1"
  echo "$3  $WORK/$2" | sha256sum --check --quiet -
}
fetch "$NGC/refine_model.onnx" foundationpose_refine.onnx "$REFINE_SHA256"
fetch "$NGC/score_model.onnx" foundationpose_score.onnx "$SCORE_SHA256"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Creating the conversion venv in $VENV"
  "$PYTHON" -m venv "$VENV"
  "$VENV/bin/pip" install --quiet -r "$ROOT/tools/requirements.txt"
fi

mkdir -p "$OUT"
cd "$ROOT/tools"
"$VENV/bin/python" convert.py --only_foundationpose \
    --refine_onnx="$WORK/foundationpose_refine.onnx" \
    --score_onnx="$WORK/foundationpose_score.onnx" \
    --score_batches="$BATCHES" --out="$OUT"
# Embeds each model's tensor spec (and regenerates $OUT/README.md).
"$VENV/bin/python" add_metadata.py --models="$OUT"
echo "FoundationPose models written to $OUT"
