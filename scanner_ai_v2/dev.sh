#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_NAME="cellularscanner/cellularscanner_ai"
IMAGE_TAG="${SCANNER_AI_VERSION:-local}"

usage() {
    cat <<'EOF'
Scanner AI v2 - Development Helper

Usage: ./dev.sh <command> [options]

Commands:
  convert           Convert .pt models to OpenVINO IR format
  run               Start the scanner AI service (port 4444)
  test              Run the service + pytest suite (CI-style)
  docker-build      Build Docker image (Dockerfile.python)
  docker-run        Build and run Docker container

Options:
  convert --int8    Also export INT8 quantized model for 2G

Examples:
  ./dev.sh convert
  ./dev.sh run
  ./dev.sh test
  ./dev.sh docker-build
  ./dev.sh docker-run
EOF
}

# ── Convert .pt models to OpenVINO ──────────────────────────────────────────
cmd_convert() {
    local do_int8=false
    for arg in "$@"; do
        [[ "$arg" == "--int8" ]] && do_int8=true
    done

    echo "==> Checking dependencies..."
    python3 -c "import ultralytics; import openvino" 2>/dev/null || {
        echo "    Installing ultralytics + openvino for export..."
        pip install ultralytics openvino
    }

    echo ""
    echo "==> Converting 2G model (FP32)..."
    if [[ -f "2G_MODEL/best.pt" ]]; then
        python3 -c "
from ultralytics import YOLO
model = YOLO('2G_MODEL/best.pt')
model.export(format='openvino', dynamic=True, half=False)
print('2G FP32 export done.')
"
        echo "    Output: 2G_MODEL/best_openvino_model/"
    else
        echo "    SKIP: 2G_MODEL/best.pt not found"
    fi

    if $do_int8; then
        echo ""
        echo "==> Converting 2G model (INT8)..."
        python3 -c "
from ultralytics import YOLO
model = YOLO('2G_MODEL/best.pt')
model.export(format='openvino', dynamic=True, half=False, int8=True)
print('2G INT8 export done.')
"
        echo "    Output: 2G_MODEL/best_int8_openvino_model/"
    fi

    echo ""
    echo "==> Converting 3G/4G model (FP32)..."
    if [[ -f "3G_4G_MODEL/best.pt" ]]; then
        python3 -c "
from ultralytics import YOLO
model = YOLO('3G_4G_MODEL/best.pt')
model.export(format='openvino', dynamic=True, half=False)
print('3G/4G FP32 export done.')
"
        echo "    Output: 3G_4G_MODEL/best_openvino_model/"
    else
        echo "    SKIP: 3G_4G_MODEL/best.pt not found"
    fi

    echo ""
    echo "==> All conversions complete."
}

# ── Run the service ─────────────────────────────────────────────────────────
cmd_run() {
    echo "==> Starting Scanner AI on port 4444..."
    export MEM_OPTIMIZATION="${MEM_OPTIMIZATION:-YES}"
    export SAVE_SAMPLES="${SAVE_SAMPLES:-NO}"
    export SCANNER_AI_PORT="${SCANNER_AI_PORT:-4444}"
    mkdir -p SAMPLES_LOW_POWER
    python3 scanner.py
}

# ── Run service + tests (CI-style) ─────────────────────────────────────────
cmd_test() {
    echo "==> Installing dependencies..."
    if command -v uv &>/dev/null; then
        uv sync --extra test
        local RUNNER="uv run"
    else
        pip install -r requirements.txt pytest coverage
        local RUNNER="python3 -m"
    fi

    echo "==> Starting scanner.py in background..."
    export MEM_OPTIMIZATION="${MEM_OPTIMIZATION:-YES}"
    export SAVE_SAMPLES="${SAVE_SAMPLES:-NO}"
    mkdir -p SAMPLES_LOW_POWER

    $RUNNER coverage run --parallel-mode scanner.py &
    local PID=$!

    echo "==> Waiting for service on port 4444..."
    local retries=0
    while ! nc -z 127.0.0.1 4444 2>/dev/null; do
        sleep 2
        retries=$((retries + 1))
        if [[ $retries -ge 60 ]]; then
            echo "    TIMEOUT: service did not start within 120s"
            kill "$PID" 2>/dev/null || true
            exit 1
        fi
    done
    echo "    Service ready."

    echo "==> Running pytest..."
    $RUNNER pytest testing/ -v || { kill -SIGTERM "$PID" 2>/dev/null; exit 1; }

    echo "==> Stopping service..."
    kill -SIGTERM "$PID" 2>/dev/null || true
    sleep 5

    echo "==> Collecting coverage..."
    $RUNNER coverage combine 2>/dev/null || true
    $RUNNER coverage report 2>/dev/null || true

    echo "==> Done."
}

# ── Docker build ────────────────────────────────────────────────────────────
cmd_docker_build() {
    echo "==> Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
    docker build -f Dockerfile.python -t "${IMAGE_NAME}:${IMAGE_TAG}" .
    echo "==> Image built: ${IMAGE_NAME}:${IMAGE_TAG}"
}

# ── Docker run ──────────────────────────────────────────────────────────────
cmd_docker_run() {
    cmd_docker_build
    echo "==> Running container on port 4444..."
    docker run --rm -it \
        -p 4444:4444 \
        -e SCANNER_AI_PORT=4444 \
        -e SAVE_SAMPLES=NO \
        -e MEM_OPTIMIZATION=YES \
        "${IMAGE_NAME}:${IMAGE_TAG}"
}

# ── Main ────────────────────────────────────────────────────────────────────
case "${1:-}" in
    convert)      shift; cmd_convert "$@" ;;
    run)          shift; cmd_run "$@" ;;
    test)         shift; cmd_test "$@" ;;
    docker-build) shift; cmd_docker_build "$@" ;;
    docker-run)   shift; cmd_docker_run "$@" ;;
    -h|--help|"") usage ;;
    *)            echo "Unknown command: $1"; usage; exit 1 ;;
esac
