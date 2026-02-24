# Scanner AI v3 - OpenVINO Runtime

Spectrum channel detector for 2G/3G/4G cellular frequencies. Runs YOLO models exported to OpenVINO IR format. No PyTorch or Ultralytics dependency at runtime.

## Architecture

```
Spectrogram (float32 FFT data)
  -> Normalize power values (-130 to -3 dBm)
  -> Apply Viridis colormap -> BGR uint8 image
  -> 3G/4G detection (YOLOv12n, OpenVINO)
  -> Extract 2G regions (gaps between 3G/4G detections)
  -> 2G detection (YOLO11n, OpenVINO)
  -> Convert pixel coordinates -> frequency (MHz)
  -> Return detected frequencies via protobuf over TCP
```

## Memory Optimization (v3)

This version eliminates memory leaks that caused unbounded growth under continuous operation:

| Fix | Before | After |
|-----|--------|-------|
| Input data handling | `np.copy(data)` duplicates entire array | Direct slicing (zero-copy views) |
| Spectrogram assembly | `np.append()` in loop (O(n^2) alloc) | Single `np.concatenate()` (O(n)) |
| 2G image building | Repeated `np.concatenate()` in loop | Collect slices, single concat |
| OpenVINO Core | Two separate `ov.Core()` instances | Single shared instance |
| Colormap objects | Created per-request | Singleton, reused across requests |
| Prediction lists | Global lists (thread-unsafe, leaked across requests) | Local per-request lists |
| Receive buffer | `chunks.append()` + `b''.join()` (2x memory) | Pre-allocated `bytearray` + `recv_into()` |
| Large array lifecycle | Freed at end of function | `del` immediately after consumption |
| Thread concurrency | Unlimited threads | `Semaphore(MAX_THREADS)` limits parallel memory |
| glibc malloc | Default 8*ncores arenas | `MALLOC_ARENA_MAX=2` |

### Memory Profile (stress tested: 10 rounds x 6 bands)

| Phase | Memory |
|-------|--------|
| Idle (startup + warmup) | ~117 MiB |
| Under load (stable) | ~203 MiB |
| After 60 consecutive inferences | ~203 MiB (zero growth) |

## Prerequisites

- Python >= 3.10
- Docker (optional, for containerized deployment)
- `.pt` model files (trained YOLO weights):
  - `2G_MODEL/best.pt` (YOLO11n, 2G/GSM detection)
  - `3G_4G_MODEL/best.pt` (YOLOv12n, 3G/4G detection)

## Step-by-Step Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/Manikanta-Reddy-Pasala/UltralyticsToOpenvino.git
cd UltralyticsToOpenvino
```

### Step 2: Place your trained model weights

Copy your `.pt` files into the model directories:

```
UltralyticsToOpenvino/
  2G_MODEL/
    best.pt
  3G_4G_MODEL/
    best.pt
```

### Step 3: Export models to OpenVINO format (one-time)

This step requires `ultralytics` and converts `.pt` weights to OpenVINO IR format.

```bash
pip install ultralytics openvino
python export_openvino.py
```

After export, you should have:

```
2G_MODEL/
  best.pt                    # original weights (not needed at runtime)
  best_openvino_model/       # FP32
    best.xml
    best.bin
    metadata.yaml

3G_4G_MODEL/
  best.pt                    # original weights (not needed at runtime)
  best_openvino_model/       # FP32
    best.xml
    best.bin
    metadata.yaml
```

### Step 4: Run the scanner

#### Option A: Docker (recommended for production)

```bash
docker build -t scanner-ai .
docker run -p 4444:4444 \
  -e SCANNER_AI_PORT=4444 \
  -e SAVE_SAMPLES=NO \
  -e MEM_OPTIMIZATION=YES \
  -e MAX_THREADS=2 \
  --memory=2g \
  scanner-ai
```

#### Option B: Docker Compose

```bash
export SCANNER_AI_VERSION=latest
export SCANNER_AI_LOW_POWER_SAMPLES=/path/to/samples
docker compose up -d
```

#### Option C: Using uv (recommended for development)

```bash
uv sync
uv run python scanner.py
```

#### Option D: Using pip

```bash
pip install -r requirements.txt
MEM_OPTIMIZATION=YES python scanner.py
```

The scanner starts a TCP server on port `4444` and waits for spectrogram data.

### Step 5: Run tests

Make sure the scanner is running first, then:

```bash
# Start scanner in background
MEM_OPTIMIZATION=YES python scanner.py &
sleep 30  # wait for model warmup

# Run tests
pytest testing/test_scanner_ai_script.py -v
```

Test samples are in `SAMPLES_UT/` covering 6 bands: B1, B3, B8, B20, B28, B40.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SCANNER_AI_IP` | `0.0.0.0` | Host to bind to |
| `SCANNER_AI_PORT` | `4444` | TCP port |
| `SAVE_SAMPLES` | `NO` | Save spectrogram images to `SAMPLES_LOW_POWER/` |
| `MEM_OPTIMIZATION` | `YES` | Split large bands (>100 MHz) into 60 MHz chunks with 12 MHz overlap |
| `MAX_THREADS` | `2` | Max concurrent inference threads |
| `OMP_NUM_THREADS` | `2` | OpenMP thread limit (set in Dockerfile) |
| `OPENBLAS_NUM_THREADS` | `2` | OpenBLAS thread limit (set in Dockerfile) |
| `MALLOC_ARENA_MAX` | `2` | glibc malloc arena limit (set in Dockerfile) |

## Protocol

TCP socket on port 4444 using length-prefixed protobuf messages:

```
1. Client -> AIPredictSampleReq  (band params: center freq, bandwidth, num chunks, overlay)
2. Server <- AIPredictSampleRes  (acknowledgment with band ID)
3. Client -> AISampleDataReq     (float32 spectrum data)
4. Server <- AISampleDataRes     (detected frequencies: lte_freqs, umts_freqs, gsm_freqs)
```

## Project Structure

```
UltralyticsToOpenvino/
  scanner.py             # Main service: TCP server + OpenVINO inference pipeline
  ai_colormap.py         # Viridis colormap and power normalization
  viridis_colormap.py    # Colormap data
  ai_model_pb2.py        # Protobuf generated message definitions
  ai_model.proto         # Protobuf schema definition
  scanner_logging.py     # Logging configuration
  export_openvino.py     # Model export script (requires ultralytics, one-time use)
  Dockerfile             # Multi-stage Docker build (uv + ubuntu)
  docker-compose.yml     # Docker Compose configuration
  pyproject.toml         # Python project config (uv/pip)
  requirements.txt       # pip dependencies
  testing/
    test_scanner_ai_script.py  # Integration tests for all 6 bands
  SAMPLES_UT/            # Test sample data (B1, B3, B8, B20, B28, B40)
  2G_MODEL/              # 2G YOLO model weights and OpenVINO export
  3G_4G_MODEL/           # 3G/4G YOLO model weights and OpenVINO export
```

## Runtime Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `openvino` | >= 2024.0.0 | Inference engine |
| `opencv-python-headless` | >= 4.8.0 | Image preprocessing |
| `numpy` | >= 1.24.0 | Array operations |
| `protobuf` | >= 4.21.0 | TCP message serialization |
| `Pillow` | latest | Image utilities |

No PyTorch, no Ultralytics, no CUDA required at runtime.

## Models

| Model | Architecture | Input Shape | Classes | Format |
|-------|-------------|-------------|---------|--------|
| 2G | YOLO11n | 640x640 | 2G (GSM) | FP32 |
| 3G/4G | YOLOv12n | 640x640 | 3G (UMTS), 4G (LTE), 4G-TDD | FP32 |

## INT8 Quantization (Optional)

INT8 quantization can improve inference speed but requires **spectrogram-specific calibration data** for accurate results. Generic calibration (e.g., COCO images) causes weak 2G signal detection loss.

If you have spectrogram calibration data:

```bash
pip install nncf
yolo export model=2G_MODEL/best.pt format=openvino int8=True data=your_spectrogram_data.yaml
```
