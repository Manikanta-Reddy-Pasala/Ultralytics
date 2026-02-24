# Scanner AI v3 - Ultralytics YOLO Spectrum Detector

Spectrum channel detector for 2G/3G/4G cellular frequencies. Uses Ultralytics YOLO for inference with memory-optimized processing pipeline.

## Architecture

```
Spectrogram (float32 FFT data)
  -> Normalize power values (-130 to -3 dBm)
  -> Apply Viridis colormap -> BGR uint8 image
  -> 3G/4G detection (YOLOv12n, Ultralytics + PyTorch)
  -> Extract 2G regions (gaps between 3G/4G detections)
  -> 2G detection (YOLO11n, Ultralytics + OpenVINO INT8)
  -> Convert pixel coordinates -> frequency (MHz)
  -> Return detected frequencies via protobuf over TCP
```

## Models

| Model | Architecture | Input | Classes | Format |
|-------|-------------|-------|---------|--------|
| 2G | YOLO11n | Dynamic (actual image size) | 2G (GSM) | OpenVINO INT8 |
| 3G/4G | YOLOv12n | 640x640 (Ultralytics default) | 3G, 4G, 4G-TDD | PyTorch (.pt) |

## Memory Optimization (v3)

This version eliminates memory leaks that caused unbounded growth under continuous operation:

| Fix | Before | After |
|-----|--------|-------|
| Model reload | Deleted & reloaded 2G model every request | Load once at startup, reuse |
| Input data handling | `np.copy(data)` duplicates entire array | Direct slicing `data[:, 357:1691]` |
| Spectrogram assembly | `np.append()` in loop (O(n^2) alloc) | Collect parts, single `np.concatenate()` |
| 2G image building | Repeated `np.concatenate()` in loop | Collect gap slices, single concat |
| Colormap objects | Created per-request | Singleton, reused across requests |
| Prediction lists | Global lists (thread-unsafe, leaked across requests) | Local per-request lists |
| Receive buffer | `chunks.append()` + `b''.join()` (2x memory) | Pre-allocated `bytearray` + `recv_into()` |
| Sample saving | `matplotlib plt.imsave()` (heavy import) | `cv2.imwrite()` |
| Large array lifecycle | Freed at end of function | `del` immediately after use |
| Thread concurrency | Unlimited threads | `Semaphore(MAX_THREADS)` limits parallel memory |
| glibc malloc | Default 8*ncores arenas | `MALLOC_ARENA_MAX=2` |

### Memory & Speed Profile (stress tested: 20 rounds x 6 bands = 120 inferences)

| Metric | Original | Optimized |
|--------|----------|-----------|
| Idle (startup) | ~1012 MiB | **794 MiB** |
| Steady state under load | Grew unbounded | **~1108 MiB (stable)** |
| Memory drift after 120 inferences | Grew every request | **0 MiB (no leak)** |
| Band 40 AI inference time | 0.83s | **0.94s** |
| Band 8 AI inference time | ~0.2s | **~0.17s** |

## Prerequisites

- Python 3.10
- Docker (recommended for production)
- Model files:
  - `2G_MODEL/best_int8_openvino_model/` (OpenVINO INT8 format)
  - `3G_4G_MODEL/best.pt` (PyTorch weights)
- `dummy.jpg` - warmup image (included in repo). Used to warm up models at startup for faster first inference. If missing, warmup is skipped with a warning.

## Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/Manikanta-Reddy-Pasala/Ultralytics.git
cd Ultralytics
```

### Step 2: Place model files

```
Ultralytics/
  dummy.jpg                      # Warmup image (included in repo)
  2G_MODEL/
    best_int8_openvino_model/    # OpenVINO INT8 model
      best.xml
      best.bin
      metadata.yaml
  3G_4G_MODEL/
    best.pt                      # PyTorch weights
```

### Step 3: Run

#### Docker (recommended)

```bash
docker build -t scanner-ai .
docker run -p 4444:4444 \
  -e SCANNER_AI_PORT=4444 \
  -e SAVE_SAMPLES=NO \
  -e MEM_OPTIMIZATION=YES \
  -e MAX_THREADS=2 \
  --memory=3g \
  scanner-ai
```

#### Docker Compose

```bash
docker compose up -d
```

#### Local development (uv)

```bash
uv sync
uv run python scanner.py
```

The scanner starts a TCP server on port `4444` and waits for spectrogram data.

### Step 4: Run tests

```bash
# Start scanner in background
MEM_OPTIMIZATION=YES python scanner.py &
sleep 30  # wait for model warmup

# Run tests
pytest testing/test_scanner_ai_script.py -v
```

Test samples cover 6 bands: B1, B3, B8, B20, B28, B40.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SCANNER_AI_IP` | `0.0.0.0` | Host to bind to |
| `SCANNER_AI_PORT` | `4444` | TCP port |
| `SAVE_SAMPLES` | `NO` | Save spectrogram images to `SAMPLES_LOW_POWER/` |
| `MEM_OPTIMIZATION` | `YES` | Split large bands (>100 MHz) into 60 MHz chunks with 12 MHz overlap |
| `MAX_THREADS` | `2` | Max concurrent inference threads |
| `OMP_NUM_THREADS` | `2` | OpenMP thread limit |
| `MKL_NUM_THREADS` | `2` | MKL thread limit |
| `MALLOC_ARENA_MAX` | `2` | glibc malloc arena limit |

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
Ultralytics/
  scanner.py             # Main service: TCP server + Ultralytics YOLO inference
  ai_colormap.py         # Viridis colormap and power normalization
  dummy.jpg              # Warmup image for model initialization
  viridis_colormap.py    # Colormap data
  ai_model_pb2.py        # Protobuf generated message definitions
  ai_model.proto         # Protobuf schema definition
  scanner_logging.py     # Logging configuration
  export_openvino.py     # Model export script (one-time, for 2G INT8)
  Dockerfile             # Multi-stage Docker build (uv + ubuntu)
  docker-compose.yml     # Docker Compose configuration
  pyproject.toml         # Python project config (uv/pip, CPU-only PyTorch)
  requirements.txt       # pip dependencies
  testing/
    test_scanner_ai_script.py  # Integration tests for all 6 bands
  2G_MODEL/              # 2G YOLO model (OpenVINO INT8)
  3G_4G_MODEL/           # 3G/4G YOLO model (PyTorch .pt)
```

## Runtime Dependencies

| Package | Purpose |
|---------|---------|
| `torch` (CPU) | PyTorch inference backend for 3G/4G model |
| `ultralytics` | YOLO model loading and inference |
| `openvino` | Inference engine for 2G INT8 model |
| `opencv-python-headless` | Image saving (samples) |
| `numpy` | Array operations |
| `Pillow` | Image utilities |
| `protobuf` | TCP message serialization |
