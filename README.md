# Scanner AI - Ultralytics YOLO to OpenVINO

Spectrum channel detector for 2G/3G/4G cellular frequencies. Converts trained Ultralytics YOLO models (.pt) to OpenVINO FP32 and runs inference without PyTorch dependencies.

## Quick Start

### 1. Place your trained models

```
2G_MODEL/best.pt
3G_4G_MODEL/best.pt
```

### 2. Convert to OpenVINO

```bash
pip install ultralytics
python export_openvino.py
```

This finds `.pt` files in `2G_MODEL/` and `3G_4G_MODEL/`, converts them to OpenVINO FP32, and outputs to `best_openvino_model/` in the same directory.

To re-export existing models:
```bash
python export_openvino.py --force
```

### 3. Run

**Docker (recommended):**
```bash
docker build -t scanner-ai .
docker run -p 4444:4444 -e SCANNER_AI_PORT=4444 -e SAVE_SAMPLES=NO -e MEM_OPTIMIZATION=YES scanner-ai
```

**Docker Compose:**
```bash
export SCANNER_AI_VERSION=latest
export SCANNER_AI_LOW_POWER_SAMPLES=/path/to/samples
docker compose up -d
```

**Direct:**
```bash
pip install -r requirements.txt
python scanner.py
```

## Model Directory Structure

After export, the directory structure should look like:

```
2G_MODEL/
    best.pt                    # trained model (input)
    best_openvino_model/       # converted model (output)
        best.xml
        best.bin
        metadata.yaml

3G_4G_MODEL/
    best.pt
    best_openvino_model/
        best.xml
        best.bin
        metadata.yaml
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SCANNER_AI_IP` | `0.0.0.0` | Host to bind to |
| `SCANNER_AI_PORT` | `4444` | TCP port |
| `SAVE_SAMPLES` | `NO` | Save spectrogram images |
| `MEM_OPTIMIZATION` | `YES` | Memory optimization for large bands |

## Testing

```bash
# Start scanner first
python scanner.py

# Run tests (requires sample data in SAMPLES_UT/)
pip install pytest
python -m pytest testing/ -v
```

## Architecture

```
Scanner AI Service (TCP :4444)
    |
    |-- Receives protobuf messages
    |-- Builds spectrogram from raw FFT samples
    |-- Applies viridis colormap
    |
    |-- 3G/4G Detection (OpenVINO FP32, 640x640)
    |       Classes: 3G, 4G, 4G-TDD
    |       Confidence threshold: 0.6
    |
    |-- 2G Detection (OpenVINO FP32)
    |       Classes: 2G
    |       Confidence threshold: 0.3
    |
    |-- Returns detected frequencies via protobuf
```
