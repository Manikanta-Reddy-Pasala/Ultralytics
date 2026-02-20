"""
Export trained YOLO models (.pt) to OpenVINO FP32 format.

Scans for .pt files in 2G_MODEL/ and 3G_4G_MODEL/ directories and converts
them to OpenVINO IR format (FP32) for accurate inference without ultralytics.

Usage:
    python export_openvino.py                    # convert all models found
    python export_openvino.py --imgsz 640        # custom image size for both
"""

import argparse
import os
import glob

from ultralytics import YOLO


MODEL_DIRS = ["2G_MODEL", "3G_4G_MODEL"]


def find_pt_files(root_dir):
    """Find .pt files in model directories."""
    found = []
    for model_dir in MODEL_DIRS:
        dir_path = os.path.join(root_dir, model_dir)
        if not os.path.isdir(dir_path):
            continue
        pt_files = glob.glob(os.path.join(dir_path, "*.pt"))
        if pt_files:
            found.append((model_dir, pt_files[0]))
    return found


def export_model(pt_path, imgsz):
    """Export a single .pt model to OpenVINO FP32."""
    print(f"\n  Loading: {pt_path}")
    model = YOLO(pt_path)
    print(f"  Classes: {model.names}")
    print(f"  Exporting to OpenVINO FP32 (imgsz={imgsz})...")

    export_path = model.export(
        format="openvino",
        imgsz=imgsz,
        half=False,
        int8=False,
    )

    # Print model size
    model_dir = export_path if os.path.isdir(export_path) else os.path.dirname(export_path)
    total_size = sum(
        os.path.getsize(os.path.join(model_dir, f))
        for f in os.listdir(model_dir)
        if os.path.isfile(os.path.join(model_dir, f))
    )
    print(f"  Output:  {model_dir}")
    print(f"  Size:    {total_size / 1024 / 1024:.1f} MB")
    return model_dir


def main():
    parser = argparse.ArgumentParser(description="Export YOLO .pt models to OpenVINO FP32")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size for export (default: 640)")
    parser.add_argument("--force", action="store_true",
                        help="Re-export even if OpenVINO model already exists")
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.abspath(__file__))
    models = find_pt_files(root_dir)

    if not models:
        print("ERROR: No .pt files found in 2G_MODEL/ or 3G_4G_MODEL/")
        print("  Place your trained models as:")
        print("    2G_MODEL/best.pt")
        print("    3G_4G_MODEL/best.pt")
        return

    print("=" * 50)
    print(" Export YOLO Models to OpenVINO FP32")
    print("=" * 50)

    for model_dir, pt_path in models:
        ov_dir = os.path.join(os.path.dirname(pt_path), "best_openvino_model")
        if os.path.isdir(ov_dir) and not args.force:
            print(f"\n  {model_dir}: Already exported (use --force to re-export)")
            continue
        export_model(pt_path, args.imgsz)

    print(f"\n{'=' * 50}")
    print(" Export Complete")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
