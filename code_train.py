# train_yolo_safe_lowmem.py
import torch
import torch.nn as nn
from ultralytics import YOLO
import logging
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clear_yolo_cache(data_yaml):
    """Xóa cache"""
    with open(data_yaml, "r") as f:
        data = yaml.safe_load(f)

    train_path = Path(data["train"]).parent
    val_path = Path(data["val"]).parent

    for path in [
        train_path.parent / "labels" / train_path.name,
        val_path.parent / "labels" / val_path.name,
    ]:
        cache_file = path / "labels.cache"
        if cache_file.exists():
            cache_file.unlink()
            logger.info(f"✅ Cleared cache: {cache_file}")


def train_yolo_lowmem(data_yaml, model_size="s", epochs=50):
    """Train YOLO - Low Memory Version"""

    clear_yolo_cache(data_yaml)

    device = 0 if torch.cuda.is_available() else "cpu"
    logger.info(f"🚀 Device: {device}")

    # Load model
    logger.info(f"📦 Loading YOLOv8{model_size}...")
    model = YOLO(f"yolov8{model_size}.pt")

    # ✅ TRAIN - Memory efficient
    logger.info(" Training (Low Memory Mode)...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=512,
        batch=12,
        device=device,
        workers=2,
        patience=20,
        # Loss & Optimizer
        box=7.0,
        cls=1.0,
        dfl=1.5,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        # Augmentation
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.2,
        degrees=5,
        translate=0.05,
        scale=0.3,
        flipud=0.1,
        fliplr=0.3,
        mosaic=0.7,
        mixup=0.1,
        # Regularization
        dropout=0.2,
        # Output
        project="/content/runs",
        name="yolo_fracture_lowmem",
        save=True,
        plots=True,
        verbose=True,
    )

    logger.info("🎉 Training complete!")
    return results


# CHẠY
if __name__ == "__main__":
    DATA_YAML = "/content/data.yaml"

    results = train_yolo_lowmem(data_yaml=DATA_YAML, model_size="n", epochs=110)
