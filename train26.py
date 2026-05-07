from multiprocessing import freeze_support
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_YAML = PROJECT_ROOT / "data" / "processed" / "data.yaml"
MODEL_PATH = PROJECT_ROOT / "runs" / "segment" / "yolo26s_seg_mvtec6_defect_v1" / "weights" / "best.pt"


def main():
    model = YOLO(str(MODEL_PATH))
    model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=1024,
        device="cuda",
        batch=-1,
        workers=12,
        name="yolo26s_seg_mvtec6_defect_v1",
    )
    model.export(format="onnx")
    model.val()


if __name__ == "__main__":
    freeze_support()
    main()
