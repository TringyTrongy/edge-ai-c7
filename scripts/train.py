import sys, os
from ultralytics import YOLO
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 30
m = YOLO(os.path.join(root, "weights", "yolo26n.pt"))
m.train(data=os.path.join(root, "data", "data.yaml"), epochs=epochs, imgsz=416, batch=16, device="cpu",
        workers=2, patience=12, project=os.path.join(root, "runs"), name="synthetic_v1", exist_ok=True,
        seed=0, plots=True, amp=False)
