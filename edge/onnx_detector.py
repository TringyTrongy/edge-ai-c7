"""ONNX Runtime detector for the two-class model (sct013, robot). Shared by M3 to M5 scripts.

Output of the exported model is (1, 6, 3549): x, y, w, h, score_sct013, score_robot per anchor,
so confidence filtering and NMS are done here.
"""
import os
import cv2
import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "runs", "synthetic_v1", "weights", "best.onnx")
NAMES = ["sct013", "robot"]
IMGSZ = 416


class Detector:
    def __init__(self, path=MODEL, threads=2, conf=0.25, iou=0.5):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.inp = self.s.get_inputs()[0].name
        self.conf, self.iou = conf, iou

    def __call__(self, bgr):
        h, w = bgr.shape[:2]
        x = cv2.resize(bgr, (IMGSZ, IMGSZ))[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        p = self.s.run(None, {self.inp: x})[0][0].T          # (3549, 6)
        sc = p[:, 4:].max(1)
        keep = sc >= self.conf
        p, sc = p[keep], sc[keep]
        if not len(p):
            return []
        cls = p[:, 4:].argmax(1)
        sx, sy = w / IMGSZ, h / IMGSZ
        boxes = [[(b[0] - b[2] / 2) * sx, (b[1] - b[3] / 2) * sy, b[2] * sx, b[3] * sy] for b in p]
        idx = cv2.dnn.NMSBoxes(boxes, sc.tolist(), self.conf, self.iou)
        out = []
        for i in np.array(idx).reshape(-1):
            bx, by, bw, bh = boxes[i]
            out.append({"cls": NAMES[int(cls[i])], "conf": float(sc[i]), "box": [bx, by, bx + bw, by + bh]})
        return out
