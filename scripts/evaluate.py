"""Evaluate the trained detector on the held-out test split, export to ONNX, check accuracy
before/after export, benchmark ONNX Runtime on CPU, and collect wrong detections.

All numbers here come from the PC/container CPU running synthetic test renders.
They are NOT measurements from the Raspberry Pi or any camera.
"""
import os, sys, json, glob, time, platform
import numpy as np, cv2
from ultralytics import YOLO
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(ROOT, "runs", "synthetic_v1")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
os.makedirs(os.path.join(RES, "wrong_detections"), exist_ok=True)
NAMES = ["sct013", "robot"]
IMGSZ = 416
DATA = os.path.join(ROOT, "data", "data.yaml")
best = os.path.join(RUN, "weights", "best.pt")


def metrics(model, tag):
    r = model.val(data=DATA, split="test", imgsz=IMGSZ, device="cpu", batch=1, plots=(tag == "pytorch"),
                  project=os.path.join(ROOT, "runs"), name=f"test_{tag}", exist_ok=True, verbose=False)
    return {
        "precision": float(r.box.mp), "recall": float(r.box.mr),
        "mAP50": float(r.box.map50), "mAP50_95": float(r.box.map),
        "per_class": {NAMES[int(c)]: {"AP50": float(r.box.ap50[i]), "AP50_95": float(r.box.ap[i]),
                                       "P": float(r.box.p[i]), "R": float(r.box.r[i])}
                      for i, c in enumerate(r.box.ap_class_index)},
    }


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


def wrong_detections(model, conf=0.25):
    """Match predictions to ground truth at IoU 0.5; collect false positives, misses and wrong classes."""
    found = []
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "images", "test", "*.jpg"))):
        im = cv2.imread(f)
        h, w = im.shape[:2]
        gt = []
        for l in open(f.replace("/images/", "/labels/").replace(".jpg", ".txt")):
            c, x, y, bw, bh = l.split()
            x, y, bw, bh = float(x) * w, float(y) * h, float(bw) * w, float(bh) * h
            gt.append((int(c), [x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2]))
        p = model.predict(im, imgsz=IMGSZ, conf=conf, device="cpu", verbose=False)[0]
        pr = [(int(b.cls), float(b.conf), b.xyxy[0].tolist()) for b in p.boxes]
        used = set()
        for gc, gb in gt:
            best_j, best_i = None, 0
            for j, (pc, s, pb) in enumerate(pr):
                v = iou(gb, pb)
                if j not in used and v > best_i:
                    best_j, best_i = j, v
            if best_j is None or best_i < 0.5:
                found.append(dict(file=os.path.basename(f), kind="miss", cls=NAMES[gc], iou=round(best_i, 2),
                                  area=round((gb[2] - gb[0]) * (gb[3] - gb[1]) / (w * h), 3), gt=gt, pred=pr))
            else:
                used.add(best_j)
                if pr[best_j][0] != gc:
                    found.append(dict(file=os.path.basename(f), kind="wrong_class", cls=NAMES[gc],
                                      pred_cls=NAMES[pr[best_j][0]], iou=round(best_i, 2), gt=gt, pred=pr))
        for j, (pc, s, pb) in enumerate(pr):
            if j not in used and all(iou(pb, g[1]) < 0.5 for g in gt):
                found.append(dict(file=os.path.basename(f), kind="false_positive", cls=NAMES[pc], conf=round(s, 2),
                                  gt=gt, pred=pr))
    return found


def draw(f, rec):
    im = cv2.imread(os.path.join(ROOT, "data", "images", "test", f))
    for c, b in rec["gt"]:
        cv2.rectangle(im, tuple(int(v) for v in b[:2]), tuple(int(v) for v in b[2:]), (0, 200, 0), 2)
        cv2.putText(im, "GT " + NAMES[c], (int(b[0]), int(b[1]) - 4), 0, .5, (0, 200, 0), 1)
    for c, s, b in rec["pred"]:
        cv2.rectangle(im, tuple(int(v) for v in b[:2]), tuple(int(v) for v in b[2:]), (0, 0, 255), 2)
        cv2.putText(im, f"{NAMES[c]} {s:.2f}", (int(b[0]), int(b[3]) + 14), 0, .5, (0, 0, 255), 1)
    return im


def preprocess(im):
    r = cv2.resize(im, (IMGSZ, IMGSZ))
    return (r[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0)


def bench(onnx_path, warm=20, runs=100):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    s = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    files = sorted(glob.glob(os.path.join(ROOT, "data", "images", "test", "*.jpg")))
    x = preprocess(cv2.imread(files[0]))
    for _ in range(warm):
        s.run(None, {name: x})
    t = []
    for k in range(runs):
        x = preprocess(cv2.imread(files[k % len(files)]))
        a = time.perf_counter()
        s.run(None, {name: x})
        t.append((time.perf_counter() - a) * 1000)
    t = np.array(t)
    return {"warmup_runs": warm, "timed_runs": runs, "batch": 1, "imgsz": IMGSZ, "threads": 2,
            "median_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95)), "mean_ms": float(t.mean()),
            "fps_from_median": float(1000 / np.median(t)),
            "host": platform.platform(), "provider": "CPUExecutionProvider",
            "note": "inference only (preprocess excluded from timing), container CPU, NOT a Raspberry Pi"}


if __name__ == "__main__":
    model = YOLO(best)
    out = {"pytorch_test": metrics(model, "pytorch")}
    onnx_path = model.export(format="onnx", imgsz=IMGSZ, simplify=True, opset=17, dynamic=False)
    out["onnx_path"] = os.path.relpath(onnx_path, ROOT)
    out["onnx_size_mb"] = round(os.path.getsize(onnx_path) / 1e6, 2)
    out["pt_size_mb"] = round(os.path.getsize(best) / 1e6, 2)
    out["onnx_test"] = metrics(YOLO(onnx_path, task="detect"), "onnx")
    out["benchmark_onnx_cpu"] = bench(onnx_path)
    wd = wrong_detections(model)
    out["wrong_detection_counts"] = {k: sum(1 for w in wd if w["kind"] == k) for k in ("miss", "wrong_class", "false_positive")}
    wd.sort(key=lambda w: ({"wrong_class": 0, "false_positive": 1, "miss": 2}[w["kind"]], -w.get("conf", 0)))
    sel = wd[:10]
    for i, w in enumerate(sel):
        cv2.imwrite(os.path.join(RES, "wrong_detections", f"wrong_{i + 1:02d}_{w['kind']}_{w['file']}"), draw(w["file"], w))
    out["wrong_detections_listed"] = [{k: v for k, v in w.items() if k not in ("gt", "pred")} for w in sel]
    json.dump(out, open(os.path.join(RES, "results.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "wrong_detections_listed"}, indent=2))
