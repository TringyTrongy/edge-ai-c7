"""M5 headline metric: "is an sct013 in the frame?" correct on N of the 80 held-out test images.

This is the pass/fail decision the M3 rule acts on, scored per image (ignoring boxes). Confusion matrix:
present/absent against detected/not detected. PC CPU, synthetic renders. Not a physical 20-part test.
"""
import os, sys, glob, json
import cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edge.onnx_detector import Detector, ROOT

det = Detector()
tp = fp = fn = tn = 0
for f in sorted(glob.glob(os.path.join(ROOT, "data", "images", "test", "*.jpg"))):
    present = any(l.split()[0] == "0" for l in open(f.replace("/images/", "/labels/").replace(".jpg", ".txt")))
    seen = any(d["cls"] == "sct013" for d in det(cv2.imread(f)))
    tp += present and seen; fn += present and not seen; fp += (not present) and seen; tn += (not present) and (not seen)
n = tp + fp + fn + tn
out = {"images": n, "correct": tp + tn, "accuracy": round((tp + tn) / n, 3),
       "confusion": {"present_detected": tp, "present_missed": fn, "absent_false_alarm": fp, "absent_clear": tn},
       "note": "per-image sct013 presence on the synthetic test split, PC CPU; not a physical test of 20 parts"}
json.dump(out, open(os.path.join(ROOT, "results", "m5_headline_metric.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
