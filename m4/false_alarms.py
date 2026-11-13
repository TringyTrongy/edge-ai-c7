"""M4 false alarms: 5 "normal" runs, no sct013 present anywhere. Any actuator fire is a false alarm.

Normal frames are the images with an empty label file (background only, with distractors and noise)
plus frames that only contain the other class (robot). Each run is a shuffled 40-frame sequence.
Raw false positives (frames where sct013 is detected) are also counted, to show what the debounce removes.
"""
import os, sys, glob, json
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edge.onnx_detector import Detector, ROOT
from m3.decide import Debounce, SimulatedActuator, step

CONF = [0.25, 0.5]


def classes(lf):
    return {l.split()[0] for l in open(lf)}


def main():
    imgs = sorted(glob.glob(os.path.join(ROOT, "data", "images", "*", "*.jpg")))
    normal = [f for f in imgs if "0" not in classes(f.replace("/images/", "/labels/").replace(".jpg", ".txt"))]
    cache = {}
    out = {"normal_frames_available": len(normal), "runs": 5, "frames_per_run": 40, "by_threshold": {}}
    for conf in CONF:
        det = Detector(conf=conf)
        rows = []
        for r in range(5):
            rng = np.random.default_rng(r)
            seq = rng.choice(normal, 40, replace=False)
            d, a = Debounce(), SimulatedActuator()
            raw = 0
            for f in seq:
                dets = det(cv2.imread(f))
                raw += any(x["cls"] == "sct013" for x in dets)
                step(d, a, dets)
            rows.append({"run": r + 1, "raw_frames_with_false_sct013": int(raw), "false_alarms_after_debounce": sum(1 for x in a.log if x[1] == "fire")})
        out["by_threshold"][str(conf)] = rows
    out["note"] = "PC CPU, synthetic renders; the threshold 0.25 is the one used by m3/decide.py"
    json.dump(out, open(os.path.join(ROOT, "results", "m4_false_alarms.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
