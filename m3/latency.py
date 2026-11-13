"""M3 end-to-end latency: frame in -> detection -> debounce -> actuator call, over 20+ trigger events.

An event is a run of N_ON consecutive frames that contain an sct013. Latency is measured from the
arrival of the FIRST frame of the run (when the part appears) to the actuator call, and also from
the LAST frame (the frame that completes the debounce). Actuator is simulated: this is the software
path only, not a relay on the Pi.
"""
import os, sys, glob, json, time
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edge.onnx_detector import Detector, ROOT
from m3.decide import Debounce, SimulatedActuator, step, N_ON

EVENTS = 30


def has_target(label_file):
    return any(l.split()[0] == "0" for l in open(label_file))


def main():
    det = Detector()
    imgs = sorted(glob.glob(os.path.join(ROOT, "data", "images", "*", "*.jpg")))
    pos = [f for f in imgs if has_target(f.replace("/images/", "/labels/").replace(".jpg", ".txt"))]
    neg = [f for f in imgs if os.path.getsize(f.replace("/images/", "/labels/").replace(".jpg", ".txt")) == 0]
    for f in pos[:3]: det(cv2.imread(f))                       # warm-up
    first, last, per_frame = [], [], []
    rng = np.random.default_rng(0)
    for e in range(EVENTS):
        d, a = Debounce(), SimulatedActuator()
        for f in rng.choice(neg, 2):                         # empty frames before the part appears
            step(d, a, det(cv2.imread(f)))
        t_first = None
        for k in range(N_ON + 1):
            f = rng.choice(pos)
            t_in = time.perf_counter()
            if t_first is None: t_first = t_in
            im = cv2.imread(f)
            ev = step(d, a, det(im))
            per_frame.append((time.perf_counter() - t_in) * 1000)
            if ev == "on":
                t_act = a.log[-1][0]
                first.append((t_act - t_first) * 1000); last.append((t_act - t_in) * 1000)
                break
    out = {"events": len(last), "debounce_frames": N_ON,
           "per_frame_ms_median": round(float(np.median(per_frame)), 1),
           "frame_to_actuator_ms": {"median": round(float(np.median(last)), 1), "p95": round(float(np.percentile(last, 95)), 1), "max": round(float(max(last)), 1)},
           "first_frame_to_actuator_ms": {"median": round(float(np.median(first)), 1), "p95": round(float(np.percentile(first, 95)), 1)},
           "note": "PC CPU, ONNX Runtime 2 threads, includes JPEG read + resize + inference + NMS + debounce; simulated actuator, not GPIO"}
    json.dump(out, open(os.path.join(ROOT, "results", "m3_latency.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
