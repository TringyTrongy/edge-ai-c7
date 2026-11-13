"""M3 two-stream pipeline: two sources run at once, one shared ONNX detector, FPS reported per stream.

Each stream is a thread that "captures" frames from a folder of test images (the PC has no camera),
puts them on a shared queue, and a worker thread runs detection + debounce + actuator per stream.
Measured on the PC CPU, not on the Pi.
"""
import os, sys, glob, json, time, threading, queue
import cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edge.onnx_detector import Detector, ROOT
from m3.decide import Debounce, SimulatedActuator, step

SECONDS = 10.0


def capture(name, files, q, stop, stats):
    i = 0
    while not stop.is_set():
        im = cv2.imread(files[i % len(files)]); i += 1
        try:
            q.put((name, time.perf_counter(), im), timeout=0.2)
        except queue.Full:
            stats[name]["dropped"] += 1
        time.sleep(1 / 15)                      # 15 fps source, like a webcam


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "data", "images", "test", "*.jpg")))
    sources = {"stream_A": files[0::2], "stream_B": files[1::2]}
    q, stop = queue.Queue(maxsize=8), threading.Event()
    stats = {k: {"frames": 0, "dropped": 0, "lat_ms": []} for k in sources}
    det = Detector()
    state = {k: (Debounce(), SimulatedActuator()) for k in sources}
    ts = [threading.Thread(target=capture, args=(k, v, q, stop, stats), daemon=True) for k, v in sources.items()]
    t0 = time.perf_counter()
    for t in ts: t.start()
    while time.perf_counter() - t0 < SECONDS:
        try:
            name, tcap, im = q.get(timeout=0.5)
        except queue.Empty:
            continue
        d, a = state[name]
        step(d, a, det(im))
        stats[name]["frames"] += 1
        stats[name]["lat_ms"].append((time.perf_counter() - tcap) * 1000)
    stop.set()
    dur = time.perf_counter() - t0
    out = {"seconds": round(dur, 1), "note": "PC CPU, folder-of-images sources at 15 fps each, not a camera or Pi", "streams": {}}
    for k, s in stats.items():
        l = sorted(s["lat_ms"])
        out["streams"][k] = {"frames": s["frames"], "fps": round(s["frames"] / dur, 1), "dropped_at_capture": s["dropped"],
                             "median_capture_to_result_ms": round(l[len(l) // 2], 1) if l else None}
    json.dump(out, open(os.path.join(ROOT, "results", "m3_two_stream.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
