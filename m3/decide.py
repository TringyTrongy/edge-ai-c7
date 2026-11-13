"""M3 decision rule: detections -> debounced state -> action.

Rule sheet (see docs/M3_decision_rules.md):
  input      per-frame detections from edge.onnx_detector (class, confidence)
  threshold  confidence >= 0.25 counts as "seen"
  debounce   a class is ACTIVE only after N=3 consecutive frames that contain it,
             and goes INACTIVE only after 3 consecutive frames without it
  action     ACTIVE sct013 -> actuator.fire("sct013"); INACTIVE -> actuator.release()
  fail-safe  if the camera gives no frame for FRAME_TIMEOUT_S, the actuator is released
"""
import time

N_ON = 3
N_OFF = 3
FRAME_TIMEOUT_S = 1.0
TARGET = "sct013"


class Debounce:
    def __init__(self, n_on=N_ON, n_off=N_OFF):
        self.n_on, self.n_off = n_on, n_off
        self.hit = self.miss = 0
        self.active = False

    def update(self, seen):
        """Feed one frame. Returns 'on', 'off' or None when the state changes."""
        if seen:
            self.hit, self.miss = self.hit + 1, 0
            if not self.active and self.hit >= self.n_on:
                self.active = True
                return "on"
        else:
            self.miss, self.hit = self.miss + 1, 0
            if self.active and self.miss >= self.n_off:
                self.active = False
                return "off"
        return None


class SimulatedActuator:
    """Stands in for the relay/LED on the PC. On the Pi, vizion/pi/leds.py drives GPIO17/27."""

    def __init__(self):
        self.on = False
        self.log = []

    def fire(self, reason=""):
        self.on = True
        self.log.append((time.perf_counter(), "fire", reason))

    def release(self):
        self.on = False
        self.log.append((time.perf_counter(), "release", ""))


def step(deb, act, dets):
    seen = any(d["cls"] == TARGET for d in dets)
    ev = deb.update(seen)
    if ev == "on":
        act.fire(TARGET)
    elif ev == "off":
        act.release()
    return ev
