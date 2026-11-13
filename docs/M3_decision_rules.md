# M3 decision rule sheet

| Item | Rule |
|---|---|
| Input | Per-frame detections from `edge/onnx_detector.py`: class and confidence for each box |
| Threshold | A frame counts as "seen" when an `sct013` box has confidence >= 0.25 |
| Debounce | ACTIVE after 3 consecutive "seen" frames. INACTIVE after 3 consecutive frames without it. One bad frame changes nothing (`m3/test_decide.py`) |
| Action | ACTIVE: `actuator.fire("sct013")` (relay, LED or conveyor-stop signal). INACTIVE: `actuator.release()` |
| Camera fails | No frame for 1.0 s: release the actuator (fail-safe). Written in `m3/decide.py` as `FRAME_TIMEOUT_S`; the timeout watchdog itself is not implemented in this repo |
| Other class | `robot` detections are ignored by the rule |

Status: the rule, the debounce and its 5 tests run on the PC (`python m3/test_decide.py`). The actuator in this repo is simulated. The real GPIO LEDs, servo and motor stop are in the Vizion repository (`pi/leds.py`, `pi/tracker.py`, `pi/motor_monitor/motor_watch.py`).
