# C7 task map

Every task in the Edge AI & Embedded Vision individual sheet, the closest thing in my two repositories, and what state it is in. Vizion is `https://github.com/TringyTrongy/vizion` (the Raspberry Pi work; it detects resistors and capacitors, not the IDRP part).

Status key: **Ran on PC** = code here, run, numbers in `results/`. **Written, not run** = design, config or code that has not been run. **Pi, other object** = ran on the Pi in the Vizion repo, on resistors/capacitors. **Not done** = nothing yet.

| Task | Closest evidence | Status |
|---|---|---|
| M1 Camera and repo | Vizion `pi/README.md`, `capture_20261006_*.jpg` | Pi, other object |
| M1 OpenCV pipeline | `edge/onnx_detector.py`; Vizion `pi/detector.py` | Ran on PC; Pi, other object |
| M1 Baseline speed | `results/results.json` (13.5 ms median); Vizion `model/README.md` (41 to 57 ms on the Pi) | Ran on PC; Pi, other object |
| M1 Dataset v1 | `data/`, `scripts/render_dataset.py` | Ran on PC (synthetic renders) |
| M1 Correct split | `data/data.yaml`, `data/manifest.json` (560 / 160 / 80) | Ran on PC |
| M1 Train and run detector | `runs/synthetic_v1/`, `results/wrong_detections/` | Ran on PC |
| M2 Export to ONNX | `runs/synthetic_v1/weights/best.onnx`, `scripts/evaluate.py` | Ran on PC |
| M2 Convert and run live | none (no TFLite INT8 or TensorRT on a device) | Not done |
| M2 Accuracy check | `results/results.json` (mAP50 0.9565 vs 0.9577) | Ran on PC |
| M2 Fair benchmark | `results/results.json` (20 warm-up, 100 timed, median, p95, FPS, size) | Ran on PC |
| M2 CPU vs accelerator + stack | `docs/M2_target_and_stack.md` | Ran on PC for CPU; no accelerator figure |
| M2 Own target and trade-off | `docs/M2_target_and_stack.md` | Written after training, not before |
| M3 Decision rule sheet | `docs/M3_decision_rules.md` | Written; rule and tests ran on PC |
| M3 Two-stream pipeline | `m3/pipeline.py`, `results/m3_two_stream.json` | Ran on PC (image folders, not cameras) |
| M3 Debounce | `m3/decide.py`, `m3/test_decide.py` (5 tests); Vizion `pi/tracker.py` | Ran on PC; Pi, other object |
| M3 Actuator fires | `m3/decide.py` (simulated); Vizion `pi/leds.py` (GPIO LEDs) | Simulated on PC; Pi, other object |
| M3 Docker | `m3/Dockerfile`, `m3/docker-compose.yml` | Written, not built or run |
| M3 End-to-end latency | `m3/latency.py`, `results/m3_latency.json` (30 events) | Ran on PC, simulated actuator |
| M4 Sensor data | Vizion `pi/sensor_monitor/export_csv.py` (DHT11, not MPU6050) | Pi, other sensor |
| M4 Anomaly model | Vizion `pi/motor_monitor/README.md`, `pi/sensor_monitor/reports/nab_evaluation.md` | Pi, other object |
| M4 False alarms | `m4/false_alarms.py`, `results/m4_false_alarms.json` (5 runs of 40 normal frames) | Ran on PC |
| M4 Secure MQTT | `m4/mosquitto.conf`, `m4/mqtt_publish.py`, `m4/mqtt_check.py` | Written, not run (no broker) |
| M4 Topics | `docs/M4_topics.md` | Written, not run |
| M4 PLC link | `docs/M4_plc_link.md` | Design only, no PLC |
| M4 Threat model | `docs/M4_threat_model.md` | Written |
| M4 Reliability | `docs/M4_reliability.md`; Vizion `pi/vizion-*.desktop` | Autostart on the Pi; no drift report, OTA or rollback |
| M5 Inspection rig | Vizion `pi/motor_monitor/motor_rig.py` | Pi, other object |
| M5 Pass/fail station | `m5/headline_metric.py` (per-image present/absent on 80 test images) | Ran on PC, not 20 physical parts |
| M5 Full chain on the device | `m3/pipeline.py` on the PC; Vizion `pi/README.md` on the Pi | Not run as one chain on the device |
| M5 Live dashboard | Vizion `pi/web/` | Pi, other object |
| M5 Product readiness | Vizion `pi/vizion-dashboard.desktop`, CSV logs | Pi, other object |
| M5 Headline metric | `results/m5_headline_metric.json` (78 of 80 correct) | Ran on PC |
| M5 Documents | this file, `README.md`, Vizion `README.md` | Written |
| M5 Live demo | none | Not done |

Every number in `results/` comes from the PC or container CPU running synthetic renders. None is from the Pi or a camera.
