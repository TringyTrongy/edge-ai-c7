# M2 target, trade-off, CPU vs accelerator and software stack

**Target (written after training, not before).** I did not record a target before training, so this is not a prediction that was checked. The target I now hold the model to: mAP50 >= 0.90 on the held-out test set, and under 50 ms per frame on the edge CPU so a 15 fps camera is not the bottleneck.

**Result.** PyTorch mAP50 0.9565, ONNX 0.9577 (`results/results.json`). 13.5 ms median and 19.2 ms p95 for ONNX Runtime on 2 threads of a PC-class CPU, 74 FPS from the median. Both numbers meet the target on the PC; neither is measured on the Pi.

**Trade-off in plain words.** ONNX is a bigger file than the PyTorch weights (9.76 MB against 5.41 MB, because the exported graph keeps float32 weights and extra nodes) but runs in ONNX Runtime without PyTorch, which is what a small device needs. Recall is 0.886 against precision 0.994: the model rarely invents a part but misses about 1 in 9, mostly when two parts overlap. For a stop signal that is the safer way round only if a missed part is cheaper than a false stop; for an inspection station it is the wrong way round, and the threshold would be lowered.

## CPU vs accelerator

| Path | Where | Measured | Result |
|---|---|---|---|
| ONNX Runtime, CPUExecutionProvider, 2 threads | PC CPU | Yes, `results/results.json` | 13.5 ms median |
| NCNN, CPU, 3 to 4 cores | Raspberry Pi 5 (Vizion, other model) | Yes, in the Vizion repo, `model/README.md` | about 41 to 57 ms per frame |
| Accelerator (TensorRT, Hailo, Coral, GPU) | none | No | The Pi 5 here has no accelerator; no accelerator figure exists |

## Software stack

```
camera / image folder -> OpenCV (read, resize, BGR->RGB)
  -> ONNX Runtime (CPUExecutionProvider) running best.onnx   [exported from Ultralytics YOLO26n]
  -> numpy decode + OpenCV NMS (edge/onnx_detector.py)
  -> debounce + rule (m3/decide.py)
  -> actuator (simulated here; GPIO via gpiozero on the Pi) -> MQTT topics (docs/M4_topics.md)
```
