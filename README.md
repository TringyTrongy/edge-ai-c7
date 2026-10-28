# Edge AI C7: detecting two CAD parts (PC-side work, synthetic data)

Individual assessment for Edge AI & Embedded Vision Systems, by Viswes Vijayaragavan Sivakumar (24BCS321).

## Read this first: what this repository is and is not

This is **PC-side work only**. There is **no Raspberry Pi, no camera and no printed part** in anything below.

- The training images are **synthetic**: they are rendered on a PC from my own Fusion 360 CAD exports and pasted on generated backgrounds. They are not photos.
- Every speed number was measured on a **cloud container CPU (2 threads)**, not on an edge device.
- Nothing here is a live demo. Device tasks (live overlay, FPS/temperature on the device, TFLite INT8 on the device, relay/LED, Docker cold boot, MPU6050, MQTT, PLC, inspection rig, dashboard, auto-start) are **not done** and are not claimed.

When the device and the printed part are available, the same pipeline can be re-run on real photos.

## Objects

| Class | Source CAD (in `cad/`) | What it is |
|---|---|---|
| `0 sct013` | `SCT-013-000_current_transformer.f3d` / `.stl` | My IDRP A6 part (SCT-013 current transformer, Fusion 360) |
| `1 robot` | `my_robot.f3d`, `robot_base_link.stl` | The chassis of my Fusion 360 differential-drive robot (from the ROS 2 A2 work) |

The brief asks for the IDRP CAD plus a printed part. Only the SCT-013 is from IDRP; the robot chassis is my second Fusion model. The printed part does not exist yet.

## M1 Detect: dataset and detector

**Dataset v1** (`data/`, YOLO format, 640x480 JPEG, 800 images). `scripts/render_dataset.py` loads each STL, draws it with random orientation, distance, colour, light and roughness (pyrender, software OpenGL), and pastes it on a random background (flat, gradient, noise, wood-like stripes, grid) with random blur, distractor shapes and sensor noise. 0 to 3 objects per image; about 6% of images are empty backgrounds. Boxes are computed from each object's rendered alpha mask, so they are tight by construction and not hand drawn. `scripts/preview.py` draws them for checking (`results/dataset_preview.jpg`).

**Split 70/20/10, no leakage.** Image indices are shuffled once (seed 7) and cut 70/20/10 before rendering: 560 train, 160 val, 80 test. Every image has its own random seed (`manifest.json`), so no image and no pose is shared between splits. The limit: all splits come from the same two CAD models, so test accuracy says how well the detector copes with new poses, lighting and backgrounds of the same objects. It says nothing about real photos.

**Training.** Ultralytics 8.4.191, YOLO26n initialised from the COCO weights `yolo26n.pt`, 30 epochs, image size 416, batch 16, CPU only (0.94 h). Settings in `runs/synthetic_v1/args.yaml`, curves in `runs/synthetic_v1/results.csv` and `results.png`.

**Test results** (80 held-out images, 115 objects, PyTorch weights, IoU 0.5 for mAP50):

| | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| all | 0.994 | 0.886 | 0.956 | 0.870 |
| sct013 | 1.000 | 0.905 | 0.964 | 0.884 |
| robot | 0.989 | 0.867 | 0.949 | 0.856 |

Confusion matrix: `runs/test_pytorch/confusion_matrix.png`. The validation split ended at mAP50 0.987 and mAP50-95 0.933, so the test numbers are a little lower, as expected.

**Wrong detections** (conf 0.25, IoU 0.5): 12 misses, 2 wrong-class, 4 false positives. Ten are drawn in `results/wrong_detections/` (green = ground truth, red = prediction):

- Wrong class (`wrong_01`, `wrong_02`): an SCT-013 that overlaps the robot chassis gets the class `robot`, with the box in the right place (IoU 0.73 and 0.86). Both are overlaps, and the SCT-013 is partly covered; I did not test why the class flips.
- False positives (`wrong_03` to `wrong_06`): extra `robot` or `sct013` boxes around two overlapping objects, i.e. duplicate or merged boxes in crowded overlaps.
- Misses (`wrong_07` to `wrong_10`): a small SCT-013 almost fully hidden behind the chassis (`wrong_07`), and overlapping pairs where the predicted box extends over a neighbour (IoU 0.45 to 0.49, just under the 0.5 cut-off) or an object is hidden (`wrong_09`, IoU 0).
- **A cause I found in my own data:** the renderer pastes objects one after another and takes each box from the object's full mask before later objects are pasted on top. So an object that ends up mostly or fully hidden still has a full box. Some of the "misses" are therefore label noise, not detector errors. I did not fix this in v1.

## M2 Optimize: ONNX export and accuracy check

`scripts/evaluate.py` exports `best.pt` to ONNX (`runs/synthetic_v1/weights/best.onnx`, opset 17, fixed 416x416, simplified) and scores both on the same 80 test images:

| | mAP50 | mAP50-95 | Precision | Recall | File size |
|---|---|---|---|---|---|
| PyTorch (`best.pt`) | 0.9565 | 0.8703 | 0.994 | 0.886 | 5.41 MB |
| ONNX (`best.onnx`) | 0.9577 | 0.8714 | 0.987 | 0.885 | 9.76 MB |

mAP is unchanged within 0.002 after export. ONNX is larger because it stores weights as float32 with the graph.

**Benchmark** (ONNX Runtime, CPUExecutionProvider, 2 threads, batch 1, 416x416, 20 warm-up runs then 100 timed runs on test images, preprocessing excluded):

| median | p95 | mean | FPS from median |
|---|---|---|---|
| 13.5 ms | 19.2 ms | 14.3 ms | 74 |

This is a **cloud container CPU, not a Raspberry Pi or any edge device**. Expect the real device to be slower. Not done for M2: TFLite INT8 conversion and CPU-vs-accelerator comparison.

## Reproduce

```sh
pip install torch torchvision ultralytics onnx onnxslim onnxruntime pyrender trimesh opencv-python
python scripts/render_dataset.py 800     # data/
python scripts/train.py 30               # runs/synthetic_v1/
python scripts/evaluate.py               # results/, ONNX export, benchmark
```

`weights/yolo26n.pt` is the COCO starting checkpoint.

## Not done

M1: camera and first commit from the device, live OpenCV overlay on the device, baseline speed (FPS, CPU load, temperature) on the device. M2: TFLite INT8 on the device, CPU vs accelerator table, own target and trade-off written before training. M3 to M5: everything.
