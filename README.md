# Edge AI C7: detecting two CAD parts with YOLO

Individual assessment, Edge AI & Embedded Vision Systems. Viswes Vijayaragavan Sivakumar, 24BCS321.

A YOLO26n detector for two parts I designed in Fusion 360, with ONNX export, an accuracy check after export and a CPU benchmark.

## Scope

| Done in this repository | Not done yet |
|---|---|
| M1: dataset, 70/20/10 split, training, test metrics, wrong-detection analysis | M1: camera/device commit, live overlay, speed on the device |
| M2: ONNX export, mAP before/after, CPU benchmark | M2: TFLite INT8 on the device, CPU vs accelerator table |
| | M3 to M5 |

## Objects

| Class | CAD source (`cad/`) | Part |
|---|---|---|
| `0 sct013` | `SCT-013-000_current_transformer.f3d`, `.stl` | IDRP A6 current transformer |
| `1 robot` | `my_robot.f3d`, `robot_base_link.stl` | Chassis of my Fusion 360 differential-drive robot |

## M1 Detect

**Dataset v1** (`data/`, YOLO format, 800 images of 640x480). `scripts/render_dataset.py` renders each STL with random orientation, distance, colour, lighting and roughness, then places it on a random background (flat, gradient, noise, wood-like, grid) with blur, distractor shapes and image noise. Each image holds 0 to 3 objects, and about 6% are empty backgrounds. `results/dataset_preview.jpg` shows 24 samples with their boxes.

**Split 70/20/10.** Image indices are shuffled once (seed 7) and cut before rendering: 560 train, 160 val, 80 test. Every image has its own random seed (`data/manifest.json`), so no image or pose appears in two splits. All splits come from the same two CAD models.

**Training.** Ultralytics 8.4.191, YOLO26n from the COCO checkpoint `weights/yolo26n.pt`, 30 epochs, image size 416, batch 16, CPU (0.94 h). Settings: `runs/synthetic_v1/args.yaml`. Curves: `runs/synthetic_v1/results.csv`, `results.png`.

**Test results** (80 held-out images, 115 objects, IoU 0.5):

| | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| all | 0.994 | 0.886 | 0.956 | 0.870 |
| sct013 | 1.000 | 0.905 | 0.964 | 0.884 |
| robot | 0.989 | 0.867 | 0.949 | 0.856 |

Confusion matrix: `runs/test_pytorch/confusion_matrix.png`. Validation mAP50 was 0.987 and mAP50-95 0.933.

**Wrong detections** (conf 0.25, IoU 0.5): 12 misses, 2 wrong-class, 4 false positives. Ten are drawn in `results/wrong_detections/` (green = ground truth, red = prediction).

- `wrong_01`, `wrong_02`, wrong class: an SCT-013 overlapping the chassis is called `robot`, with the box in the right place (IoU 0.73, 0.86). I did not test why.
- `wrong_03` to `wrong_06`, false positives: extra boxes around two overlapping objects.
- `wrong_07` to `wrong_10`, misses: a small SCT-013 almost hidden behind the chassis (`wrong_07`), and overlapping pairs where the box extends over a neighbour (IoU 0.45 to 0.49, just under the cut-off) or an object is hidden (`wrong_09`).
- Known label issue: objects are placed one after another and each box is taken from the full object before later ones cover it, so a mostly hidden object still has a full box. Some misses are therefore label noise.

## M2 Optimize

`scripts/evaluate.py` exports `best.pt` to ONNX (`runs/synthetic_v1/weights/best.onnx`, opset 17, fixed 416x416, simplified) and scores both on the same 80 test images:

| | mAP50 | mAP50-95 | Precision | Recall | Size |
|---|---|---|---|---|---|
| PyTorch | 0.9565 | 0.8703 | 0.994 | 0.886 | 5.41 MB |
| ONNX | 0.9577 | 0.8714 | 0.987 | 0.885 | 9.76 MB |

mAP is unchanged within 0.002 after export.

**Benchmark.** ONNX Runtime, CPUExecutionProvider, 2 threads, batch 1, 416x416, 20 warm-up runs then 100 timed runs, preprocessing excluded:

| median | p95 | mean | FPS (from median) |
|---|---|---|---|
| 13.5 ms | 19.2 ms | 14.3 ms | 74 |

Measured on a PC-class CPU, not on the edge device. Raw numbers: `results/results.json`.

## Reproduce

```sh
pip install torch torchvision ultralytics onnx onnxslim onnxruntime pyrender trimesh opencv-python
python scripts/render_dataset.py 800     # data/
python scripts/train.py 30               # runs/synthetic_v1/
python scripts/evaluate.py               # results/, ONNX export, benchmark
```

## Layout

```
cad/        Fusion 360 files and STL exports
data/       images, labels, data.yaml, manifest.json
scripts/    render_dataset.py, preview.py, train.py, evaluate.py
runs/       training and test outputs, weights (best.pt, best.onnx)
results/    results.json, dataset preview, wrong detections
weights/    COCO starting checkpoint
```
