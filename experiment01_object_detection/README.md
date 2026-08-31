# Experiment 01: Object Detection and Recognition

This experiment implements a reproducible robot-vision pipeline: collect and label desktop objects, train
a YOLOv8 model, run real-time inference on a Jetson Orin NX, and expose the results through ROS 2, a web
interface, and an acceptance-test script.

[Back to the project overview](../README.md) | [Download the demo model and video](https://github.com/dc4008-rgb/robot-integration-project/releases/tag/experiment1-demo-20260831)

## 1. Objectives

- Detect `cup`, `mouse`, `keyboard`, and `bottle`.
- Achieve at least 5 FPS on the Jetson.
- Publish ROS 2 detection messages and display classes, confidence scores, and bounding boxes in real time.
- Run a repeatable 20-trial physical-object test with a target accuracy of at least 80%.

## 2. Environment

| Location | Environment |
|---|---|
| Data processing and training | Python 3.10+, Ultralytics, OpenCV, PyYAML |
| Training server | NVIDIA GPU, CUDA, PyTorch |
| Edge device | Jetson Orin NX 16 GB, JetPack 6.2 |
| Robotics middleware | ROS 2 Humble, `vision_msgs`, `sensor_msgs`, `cv_bridge` |
| Camera | UVC USB camera, MJPG 1280 x 720 at 30 FPS |

Install the basic local dependencies:

```bash
python -m pip install ultralytics opencv-python pyyaml
```

## 3. End-to-End Workflow

Run the following commands from `experiment01_object_detection/`.

### Step 1: Check the Camera and Capture Images

Compare actual MJPG and YUYV capture throughput first:

```bash
python 01_data_collection/check_camera.py --index 0
```

With a display available, press `s` to save an image and `q` to quit:

```bash
python 01_data_collection/capture_images.py \
  --source 0 --tag desktop --output-dir 02_dataset/raw
```

Use automatic capture for a headless SSH session:

```bash
python 01_data_collection/capture_images.py \
  --source 0 --tag desktop --output-dir 02_dataset/raw \
  --no-display --interval 0.5 --count 100
```

### Step 2: Pre-label, Review, and Split the Dataset

Generate YOLO-format pre-labels with COCO weights:

```bash
python 02_dataset/auto_label.py \
  --input-dir 02_dataset/raw --model yolov8m.pt
```

Review every pre-label manually. Add missing boxes, remove incorrect boxes, and adjust box boundaries.
Then create the training and validation sets:

```bash
python 02_dataset/split_dataset.py \
  --raw-dir 02_dataset/raw --val-ratio 0.2 --seed 42
```

If a separately captured validation set is available, prefer the independent directory:

```bash
python 02_dataset/split_dataset.py \
  --raw-dir 02_dataset/raw \
  --val-dir 02_dataset/independent_val
```

Optionally download all four classes from Open Images and add them only to the training set:

```bash
python -m pip install fiftyone setuptools
python 02_dataset/download_openimages.py \
  --cup 25 --mouse 25 --keyboard 25 --bottle 25

python 02_dataset/split_dataset.py \
  --raw-dir 02_dataset/raw \
  --val-dir 02_dataset/independent_val \
  --external-yolo-dir 02_dataset/external_openimages_yolo
```

### Step 3: Train YOLOv8

Train directly on a GPU workstation:

```bash
python 03_training/train_yolo.py \
  --data 02_dataset/data.yaml \
  --model yolov8n.pt --epochs 150 --batch 16 \
  --name desktop_det
```

Alternatively, synchronize the data to a training server over SSH:

```bash
export REMOTE_HOST=user@server.example.com
export REMOTE_PORT=22

cd 03_training
./sync.sh up
./sync.sh train
./sync.sh log
./sync.sh down
cd ..
```

The best checkpoint is written to `weights/best.pt` in the training output. The remote workflow downloads
it to `03_training/weights/best.pt`.

### Step 4: Deploy to the Jetson

Set the device address, check its environment, and upload the code and weights:

```bash
export JETSON_HOST=nvidia@192.168.55.1

./04_jetson_deployment/deploy_to_jetson.sh check
./04_jetson_deployment/deploy_to_jetson.sh push
./04_jetson_deployment/deploy_to_jetson.sh model 03_training/weights/best.pt
```

Start the ROS 2 detection node on the Jetson:

```bash
source /opt/ros/humble/setup.bash
cd ~/robot_det

python3 04_jetson_deployment/yolo_ros2_node.py \
  --weights 04_jetson_deployment/best.pt \
  --source 0 --conf 0.25 --iou 0.5 \
  --camera-fps 30 --no-display --web-port 8080
```

Open `http://<Jetson-IP>:8080/` in a browser to view live detections. For TensorRT inference, export the
engine on the Jetson itself and then pass `best.engine` to the node:

```bash
python3 04_jetson_deployment/export_engine.py \
  --weights 04_jetson_deployment/best.pt --half
```

### Step 5: Check ROS 2 Output and Run Acceptance Tests

```bash
ros2 topic hz /detections
ros2 topic echo /detections --once
```

Run the 20-trial physical-object acceptance test:

```bash
python3 05_acceptance_testing/eval_acceptance.py \
  --weights 04_jetson_deployment/best.engine \
  --classes cup mouse keyboard bottle \
  --conf 0.25 --iou 0.5 --trials 20
```

The script automatically saves `summary.json`, `records.csv`, and annotated error images.

## 4. System Outputs

| Output | Type | Purpose |
|---|---|---|
| `/detections` | `vision_msgs/Detection2DArray` | Classes, confidence scores, and bounding boxes |
| `/detection_image` | `sensor_msgs/Image` | Annotated camera image |
| `/` | Browser page | Live MJPEG detection stream |
| `/status.json` | JSON | FPS, object count, and timing information |
| `/snapshot.jpg` | JPEG | Current annotated frame |
| `/raw.jpg` | JPEG | Raw frame from the same time point |

The web interface and ROS 2 reuse the same inference result, so visualization does not run the model twice.

## 5. Final Configuration and Results

| Item | Configuration or Result |
|---|---|
| Input size | 640 x 640 |
| Detection thresholds | Confidence 0.25, NMS IoU 0.5 |
| Latest training set | 709 images, 1,155 bounding boxes, 44 pure-negative images |
| Frozen independent validation set | 24 images, 73 bounding boxes |
| Demonstration model metrics | mAP50 0.5236, mAP50-95 0.3797 |
| Jetson measurement | Approximately 15 FPS and approximately 15 Hz ROS 2 output |
| Demonstration video | H.264, 1280 x 720, 15 FPS, 20 seconds, 300 frames |

The demonstration model completes the four-class live pipeline on the Jetson. It still has a known risk of
confusing headphone cases with mice, so the current result demonstrates the end-to-end system rather than
claiming final acceptance accuracy. Final accuracy must come from a new 20-trial acceptance run.

## 6. Acceptance Criteria

| Requirement | Verification | Current Status |
|---|---|---|
| At least two object classes | Four-class model and multi-class live demonstration | Complete |
| At least 5 FPS on the Jetson | Live node statistics and FPS benchmark | Approximately 15 FPS; complete |
| Display class, box, and confidence | Browser view or ROS image topic | Complete |
| Publish ROS 2 detections | `/detections` message and frequency checks | Complete |
| At least 80% accuracy over 20 objects | `eval_acceptance.py` | Final retest pending |

## 7. File Management

The Git repository contains only the main workflow code and documentation. The following large artifacts
are kept out of regular source commits:

- Raw images, YOLO labels, and public-dataset caches
- Model weights, TensorRT engines, and training plots
- Per-run evaluation reports, error-analysis material, and recorded videos

The demonstration `best.pt` and `demo.mp4` are available from the GitHub Release linked at the top of this
document, keeping the source tree concise.