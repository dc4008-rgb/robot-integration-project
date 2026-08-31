# 实验01：目标检测与识别

本实验完成一条可复现的机器人视觉链路：采集与标注桌面物体，训练 YOLOv8 模型，在
Jetson Orin NX 上实时推理，并通过 ROS 2、浏览器和验收脚本输出结果。

[返回项目首页](../README.md) ·
[下载展示模型与视频](https://github.com/dc4008-rgb/robot-integration-project/releases/tag/experiment1-demo-20260831)

## 1. 实验目标

- 识别 `cup`、`mouse`、`keyboard`、`bottle` 四类物体。
- 在 Jetson 上达到至少 5 FPS。
- 发布 ROS 2 检测结果，并实时显示类别、置信度和边界框。
- 用统一脚本完成 20 次实物测试，目标正确率不低于 80%。

## 2. 运行环境

| 位置 | 环境 |
|---|---|
| 数据处理与训练 | Python 3.10+、Ultralytics、OpenCV、PyYAML |
| 训练服务器 | NVIDIA GPU、CUDA、PyTorch |
| 边缘设备 | Jetson Orin NX 16GB、JetPack 6.2 |
| 机器人中间件 | ROS 2 Humble、`vision_msgs`、`sensor_msgs`、`cv_bridge` |
| 摄像头 | UVC USB 摄像头，MJPG 1280×720@30 FPS |

本地安装基础依赖：

```bash
python -m pip install ultralytics opencv-python pyyaml
```

## 3. 一次完整运行

以下命令均从 `实验01_目标检测与识别/` 目录执行。

### 第一步：检查摄像头并采集

先比较 MJPG 与 YUYV 的实际采集速度：

```bash
python 01_数据采集/check_camera.py --index 0
```

有显示器时按 `s` 保存图片、按 `q` 退出：

```bash
python 01_数据采集/capture_images.py \
  --source 0 --tag desktop --output-dir 02_数据集/raw
```

通过 SSH 无窗口采集时使用自动连拍：

```bash
python 01_数据采集/capture_images.py \
  --source 0 --tag desktop --output-dir 02_数据集/raw \
  --no-display --interval 0.5 --count 100
```

### 第二步：预标注、人工复核并划分数据

使用 COCO 权重生成 YOLO 格式预标注：

```bash
python 02_数据集/auto_label.py \
  --input-dir 02_数据集/raw --model yolov8m.pt
```

预标注后必须逐图人工检查，补漏框、删误框并调整边界框。确认后生成训练集和验证集：

```bash
python 02_数据集/split_dataset.py \
  --raw-dir 02_数据集/raw --val-ratio 0.2 --seed 42
```

如果已经单独拍摄验证集，优先使用独立验证目录：

```bash
python 02_数据集/split_dataset.py \
  --raw-dir 02_数据集/raw \
  --val-dir 02_数据集/independent_val
```

可选：下载 Open Images 四类公开数据，并且只加入训练集：

```bash
python -m pip install fiftyone setuptools
python 02_数据集/download_openimages.py \
  --cup 25 --mouse 25 --keyboard 25 --bottle 25

python 02_数据集/split_dataset.py \
  --raw-dir 02_数据集/raw \
  --val-dir 02_数据集/independent_val \
  --external-yolo-dir 02_数据集/external_openimages_yolo
```

### 第三步：训练 YOLOv8

在 GPU 工作站直接训练：

```bash
python 03_训练/train_yolo.py \
  --data 02_数据集/data.yaml \
  --model yolov8n.pt --epochs 150 --batch 16 \
  --name desktop_det
```

也可以用 SSH 将数据上传到训练服务器：

```bash
export REMOTE_HOST=user@server.example.com
export REMOTE_PORT=22

cd 03_训练
./sync.sh up
./sync.sh train
./sync.sh log
./sync.sh down
cd ..
```

最佳权重位于训练输出的 `weights/best.pt`。远程流程会将其下载到
`03_训练/weights/best.pt`。

### 第四步：部署到 Jetson

设置板子地址，检查环境并同步代码和权重：

```bash
export JETSON_HOST=nvidia@192.168.55.1

./04_Jetson部署/deploy_to_jetson.sh check
./04_Jetson部署/deploy_to_jetson.sh push
./04_Jetson部署/deploy_to_jetson.sh model 03_训练/weights/best.pt
```

在 Jetson 上启动 ROS 2 检测节点：

```bash
source /opt/ros/humble/setup.bash
cd ~/robot_det

python3 04_Jetson部署/yolo_ros2_node.py \
  --weights 04_Jetson部署/best.pt \
  --source 0 --conf 0.25 --iou 0.5 \
  --camera-fps 30 --no-display --web-port 8080
```

浏览器打开 `http://<Jetson-IP>:8080/` 即可查看实时检测画面。需要 TensorRT 时，先在
Jetson 本机导出引擎，再将节点的权重改为 `best.engine`：

```bash
python3 04_Jetson部署/export_engine.py \
  --weights 04_Jetson部署/best.pt --half
```

### 第五步：检查 ROS 2 输出并验收

```bash
ros2 topic hz /detections
ros2 topic echo /detections --once
```

执行 20 次实物验收：

```bash
python3 05_验收测试/eval_acceptance.py \
  --weights 04_Jetson部署/best.engine \
  --classes cup mouse keyboard bottle \
  --conf 0.25 --iou 0.5 --trials 20
```

脚本会自动保存 `summary.json`、`records.csv` 和错误案例图片。

## 4. 系统输出

| 输出 | 类型 | 作用 |
|---|---|---|
| `/detections` | `vision_msgs/Detection2DArray` | 类别、置信度和边界框 |
| `/detection_image` | `sensor_msgs/Image` | 带检测框图像 |
| `/` | 浏览器页面 | MJPEG 实时检测画面 |
| `/status.json` | JSON | FPS、目标数和耗时 |
| `/snapshot.jpg` | JPEG | 当前带框画面 |
| `/raw.jpg` | JPEG | 同一时刻的原始画面 |

网页与 ROS 2 复用同一次模型推理结果，不会为显示功能重复运行模型。

## 5. 最终配置与结果

| 项目 | 配置或结果 |
|---|---|
| 输入尺寸 | 640×640 |
| 检测阈值 | confidence 0.25，NMS IoU 0.5 |
| 最新训练集 | 709 图、1155 框、44 张纯负样本 |
| 冻结独立验证集 | 24 图、73 框 |
| 展示模型指标 | mAP50 0.5236，mAP50-95 0.3797 |
| Jetson 实测 | 约 15 FPS，ROS 2 约 15 Hz |
| 展示视频 | H.264、1280×720、15 FPS、20 秒、300 帧 |

展示模型已经完成 Jetson 四类实时演示。它仍存在耳机盒相似物误报风险，因此当前结果用于
展示端到端系统，最终正确率需以重新执行的 20 次正式验收为准。

## 6. 验收标准

| 要求 | 验证方式 | 当前状态 |
|---|---|---|
| 至少 2 类物体 | 四类模型与多类同屏演示 | 已完成 |
| Jetson ≥5 FPS | 节点实时统计与 FPS 基准 | 约 15 FPS，已完成 |
| 显示类别、框、置信度 | 浏览器或 ROS 图像话题 | 已完成 |
| 发布 ROS 2 检测结果 | `/detections` 频率与消息检查 | 已完成 |
| 20 个物体正确率 ≥80% | `eval_acceptance.py` | 最终复测待完成 |

## 7. 文件管理

Git 仓库只保存主流程代码和说明。以下大文件不直接进入源码提交：

- 原始图片、YOLO 标签和公开数据缓存
- 模型权重、TensorRT 引擎和训练曲线
- 逐轮筛选报告、错误分析素材和录制视频

用于展示的 `best.pt` 和 `demo.mp4` 放在 GitHub Release 中，仓库目录因此保持简洁。