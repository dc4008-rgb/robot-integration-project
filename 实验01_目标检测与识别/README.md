# 实验01：目标检测与识别

- [实验一阶段记录](实验一阶段记录.md)：已完成内容、实测结果、待办和提交物
- [返回课程作业目录](../README.md)

以下命令均默认从本实验目录执行。

Jetson Orin NX 16GB (JetPack 6.2 / L4T R36.4.7 / Ubuntu 22.04.5 / ROS2 Humble)
+ USB 摄像头 + YOLOv8 + AutoDL 远程训练。

> **板子环境已实测就绪，无需安装任何依赖**：torch 2.5.0a0+nv24.08 (CUDA 可用)、
> ultralytics 8.4.52、TensorRT 10.7.0、rclpy、cv_bridge、vision_msgs、ROS2 Humble 全部齐全。
> 摄像头 `/dev/video0` 支持 **MJPG 1280x720@30fps**。

## 已验证的链路基线（2026-08-24 实测）

用 COCO 预训练 `yolov8n.pt` 在板子上跑通了全链路，作为徏后对照基线：

| 环节 | 实测值 |
|---|---|
| 取帧耗时 | 33.8 ms/帧（摄像头跑满 30 FPS，MJPG 生效）|
| 推理耗时 | 32.7 ms/帧（`.pt` 未转 TensorRT）|
| **端到端 FPS** | **15.0**（取帧与推理串行，33.8+32.7≈66.5ms）|
| ROS2 `/detections` 发布频率 | **14.95 Hz**（稳定，std dev 0.003s）|
| 消息格式 | `vision_msgs/Detection2DArray`，`class_id` + `score` + `bbox` 正常 |
| 无窗口采集 | 可用（SSH 下必须加 `--no-display`）|

**结论**：验收要求 ≥5 FPS，当前 15 FPS 有 3 倍余量，**速度不是风险点**，
甚至不一定需要转 TensorRT。重点应放在**数据质量和正确率**上。

> 预训练模型在实际画面里把东西误识别成 `toilet`、0.89 置信度，且没能把保温杯
> 识别为 `cup`——这正是必须自己采集数据训练的原因。

## 目录

| 目录 | 内容 |
|---|---|
| `01_数据采集/` | 摄像头采图脚本 |
| `02_数据集/` | 原图、标注、划分脚本、`data.yaml` |
| `03_训练/` | 服务器训练脚本、数据/权重同步脚本 |
| `04_Jetson部署/` | TensorRT 导出、ROS2 检测节点 |
| `05_验收测试/` | 正确率 / FPS 测试、错误案例 |

---

## 步骤 0 · 本地环境（已验证）

本机已装好 `ultralytics 8.4.127` + `torch 2.11.0`（MPS 可用）。

> ⚠️ 本机 `pip` 命令指向的是系统 Python 3.8，和 `python`（miniconda 3.13）**不是同一个环境**。
> 装包一律用 `python -m pip install ...`，直接用 `pip` 会装错地方。

先拿 COCO 预训练模型对着自己桌面跑一下，确认摄像头正常、顺便看看类别选得好不好：

```bash
yolo predict model=yolov8n.pt source=0 show=True classes=41,64,66,73
```

（`classes` 是 COCO 的 cup/mouse/keyboard/book。）

> 报 `Failed to read images from 0` 是 **macOS 摄像头权限**问题，不是代码错：
> 系统设置 → 隐私与安全性 → 摄像头 → 允许 **Visual Studio Code**（或改用系统自带的
> 终端.app 跑，首次会弹授权），授权后**重启 VS Code**。
> 这一步只影响本地预览，Jetson 上不存在这个限制。

---

## 步骤 1 · 采集数据

### 1.1 先给摄像头体检（USB 外接摄像头必做）

```bash
python3 01_数据采集/check_camera.py
```

**为什么必须先做**：UVC（USB）摄像头默认输出 **YUYV 未压缩格式**，720p 下带宽跑满，
很多摄像头只能给 **5~10 FPS**。这是摄像头的输出上限，**跟模型快不快无关**，
模型再快也救不回来，会直接卡死「≥5 FPS」这个验收指标。
本项目所有脚本已统一强制切成 **MJPG**，体检脚本会把两种格式的实测帧率都打出来对比。

确认设备号：

```bash
ls /dev/video*        # USB 摄像头通常是 video0
```

### 1.2 正式采集

在 Jetson 上（用最终部署的那个摄像头采集，域差异最小）：

```bash
python3 01_数据采集/capture_images.py --tag cup --interval 0.5 --source 0
```

采集要点：每类 **150~300 张**，覆盖不同角度、距离、光照、背景、遮挡，并且要有
**多个物体同时出现**的图，否则实机多物体场景会漏检。

## 步骤 2 · 标注

**先跑自动预标注**。`cup / mouse / keyboard / book` 四类都在 COCO 80 类里（id 41/64/66/73），
所以可以直接用预训练模型把框先画好，人工只需要检查修正：

```bash
python 02_数据集/auto_label.py --conf 0.35
```

然后用 [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) 或 `labelImg` 逐张核对：
**补漏检、删误检、修框**。预标注只是省力，不能代替人工核对——
没核对就训练，等于把预训练模型的错误学进去。

```bash
python -m pip install labelImg && labelImg 02_数据集/raw
```

标注格式必须是 **YOLO**，`.txt` 与图片同名同目录，类别顺序与
[02_数据集/data.yaml](02_数据集/data.yaml) 里的 `names` 完全一致。

划分训练/验证集：

```bash
python 02_数据集/split_dataset.py --val-ratio 0.2
```

## 步骤 3 · 远程训练

先在 `~/.ssh/config` 中配好训练服务器别名（换实例时只需修改 `Port`）。

```bash
cd 03_训练
./sync.sh up       # 上传数据集
./sync.sh train    # 后台启动训练
./sync.sh log      # 看进度
./sync.sh down     # 训练完拉回 best.pt
```

服务器上首次需要 `pip install ultralytics`。数据集小于 1000 张时 `yolov8n` 足够，
若 mAP50 低于 0.85 再换 `yolov8s`。

## 步骤 4 · Jetson 部署

### 4.1 连上板子

**Mac 上不需要装任何 Jetson 环境**（JetPack 是刷进板子的系统，SDK Manager 只支持 x86 Ubuntu，
也没有模拟器）。工作方式是 Mac 当编辑器、Jetson 当运行环境。

#### 方式 A：USB 直连（本项目实际使用，不需要显示器/键鼠/路由器）

Jetson 的**烧录口**用 USB 线接到 Mac 后，会同时模拟出网卡、串口和 U 盘：

| 通道 | 地址 | 用途 |
|---|---|---|
| USB 网卡 | 板子 `192.168.55.1`，Mac `192.168.55.100` | SSH / SFTP，日常就用这个 |
| 串口 | `/dev/cu.usbmodem*` | 网络挂了时的后路，能看启动日志 |
| U 盘 | `/Volumes/L4T-README` | NVIDIA 自带说明文档 |

```bash
ssh nvidia@192.168.55.1          # 默认用户名 nvidia（串口登录提示是 nvidia-desktop）
```

连不上时先自查：

```bash
ifconfig | grep 192.168.55       # Mac 应拿到 192.168.55.100
ping -c 3 192.168.55.1           # 板子是否活着
ls /dev/cu.usbmodem*             # 串口是否存在
screen /dev/cu.usbmodem* 115200  # 走串口登录（退出: Ctrl+A 然后 K, y）
```

> USB 直连**没有外网**。板子要 `apt install` / `pip install` 时，得另外插网线或连 WiFi
> （参考 `/Volumes/L4T-README/README-wifi.txt`）。

#### 方式 B：局域网

在 Jetson 上查 IP（`hostname -I`），然后 `ssh 用户名@那个IP`。

#### 配置本项目的部署脚本

```bash
export JETSON_HOST=nvidia@192.168.55.1

cd 04_Jetson部署
./deploy_to_jetson.sh check  # 体检：JetPack 版本、依赖、摄像头、ROS2
./deploy_to_jetson.sh push   # 同步代码
```

> 更省事的做法：装 VS Code 的 **Remote-SSH** 扩展直接连板子，文件存在 Jetson 上，
> 终端也是 Jetson 的终端，不用来回同步。

### 4.2 装依赖并运行

```bash
# 依赖（JetPack 6 自带 CUDA/TensorRT，PyTorch 要用 NVIDIA 官方 Jetson wheel，
# 直接 pip install torch 装到的是 CPU 版，GPU 用不上）
pip3 install ultralytics
sudo apt install ros-humble-vision-msgs ros-humble-cv-bridge

# 导出 TensorRT 引擎（必须在 Jetson 本机跑，引擎不能跨设备复用）
python3 04_Jetson部署/export_engine.py --weights best.pt --half

# 跑起来
source /opt/ros/humble/setup.bash
python3 04_Jetson部署/yolo_ros2_node.py --weights best.engine --source 0 \
    --save-video demo.mp4
```

没有显示器时，通过浏览器实时查看类别、检测框、置信度和端到端 FPS：

```bash
source /opt/ros/humble/setup.bash
python3 04_Jetson部署/yolo_ros2_node.py \
    --weights best.engine --source 0 --no-display --web-port 8080 \
    --save-video demo.mp4
```

Mac 打开 **<http://192.168.55.1:8080/>**。页面和 ROS2 共用同一份检测结果，
不会额外运行第二次推理；`/status.json` 返回当前 FPS 和目标数，`/snapshot.jpg`
用于保存单帧证据。使用 COCO 预训练模型实测网页显示为 15.0 FPS，同时
`/detections` 保持 15.01 Hz。最终验收时把权重替换为自训练的 `best.engine`。

> 网页服务没有登录认证，只应在 USB 直连或受信任局域网中使用。

权重和结果的传输：

```bash
./deploy_to_jetson.sh model 03_训练/weights/best.pt   # 推权重上去
./deploy_to_jetson.sh pull                            # 把测试结果和视频拉回来
```

验证 ROS2 输出：

```bash
ros2 topic echo /detections          # 类别、置信度、检测框
ros2 topic hz /detections            # 实测发布频率
rviz2                                # 订阅 /detection_image 看画面
```

Jetson 跑之前先开最高性能模式，否则 FPS 会明显偏低：

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks
```

## 步骤 5 · 验收测试

```bash
python3 05_验收测试/eval_acceptance.py --weights best.engine \
    --classes cup mouse keyboard book --trials 20
```

脚本会先测 FPS，再逐个物体交互式测试，最后输出：

- `results/summary.json` — 正确率、FPS、是否达标、每类统计、混淆情况
- `results/records.csv` — 20 条逐次记录
- `results/errors/` — 错误案例图（图上标了真值和预测）

---

## 验收指标对照

| 要求 | 怎么满足 | 证据 | 当前状态 |
|---|---|---|---|
| ≥2 类物体 | 训练 4 类 | `data.yaml` + 验证集 mAP | 待采集数据 |
| 20 个物体正确率 ≥80% | 逐物体测试 | `results/summary.json` | 待训练 |
| Jetson ≥5 FPS | 实测 15 FPS | `summary.json` 的 `fps` | **已达标** |
| 实时显示类别/框/置信度 | 检测节点 OpenCV 窗口 | 结果视频 `demo.mp4` | 代码就绪 |
| ROS2 发布 | `/detections` (Detection2DArray) | `ros2 topic echo` 录屏 | **已验证 14.95Hz** |
| 保存结果和错误案例 | 验收脚本自动保存 | `results/` 目录 | 代码就绪 |

---

## 常见问题

**FPS 上不去** → 先跑 `check_camera.py` 分清是**摄像头**还是**模型**慢：
若 MJPG 下采集帧率本身就低，优化模型没用，要降分辨率或换 USB 3.0 口；
若采集快而推理慢，才去确认用的是 `.engine` 而非 `.pt`、开 `jetson_clocks`、把 `--imgsz` 降到 416。

**某类总是识别错** → 看 `results/errors/` 里的错误案例，通常是该类数据太少或场景单一，
补采那个类在出错场景下的图，重新训练。

**摄像头打不开** → `ls /dev/video*` 确认编号（USB 摄像头通常 `--source 0`）；
别插在 HUB / 扩展坞上，带宽不够，直插板子的 USB 口；CSI 相机才用 `--source csi`。
