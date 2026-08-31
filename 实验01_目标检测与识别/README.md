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
yolo predict model=yolov8n.pt source=0 show=True classes=41,64,66,39
```

（`classes` 是 COCO 的 cup/mouse/keyboard/bottle。）

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

针对实测暴露的困难场景，应按拍摄批次隔离训练集和验证集，不要把同一次连拍随机拆到
两边。当前鼠标优化建议补采 40~60 张训练图，重点覆盖白色/粉色鼠标、远距离、侧面、
高光和多鼠标场景；另拍 20~24 张独立验证图，四类各至少 4 张且鼠标适当多留。每次
移动物体或改变光照后先稳住摄像头再保存，连续近似帧不算新的有效样本：

```bash
python3 01_数据采集/capture_images.py --tag mouse_hard_train --source 0 \
    --output-dir 02_数据集/recollection/train \
    --fixed-camera-fps --auto-exposure 0.25 --exposure 140 --gain 64 \
    --backlight-compensation 0 --gamma 400 --quality-metrics
python3 01_数据采集/capture_images.py --tag independent_val --source 0 \
    --output-dir 02_数据集/recollection/val \
    --fixed-camera-fps --auto-exposure 0.25 --exposure 140 --gain 64 \
    --backlight-compensation 0 --gamma 400 --quality-metrics
```

在 Mac 上按批次拉回补采数据，默认范围为 `val`，避免已人工筛选的训练目录被板端旧副本覆盖：

```bash
export JETSON_HOST=nvidia@192.168.55.1
./04_Jetson部署/deploy_to_jetson.sh pull-data train
./04_Jetson部署/deploy_to_jetson.sh pull-data val
```

## 步骤 2 · 标注

**先跑自动预标注**。`cup / mouse / keyboard / bottle` 四类都在 COCO 80 类里（id 41/64/66/39），
所以可以直接用预训练模型把框先画好，人工只需要检查修正：

```bash
python 02_数据集/auto_label.py --conf 0.35
python 02_数据集/auto_label.py --input-dir 02_数据集/recollection/train --conf 0.25
python 02_数据集/auto_label.py --input-dir 02_数据集/recollection/val --conf 0.25
```

然后用 [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) 或 `labelImg` 逐张核对：
**补漏检、删误检、修框**。预标注只是省力，不能代替人工核对——
没核对就训练，等于把预训练模型的错误学进去。

```bash
python -m pip install labelImg && labelImg 02_数据集/raw
```

标注格式必须是 **YOLO**，`.txt` 与图片同名同目录，类别顺序与
[02_数据集/data.yaml](02_数据集/data.yaml) 里的 `names` 完全一致。

耳机盒误报为 `mouse`、手机或电源盒误报为 `bottle` 时，需要补充困难负样本。另拍
30~50 张只包含这些干扰物、完全不含四类目标的图片，覆盖不同角度、距离、位置和背景；
不要对该目录运行自动预标注，划分脚本会为它们生成空标签：

```bash
python3 01_数据采集/capture_images.py --tag hard_negative --source 0 \
    --output-dir 02_数据集/recollection/negative_train --interval 0.8 --count 40 \
    --fixed-camera-fps --auto-exposure 0.25 --exposure 140 --gain 64 \
    --backlight-compensation 0 --gamma 400 --quality-metrics
```

采集后必须逐张确认画面中没有 `cup / mouse / keyboard / bottle`。如果画面同时包含真实
鼠标和耳机盒，应放入普通训练目录，只标真实鼠标，耳机盒不画框；不能把整张混合画面
作为空标签，否则模型会同时学着忽略真实鼠标。连续近似帧应筛掉，并另留 12~20 张未参与
训练的负样本作为误报回归集。

根目录 `实际负样本/` 中的实拍图可用下面的脚本整理。脚本会先核对源文件 SHA-256，
再输出四个相互隔离的目录：

```bash
python 02_数据集/curate_actual_negatives.py --overwrite
```

本次 9 张实拍图中，4 张耳机盒照片同时拍到了真实键盘，因此不能作为空标签：其中 3 张
进入 `recollection/actual_negatives/train_mixed/` 并只标注键盘，1 张留在
`regression_mixed/`。其余 5 张电源适配器照片不含四类目标，3 张进入 `train_pure/`，
2 张留在 `regression_pure/`。当前模型在 9 张中有 8 张产生 `conf >= 0.25` 误报，最高
置信度为 0.90。两个 `regression_*` 目录只用于新旧模型对比，**绝不能加入训练集**。

划分训练/验证集：

```bash
python 02_数据集/split_dataset.py --raw-dir 训练数据集 --val-ratio 0.2
```

完成困难样本补采后，推荐使用独立验证批次。旧本地数据与新训练批次都会进入训练集，
Open Images 仍只进入训练集；脚本会拒绝训练来源中的同名文件，以及训练集与验证集中的
完全重复图片：

```bash
python 02_数据集/split_dataset.py \
    --raw-dir 训练数据集 \
    --train-extra-dir 02_数据集/recollection/train \
    --negative-dir 02_数据集/recollection/negative_train \
    --val-dir 02_数据集/recollection/val \
    --external-yolo-dir 02_数据集/external_openimages_yolo \
    --train-hardlink
```

`--train-hardlink` 只让本地生成的训练目录复用源文件数据块，验证集仍为独立副本；rsync
上传到服务器后仍是普通文件。磁盘空间充足时可以省略该参数。

### 2.1 可选：补充 Open Images 数据

Open Images V7 已有人工核验的目标框，不需要从零手动画框，但原始标注不是本项目的
YOLO 类别编号，也不保证穷尽画面中的所有物体，仍需**筛选、转换、重映射和逐张检查**。
下载脚本会自动完成前三项，并保留每张图片中已有的全部项目目标框：

| Open Images 类别 | 本项目类别 |
|---|---|
| `Coffee cup` | `0 cup` |
| `Computer mouse` | `1 mouse` |
| `Computer keyboard` | `2 keyboard` |
| `Bottle` | `3 bottle` |

这里必须使用 `Computer mouse` 和 `Computer keyboard`，不能误用动物 `Mouse` 或
`Musical keyboard`。针对相似物误报，推荐先建立候选池，再由当前模型排序并人工审计；
不要把下载结果直接加入训练集。下面的轻量下载器按类别读取 Open Images 可视化器索引，
不依赖 FiftyOne 的本地数据库，也不会下载数 GB 的全量元数据：

```bash
python 02_数据集/download_openimages_candidates.py \
    --positive-per-class 20 \
    --mobile-phone 40 --headphones 30 --power-plugs 30 \
    --exclude-dir 02_数据集/images \
    --exclude-dir 02_数据集/external_openimages_yolo \
    --exclude-dir 02_数据集/external_openimages_cup_extra_yolo

python 02_数据集/screen_openimages_candidates.py \
    --weights 03_训练/weights/best_desktop_det_recollection_20260829.pt \
    --device mps --overwrite

python 02_数据集/curate_openimages_candidates.py --overwrite
```

`--device mps` 适用于当前 Apple Silicon Mac；没有 MPS 时可省略。下载器会排除已知含
四类目标的负候选，并按 Open Images ID 和 SHA-256 与既有数据去重。筛选器用选定模型
找出高置信误报负样本，以及漏检或低置信的正样本，同时生成联系表供逐张复核。

本次候选池共 180 张：80 张正候选、100 张负候选。模型筛选后，经十张联系表和高风险
原图复核，最终保留 15 张正样本（28 个完整目标框）、36 张训练负样本，并另留 15 张
独立负回归样本。产物位于
`02_数据集/external_openimages_bidirectional_candidates/selected/`，详细来源、许可、
哈希、模型预测和人工剔除原因记录在 `selection.json`。其中
`negative_regression/` **绝不能加入训练集**。

当前训练集已用以下命令接入筛选后的公开正负样本与实拍样本：

```bash
python 02_数据集/split_dataset.py \
    --raw-dir 训练数据集 \
    --train-extra-dir 02_数据集/recollection/train \
    --train-extra-dir 02_数据集/recollection/actual_negatives/train_mixed \
    --negative-dir 02_数据集/external_openimages_bidirectional_candidates/selected/negative_train \
    --negative-dir 02_数据集/recollection/actual_negatives/train_pure \
    --val-dir 02_数据集/recollection/val \
    --external-yolo-dir 02_数据集/external_openimages_yolo \
    --external-yolo-dir 02_数据集/external_openimages_bidirectional_candidates/selected/positive_yolo \
    --train-hardlink
```

重建结果为 292 张训练图、534 个目标框和 39 个空标签；四类框数依次为
`cup 76 / mouse 154 / keyboard 155 / bottle 149`。冻结验证集仍为 24 张图、73 个框，
清单中的 48 个图片/标签文件全部通过 SHA-256 校验。15 张公开回归图和 3 张实拍回归图
与训练集、验证集均无哈希重合。

若只需重建旧的四目标 Open Images 初始补充集，仍可使用 FiftyOne 下载器；当前 Python
3.13 环境中其数据库服务可能无法启动，因此优先使用上面的轻量候选流程：

```bash
python -m pip install fiftyone setuptools
python 02_数据集/download_openimages.py
```

输出位于 `02_数据集/external_openimages_yolo/`，图片、标签和下载缓存均已被 Git 忽略。
逐张检查四个目标类，补齐漏框并删除误框后，再将外部数据仅加入训练集；验证集保持为
本地摄像头数据，才能真实反映 Jetson 场景效果：

```bash
python 02_数据集/split_dataset.py \
    --raw-dir 训练数据集 \
    --external-yolo-dir 02_数据集/external_openimages_yolo \
    --val-ratio 0.2
```

Open Images 标注采用 CC BY 4.0；图片列为 CC BY 2.0，但再分发前仍应核对每张图片的
元数据。脚本会在导出目录保存 `source.json` 记录来源、许可证、映射和框数量。

## 步骤 3 · 远程训练

`sync.sh` 默认连接当前训练实例，并将项目放在高速数据盘
`/root/autodl-tmp/robot_det`。更换实例时可用 `REMOTE_HOST`、`REMOTE_PORT` 和
`REMOTE_PYTHON` 覆盖默认值，不需要修改训练代码：

本轮训练实例为 `root@connect.westd.seetacloud.com:18507`，已验证 RTX 5090、CUDA 和
Ultralytics 8.4.128 可用。云实例重启或更换后端口可能变化，届时优先用环境变量覆盖，
不要在命令历史或脚本中保存密码。

```bash
cd 03_训练
./sync.sh up       # 上传数据集
./sync.sh train    # 后台启动训练
./sync.sh log      # 看进度
./sync.sh down     # 训练完拉回 best.pt
```

服务器上首次需要安装 `ultralytics`。若默认镜像没有该包，可运行
`python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ultralytics==8.4.128`。
数据集小于 1000 张时 `yolov8n` 足够，
若 mAP50 低于 0.85 再换 `yolov8s`。

### 3.1 双向增强训练结果

在冻结验证集上完成了两轮对照。直接从 `yolov8n.pt` 重训的模型降低了误报，但召回
下降；从此前实拍困难样本模型继续训练的增量模型更均衡，因此将其作为下一次 Jetson
实测候选，同时保留旧模型作为回退，不覆盖原权重。

| 模型 | mAP50 | mAP50-95 | `conf=0.25` 验证 TP/FP | 回归集额外误报 |
|---|---:|---:|---:|---:|
| 旧实拍困难样本模型 | 0.4401 | 0.2741 | 31 / 28 | 25 框，14 图 |
| 双向增强，从 COCO 重训 | 0.4190 | 0.3026 | 27 / 14 | 6 框，4 图 |
| 双向增强，旧模型增量训练 | 0.4374 | 0.3283 | 37 / 13 | **0 框，0 图** |

增量模型位于
`03_训练/weights/best_desktop_det_bidirectional_finetune_20260829.pt`，SHA-256 为
`1c734e9738a11353f9cd7467cdf5694892408578bb0312f291c5fc2e31f33ed2`。固定阈值统计采用
CPU 单图推理，避免 MPS 批处理的 NMS 超时；18 张隔离回归图中，17 张纯负图无预测，
混合耳机盒图仅正确检出真实键盘。详细口径和逐模型结果见
`03_训练/runs_independent_eval/comparison_bidirectional_20260829.json`。

该模型仍不是最终验收模型：冻结集的 cup/bottle AP50 分别为 0.1740/0.2721，且 9 张
实拍样本中仍有 1 张在画面边缘产生 `cup 0.610` 误报。下一步应在 Jetson 相同摄像头和
曝光参数下，与旧模型做并排实测，再从零执行 20 次正式验收。

### 3.2 四类均衡增样与补测复训

2026-08-31 又进行了两阶段实验。第一阶段人工复核并加入四类各 100 张 Open Images
图片，训练集扩展到 692 图、1143 框和 39 个空标签；第二阶段再加入 12 张灰色保温瓶
正样本及 5 张耳机盒空标签样本，形成 709 图、1155 框和 44 个空标签的最新训练集。
冻结验证集始终保持 24 图、73 框，未参与训练。

| 实验 | mAP50 | mAP50-95 | 独立回归结果 | 决策 |
|---|---:|---:|---|---|
| 四类各增 100 张，旧候选增量训练 | 0.5993 | 0.4430 | 纯负误报 11/17，耳机误报 10/20，灰瓶 bottle 0.0703 | 淘汰 |
| 四类各增 100 张，从实拍模型起训 | 0.5582 | 0.3719 | 16 个检查点无一同时通过四道门 | 淘汰 |
| 加入补测灰瓶与耳机盒 | 0.5236 | 0.3797 | 纯负误报 8/17，耳机误报 7/20，灰瓶 bottle 0.4578 | 仅作实验，不部署 |

第三阶段最佳权重为
`best_desktop_det_balanced100_retest_finetune_20260831.pt`，SHA-256 为
`1e2e90c72c138aa2dc5ff7a6615bdbf5be5009fe86ae408c0c3ee43558cf1455`。通过确定性
重放覆盖 57 个训练轮次，并连同 baseline、best、last 共检查 60 个模型文件；固定
`conf=0.25` 和统一可调阈值下，均没有模型同时通过纯负不误报、耳机不报 mouse、混合图
检出真实键盘、灰瓶检出 bottle 四道门。因此本轮没有覆盖此前候选，不能只凭 mAP 提升
宣布模型通过。详细摘要见
`03_训练/runs_independent_eval/comparison_balanced100_retest_20260831.json`。

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
不会额外运行第二次推理；`/status.json` 返回当前 FPS、目标数和取帧/推理耗时；开启
`--quality-metrics` 后还会返回高光剪切率和清晰度。`/snapshot.jpg` 用于保存带框证据，
`/raw.jpg` 返回同一实时流的无框原始画面；`/status.json` 还包含类别、置信度和坐标，
便于对不同模型做同帧比较。
使用 COCO 预训练模型实测网页显示为 15.0 FPS，同时
`/detections` 保持 15.01 Hz。最终验收时把权重替换为自训练的 `best.engine`。

UVC 摄像头采集目标默认是 30 FPS，与检测节点的 15 FPS 定时频率相互独立。强光场景先
查看 `clipped_percent`，运动模糊则比较 `sharpness`；低 FPS 本身不会造成过曝。需要锁定
曝光时，先用 `v4l2-ctl -d /dev/video0 --list-ctrls-menus` 查询摄像头范围，再传入板上实测值：

```bash
python3 04_Jetson部署/yolo_ros2_node.py --weights best.pt --source 0 \
    --conf 0.25 --iou 0.5 --camera-fps 30 --fixed-camera-fps \
    --auto-exposure 0.25 --exposure 140 --gain 64 --backlight-compensation 0 --gamma 400 \
    --quality-metrics --no-display --web-port 8080
```

上述数值来自当前 USB 摄像头在桌面强光场景下的实测起点：自动曝光时高光剪切约 27%，
手动曝光 140 时约 2%~3%，同时维持 15 FPS。更换摄像头或现场光照后必须重新测量，
不能照搬该数值。

2026-08-31 低照实验室展示时，现场改用曝光 320、亮度 12、增益 64、gamma 400，并保持
约 15 FPS。该组参数仅用于同一摄像头和当时光照，强光环境仍应从曝光 140 附近重新调节。

现场单透明瓶曾在默认 NMS IoU 0.7 下产生两个高度重叠的 `bottle` 框；`--iou 0.5` 可在
保留主框的同时抑制重复框，因此当前实时测试和正式验收统一使用 0.5。

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
    --classes cup mouse keyboard bottle --conf 0.25 --iou 0.5 --trials 20
```

脚本会先测 FPS，再逐个物体交互式测试，最后输出：

- `results/summary.json` — 正确率、FPS、是否达标、每类统计、混淆情况
- `results/records.csv` — 20 条逐次记录
- `results/errors/` — 错误案例图（图上标了真值和预测）

2026-08-30 的 Jetson 现场诊断发现，双向增强模型虽然解决了手机、电源适配器的大部分
误报，但耳机和耳机盒仍持续误报为 `mouse`，灰色保温瓶在 `conf=0.25` 下漏检。8 月 31 日
补测模型改善了灰瓶，并完成两段 20 秒带框展示视频；但它仍未通过上述独立回归门，因此
视频只证明实时链路和四类展示可用，不代表正式 20 次验收通过，也没有覆盖旧候选模型。

---

## 验收指标对照

| 要求 | 怎么满足 | 证据 | 当前状态 |
|---|---|---|---|
| ≥2 类物体 | 训练 4 类 | `data.yaml` + 多类同屏截图 | **已验证** |
| 20 个物体正确率 ≥80% | 逐物体测试 | `results/summary.json` | 预测试发现域差异，优化后从零正式验收 |
| Jetson ≥5 FPS | 自训练 `.pt` 通常实测 15 FPS | `summary.json` 的 `fps` | **预验证达标，待正式归档** |
| 实时显示类别/框/置信度 | 检测节点网页或 OpenCV 窗口 | 两段 20 秒、300 帧结果视频 | **已验证** |
| ROS2 发布 | `/detections` (Detection2DArray) | `ros2 topic echo` 录屏 | **已验证 14.95Hz** |
| 保存结果和错误案例 | 验收脚本自动保存 | `results/` 目录 | 预测试证据已保存，正式结果待重测 |

---

## 常见问题

**FPS 上不去** → 先跑 `check_camera.py` 分清是**摄像头**还是**模型**慢：
若 MJPG 下采集帧率本身就低，优化模型没用，要降分辨率或换 USB 3.0 口；
实时节点可进一步查看 `capture_ms` 与 `inference_ms`。若取帧快而推理慢，才去确认用的是
`.engine` 而非 `.pt`、开 `jetson_clocks`、把 `--imgsz` 降到 416。

**强光下漏检** → 查看 `clipped_percent`，先降低并锁定曝光/增益，避免目标纹理被剪成纯白；
再补采同一摄像头下的高光、远距离和不同颜色实物。不要把过曝简单归因于 FPS，也不要只
增加同一角度的连续帧。

**某类总是识别错** → 看 `results/errors/` 里的错误案例，通常是该类数据太少或场景单一，
补采那个类在出错场景下的图，重新训练。

**摄像头打不开** → `ls /dev/video*` 确认编号（USB 摄像头通常 `--source 0`）；
别插在 HUB / 扩展坞上，带宽不够，直插板子的 USB 口；CSI 相机才用 `--source csi`。
