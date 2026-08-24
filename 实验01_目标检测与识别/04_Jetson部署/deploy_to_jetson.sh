#!/usr/bin/env bash
# 把项目同步到 Jetson 并检查板子上的环境。
#
# 先设好板子地址（Jetson 上跑 `hostname -I` 查 IP）:
#   export JETSON_HOST=jetson@192.168.1.100
#
# 用法:
#   ./deploy_to_jetson.sh push      同步代码到板子（不含数据集和权重）
#   ./deploy_to_jetson.sh model best.pt   单独推送训练好的权重
#   ./deploy_to_jetson.sh check     检查板子上的环境是否齐全
#   ./deploy_to_jetson.sh pull      把板子上的测试结果和视频拉回来
set -euo pipefail

HOST="${JETSON_HOST:-}"
REMOTE_DIR="${JETSON_DIR:-~/robot_det}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "$HOST" ]]; then
  echo "请先设置板子地址，例如:"
  echo "  export JETSON_HOST=jetson@192.168.1.100"
  exit 1
fi

case "${1:-}" in
  push)
    rsync -avz --progress \
      --exclude '02_数据集/raw/' --exclude '02_数据集/images/' \
      --exclude '02_数据集/labels/' --exclude '03_训练/runs_*' \
      --exclude '__pycache__/' --exclude '.DS_Store' --exclude '.git/' \
      --exclude '*.pt' --exclude '*.engine' \
      "$LOCAL_ROOT/" "$HOST:$REMOTE_DIR/"
    echo "代码已同步到 $HOST:$REMOTE_DIR"
    ;;
  model)
    weights="${2:-$LOCAL_ROOT/03_训练/weights/best.pt}"
    [[ -f "$weights" ]] || { echo "找不到权重: $weights"; exit 1; }
    rsync -avz --progress "$weights" "$HOST:$REMOTE_DIR/04_Jetson部署/"
    echo "权重已推送。接下来在板子上导出 TensorRT 引擎:"
    echo "  ssh $HOST 'cd $REMOTE_DIR/04_Jetson部署 && python3 export_engine.py --weights best.pt --half'"
    ;;
  check)
    ssh "$HOST" 'bash -s' <<'EOF'
echo "===== 设备信息 ====="
cat /etc/nv_tegra_release 2>/dev/null | head -1 || echo "(读不到 JetPack 版本)"
echo "型号: $(cat /proc/device-tree/model 2>/dev/null | tr -d '\0')"
echo "功耗模式: $(sudo -n nvpmodel -q 2>/dev/null | tail -1 || echo '需要手动查 nvpmodel -q')"

echo
echo "===== Python 依赖 ====="
python3 -c "
import importlib.util as u
for m in ['ultralytics','cv2','torch','torchvision','tensorrt','rclpy','cv_bridge','vision_msgs']:
    print(f'  {m:14s}', '已安装' if u.find_spec(m) else '缺失')
"
python3 -c "import torch; print('  CUDA 可用:', torch.cuda.is_available())" 2>/dev/null || true

echo
echo "===== 摄像头 ====="
ls /dev/video* 2>/dev/null || echo "  未检测到摄像头，检查 USB 连接"

echo
echo "===== ROS2 ====="
ls /opt/ros/ 2>/dev/null || echo "  未安装 ROS2"
EOF
    ;;
  pull)
    mkdir -p "$LOCAL_ROOT/05_验收测试/results"
    rsync -avz "$HOST:$REMOTE_DIR/05_验收测试/results/" \
      "$LOCAL_ROOT/05_验收测试/results/"
    rsync -avz --include '*.mp4' --exclude '*' \
      "$HOST:$REMOTE_DIR/04_Jetson部署/" "$LOCAL_ROOT/04_Jetson部署/" || true
    echo "结果已拉回本地"
    ;;
  *)
    sed -n '2,12p' "${BASH_SOURCE[0]}"
    exit 1
    ;;
esac
