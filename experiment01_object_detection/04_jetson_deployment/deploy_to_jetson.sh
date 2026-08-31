#!/usr/bin/env bash
# Sync the project to the Jetson and check its environment.
#
# Configure the device address first (run `hostname -I` on the Jetson to find its IP):
#   export JETSON_HOST=jetson@192.168.1.100
#
# Usage:
#   ./deploy_to_jetson.sh push      Sync code to the device, excluding datasets and weights
#   ./deploy_to_jetson.sh model best.pt   Upload trained weights separately
#   ./deploy_to_jetson.sh check     Check whether the device environment is complete
#   ./deploy_to_jetson.sh pull-data [train|val|all]  Download recollected data (default: val)
#   ./deploy_to_jetson.sh pull      Download test results and videos from the device
set -euo pipefail

HOST="${JETSON_HOST:-}"
REMOTE_DIR="${JETSON_DIR:-~/robot_det}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "$HOST" ]]; then
  echo "Configure the device address first, for example:"
  echo "  export JETSON_HOST=jetson@192.168.1.100"
  exit 1
fi

case "${1:-}" in
  push)
    rsync -avz --progress \
      --exclude '02_dataset/raw/' --exclude '02_dataset/images/' \
      --exclude '02_dataset/labels/' --exclude '02_dataset/recollection/' \
      --exclude '02_dataset/external_openimages_yolo/' \
      --exclude '02_dataset/.fiftyone_openimages_cache/' \
      --exclude 'training_dataset/' \
      --exclude '03_training/runs_*' \
      --exclude '__pycache__/' --exclude '.DS_Store' --exclude '.git/' \
      --exclude '*.pt' --exclude '*.engine' \
      "$LOCAL_ROOT/" "$HOST:$REMOTE_DIR/"
    echo "Code synced to $HOST:$REMOTE_DIR"
    ;;
  model)
    weights="${2:-$LOCAL_ROOT/03_training/weights/best.pt}"
    [[ -f "$weights" ]] || { echo "Weights file not found: $weights"; exit 1; }
    rsync -avz --progress "$weights" "$HOST:$REMOTE_DIR/04_jetson_deployment/"
    echo "Weights uploaded. Next, export the TensorRT engine on the device:"
    echo "  ssh $HOST 'cd $REMOTE_DIR/04_jetson_deployment && python3 export_engine.py --weights best.pt --half'"
    ;;
  check)
    ssh "$HOST" 'bash -s' <<'EOF'
echo "===== Device Information ====="
cat /etc/nv_tegra_release 2>/dev/null | head -1 || echo "(Could not read the JetPack version)"
echo "Model: $(cat /proc/device-tree/model 2>/dev/null | tr -d '\0')"
echo "Power mode: $(sudo -n nvpmodel -q 2>/dev/null | tail -1 || echo 'Run nvpmodel -q manually')"

echo
echo "===== Python Dependencies ====="
python3 -c "
import importlib.util as u
for m in ['ultralytics','cv2','torch','torchvision','tensorrt','rclpy','cv_bridge','vision_msgs']:
    print(f'  {m:14s}', 'installed' if u.find_spec(m) else 'missing')
"
python3 -c "import torch; print('  CUDA available:', torch.cuda.is_available())" 2>/dev/null || true

echo
echo "===== Camera ====="
ls /dev/video* 2>/dev/null || echo "  No camera detected; check the USB connection"

echo
echo "===== ROS2 ====="
ls /opt/ros/ 2>/dev/null || echo "  ROS2 is not installed"
EOF
    ;;
  pull-data)
    scope="${2:-val}"
    case "$scope" in
      train|val)
        mkdir -p "$LOCAL_ROOT/02_dataset/recollection/$scope"
        rsync -avz --progress "$HOST:$REMOTE_DIR/02_dataset/recollection/$scope/" \
          "$LOCAL_ROOT/02_dataset/recollection/$scope/"
        ;;
      all)
        mkdir -p "$LOCAL_ROOT/02_dataset/recollection"
        rsync -avz --progress "$HOST:$REMOTE_DIR/02_dataset/recollection/" \
          "$LOCAL_ROOT/02_dataset/recollection/"
        ;;
      *)
        echo "Unknown data scope: ${scope}; choose train, val, or all"
        exit 1
        ;;
    esac
    echo "Recollected $scope data downloaded to $LOCAL_ROOT/02_dataset/recollection"
    ;;
  pull)
    mkdir -p "$LOCAL_ROOT/05_acceptance_testing/results"
    rsync -avz "$HOST:$REMOTE_DIR/05_acceptance_testing/results/" \
      "$LOCAL_ROOT/05_acceptance_testing/results/"
    rsync -avz --include '*.mp4' --exclude '*' \
      "$HOST:$REMOTE_DIR/04_jetson_deployment/" "$LOCAL_ROOT/04_jetson_deployment/" || true
    echo "Results downloaded locally"
    ;;
  *)
    sed -n '2,12p' "${BASH_SOURCE[0]}"
    exit 1
    ;;
esac
