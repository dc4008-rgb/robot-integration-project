#!/usr/bin/env bash
# 数据集上传 / 权重下载。走 SSH 公钥认证，不需要密码，也不在脚本里存密码。
#
# 通过 REMOTE_HOST、REMOTE_PORT 指定训练服务器。
#
# 用法:
#   ./sync.sh up          上传数据集 + 训练脚本到服务器
#   ./sync.sh down        下载训练好的权重和曲线图
#   ./sync.sh train       在服务器后台启动训练
#   ./sync.sh log         查看训练日志
set -euo pipefail

HOST="${REMOTE_HOST:-}"
PORT="${REMOTE_PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:-~/robot_det}"
REMOTE_PYTHON="${REMOTE_PYTHON:-python3}"
BASE_MODEL="${BASE_MODEL:-yolov8n.pt}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${RUN_NAME:-desktop_det}"
LOG="${REMOTE_LOG:-$REMOTE_DIR/train_${RUN_NAME}.log}"
SSH=(ssh -p "$PORT" -o StrictHostKeyChecking=accept-new)
RSYNC_RSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new"

if [[ -z "$HOST" ]]; then
  echo "请先设置训练服务器，例如:"
  echo "  export REMOTE_HOST=user@server.example.com"
  echo "  export REMOTE_PORT=22"
  exit 1
fi

case "${1:-}" in
  up)
    "${SSH[@]}" "$HOST" "mkdir -p $REMOTE_DIR/02_数据集"
    rsync -avz --delete --progress --exclude '.DS_Store' -e "$RSYNC_RSH" \
      "$LOCAL_ROOT/02_数据集/images/" "$HOST:$REMOTE_DIR/02_数据集/images/"
    rsync -avz --delete --progress --exclude '.DS_Store' -e "$RSYNC_RSH" \
      "$LOCAL_ROOT/02_数据集/labels/" "$HOST:$REMOTE_DIR/02_数据集/labels/"
    rsync -avz -e "$RSYNC_RSH" \
      "$LOCAL_ROOT/02_数据集/data.yaml" "$HOST:$REMOTE_DIR/02_数据集/"
    rsync -avz -e "$RSYNC_RSH" \
      "$LOCAL_ROOT/03_训练/train_yolo.py" "$HOST:$REMOTE_DIR/"
    echo "上传完成"
    ;;
  train)
    "${SSH[@]}" "$HOST" "cd $REMOTE_DIR; nohup $REMOTE_PYTHON train_yolo.py \
      --data $REMOTE_DIR/02_数据集/data.yaml --model $BASE_MODEL \
      --name $RUN_NAME \
      > $LOG 2>&1 < /dev/null &"
    echo "训练已在后台启动，用 ./sync.sh log 查看进度"
    ;;
  log)
    "${SSH[@]}" "$HOST" "tail -n 40 $LOG"
    ;;
  down)
    mkdir -p "$LOCAL_ROOT/03_训练/weights"
    rsync -avz -e "$RSYNC_RSH" \
      "$HOST:$REMOTE_DIR/runs/detect/$RUN_NAME/weights/best.pt" \
      "$LOCAL_ROOT/03_训练/weights/"
    rsync -avz -e "$RSYNC_RSH" \
      "$HOST:$REMOTE_DIR/runs/detect/$RUN_NAME/" \
      "$LOCAL_ROOT/03_训练/runs_$RUN_NAME/" \
      --exclude 'weights/*.pt'
    echo "权重已下载到 03_训练/weights/best.pt"
    ;;
  *)
    sed -n '2,20p' "${BASH_SOURCE[0]}"
    exit 1
    ;;
esac
