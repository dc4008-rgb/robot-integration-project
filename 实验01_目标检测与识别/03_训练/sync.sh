#!/usr/bin/env bash
# 数据集上传 / 权重下载。走 SSH 免密别名，不需要密码，也不在脚本里存密码。
#
# 前置: ~/.ssh/config 里配好别名（见 11_远程运维脚本_副本/SSH免密登录配置指南.md）
#   Host advrm
#     HostName connect.westd.seetacloud.com
#     Port <换实例时改这一行>
#     User root
#     IdentityFile ~/.ssh/id_ed25519
#
# 用法:
#   ./sync.sh up          上传数据集 + 训练脚本到服务器
#   ./sync.sh down        下载训练好的权重和曲线图
#   ./sync.sh train       在服务器后台启动训练
#   ./sync.sh log         查看训练日志
set -euo pipefail

HOST="${REMOTE_HOST:-advrm}"
REMOTE_DIR="${REMOTE_DIR:-/root/robot_det}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${RUN_NAME:-desktop_det}"
LOG="/root/train_${RUN_NAME}.log"

case "${1:-}" in
  up)
    ssh "$HOST" "mkdir -p $REMOTE_DIR"
    rsync -avz --progress \
      --exclude 'raw/' --exclude '.DS_Store' \
      "$LOCAL_ROOT/02_数据集/" "$HOST:$REMOTE_DIR/02_数据集/"
    rsync -avz "$LOCAL_ROOT/03_训练/train_yolo.py" "$HOST:$REMOTE_DIR/"
    echo "上传完成"
    ;;
  train)
    ssh "$HOST" "cd $REMOTE_DIR && nohup python train_yolo.py \
      --data $REMOTE_DIR/02_数据集/data.yaml --name $RUN_NAME \
      > $LOG 2>&1 &"
    echo "训练已在后台启动，用 ./sync.sh log 查看进度"
    ;;
  log)
    ssh "$HOST" "tail -n 40 $LOG"
    ;;
  down)
    mkdir -p "$LOCAL_ROOT/03_训练/weights"
    rsync -avz \
      "$HOST:$REMOTE_DIR/runs/detect/$RUN_NAME/weights/best.pt" \
      "$LOCAL_ROOT/03_训练/weights/"
    rsync -avz \
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
