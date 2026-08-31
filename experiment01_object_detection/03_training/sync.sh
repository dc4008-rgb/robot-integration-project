#!/usr/bin/env bash
# Upload datasets and download weights using SSH public-key authentication; no password is required or stored in this script.
#
# Specify the training server with REMOTE_HOST and REMOTE_PORT.
#
# Usage:
#   ./sync.sh up          Upload the dataset and training script to the server
#   ./sync.sh down        Download trained weights and plots
#   ./sync.sh train       Start training in the background on the server
#   ./sync.sh log         View the training log
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
  echo "Configure the training server first, for example:"
  echo "  export REMOTE_HOST=user@server.example.com"
  echo "  export REMOTE_PORT=22"
  exit 1
fi

case "${1:-}" in
  up)
    "${SSH[@]}" "$HOST" "mkdir -p $REMOTE_DIR/02_dataset"
    rsync -avz --delete --progress --exclude '.DS_Store' -e "$RSYNC_RSH" \
      "$LOCAL_ROOT/02_dataset/images/" "$HOST:$REMOTE_DIR/02_dataset/images/"
    rsync -avz --delete --progress --exclude '.DS_Store' -e "$RSYNC_RSH" \
      "$LOCAL_ROOT/02_dataset/labels/" "$HOST:$REMOTE_DIR/02_dataset/labels/"
    rsync -avz -e "$RSYNC_RSH" \
      "$LOCAL_ROOT/02_dataset/data.yaml" "$HOST:$REMOTE_DIR/02_dataset/"
    rsync -avz -e "$RSYNC_RSH" \
      "$LOCAL_ROOT/03_training/train_yolo.py" "$HOST:$REMOTE_DIR/"
    echo "Upload complete"
    ;;
  train)
    "${SSH[@]}" "$HOST" "cd $REMOTE_DIR; nohup $REMOTE_PYTHON train_yolo.py \
      --data $REMOTE_DIR/02_dataset/data.yaml --model $BASE_MODEL \
      --name $RUN_NAME \
      > $LOG 2>&1 < /dev/null &"
    echo "Training started in the background. Run ./sync.sh log to view progress."
    ;;
  log)
    "${SSH[@]}" "$HOST" "tail -n 40 $LOG"
    ;;
  down)
    mkdir -p "$LOCAL_ROOT/03_training/weights"
    rsync -avz -e "$RSYNC_RSH" \
      "$HOST:$REMOTE_DIR/runs/detect/$RUN_NAME/weights/best.pt" \
      "$LOCAL_ROOT/03_training/weights/"
    rsync -avz -e "$RSYNC_RSH" \
      "$HOST:$REMOTE_DIR/runs/detect/$RUN_NAME/" \
      "$LOCAL_ROOT/03_training/runs_$RUN_NAME/" \
      --exclude 'weights/*.pt'
    echo "Weights downloaded to 03_training/weights/best.pt"
    ;;
  *)
    sed -n '2,20p' "${BASH_SOURCE[0]}"
    exit 1
    ;;
esac
