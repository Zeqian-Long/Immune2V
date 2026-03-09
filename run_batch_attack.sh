#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

# modify the paths if needed
IMAGE_DIR="data/images"
TGT_IMAGE_DIR="data/images"
CONFIG_FILE="config.yaml"
LOGDIR="logs"
PROGRESS_FILE="logs/progress.txt"

mkdir -p "$LOGDIR"
touch "$PROGRESS_FILE"

for img_path in "$IMAGE_DIR"/*.jpg; do

    img_name=$(basename "$img_path")

    if grep -Fxq "$img_name" "$PROGRESS_FILE"; then
        echo "Skipping $img_name (already done)"
        continue
    fi

    echo "======================================"
    echo "Running attack for $img_name"
    echo "======================================"

    python - <<EOF
import yaml, json, os

CONFIG_FILE = "$CONFIG_FILE"
JSON_PATH = "data/prompts.json"
img_name = os.path.basename("$img_path")

with open(CONFIG_FILE, "r") as f:
    cfg = yaml.safe_load(f)

with open(JSON_PATH, "r") as f:
    data = json.load(f)

item = next(d for d in data if d["img_path"] == img_name)

tgt_img_name = item["tgt_img_path"]
tgt_img_path = os.path.join("$TGT_IMAGE_DIR", tgt_img_name)

cfg["data"]["image_path"] = "$img_path"
cfg["data"]["target_image_path"] = tgt_img_path
cfg["prompt"]["source"] = item["good"]
cfg["prompt"]["target"] = item["bad"]

with open(CONFIG_FILE, "w") as f:
    yaml.safe_dump(cfg, f)

print("Updated config:")
print("image:", "$img_path")
print("target:", tgt_img_path)
print("source prompt:", item["good"])
print("target prompt:", item["bad"])
EOF

    python -m run_attack.preprocess_data \
        2>&1 | tee "$LOGDIR/${img_name}_preprocess.log"

    python -m run_attack.Immune-attack \
        2>&1 | tee "$LOGDIR/${img_name}_attack.log"

    echo "$img_name" >> "$PROGRESS_FILE"

    echo "Finished $img_name"

done

echo "All batch attacks completed."