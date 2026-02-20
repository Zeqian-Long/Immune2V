#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

IMAGE_DIR="data/images"
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


    all_images=("$IMAGE_DIR"/*.jpg)
    while true; do
        rand_index=$((RANDOM % ${#all_images[@]}))
        target_path="${all_images[$rand_index]}"
        if [[ "$target_path" != "$img_path" ]]; then
            break
        fi
    done


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

cfg["data"]["image_path"] = "$img_path"
cfg["data"]["target_image_path"] = "$target_path"
cfg["prompt"]["source"] = item["good"]
cfg["prompt"]["target"] = item["bad"]

with open(CONFIG_FILE, "w") as f:
    yaml.safe_dump(cfg, f)

print("Updated config:")
print("image:", "$img_path")
print("target:", "$target_path")
print("source prompt:", item["good"])
print("target prompt:", item["bad"])
EOF

    python ./run_attack/preprocess_data.py \
        2>&1 | tee "$LOGDIR/${img_name}_preprocess.log"

    python ./run_attack/MotionCollapse-attack.py \
        2>&1 | tee "$LOGDIR/${img_name}_attack.log"

    echo "$img_name" >> "$PROGRESS_FILE"

    echo "Finished $img_name"

done

echo "All batch attacks completed."
