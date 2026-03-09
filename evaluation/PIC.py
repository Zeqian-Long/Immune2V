import torch
import numpy as np
from pathlib import Path
from PIL import Image
from dreamsim import dreamsim


def compute_video_pic(video_dir, input_dir, model, preprocess, device):
    video_dir = Path(video_dir)
    input_dir = Path(input_dir)
    frame_paths = sorted(video_dir.glob("*.png"))
    if len(frame_paths) < 1:
        return None
    input_path = input_dir / f"{video_dir.name}.png"
    if not input_path.exists():
        print(f"Missing input image: {input_path}")
        return None
    input_img = Image.open(input_path).convert("RGB")
    input_tensor = preprocess(input_img).to(device)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)
    distances = []
    with torch.no_grad():
        for frame_path in frame_paths:
            img = Image.open(frame_path).convert("RGB")
            tensor = preprocess(img).to(device)
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            d = model(input_tensor, tensor)
            distances.append(d.item())
    pic = np.mean([1 - d for d in distances])
    return pic


def compute_dataset_pic(root_dir, input_dir, model, preprocess, device):
    root = Path(root_dir)
    video_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    all_scores = []
    for video_dir in video_dirs:
        score = compute_video_pic(video_dir, input_dir, model, preprocess, device)
        if score is None:
            continue
        print(f"{video_dir.name}: {score:.4f}")
        all_scores.append(score)
    if len(all_scores) == 0:
        print("No valid videos found.")
        return None
    dataset_mean = np.mean(all_scores)
    print("\n==============================")
    print(f"Dataset Average PIC: {dataset_mean:.4f}")
    print("==============================")
    return dataset_mean


def compute_pic(frame_root, input_root):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = dreamsim(pretrained=True)
    model = model.to(device)
    model.eval()
    return compute_dataset_pic(frame_root, input_root, model, preprocess, device)

# Modify the paths if needed
if __name__ == "__main__":
    compute_pic(
        "./i2v_attack_dataset/frames",
        "./data/images"
    )