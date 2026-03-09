import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"
dino = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
dino.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])

def load_frames(frame_paths):
    imgs = []
    for p in frame_paths:
        img = Image.open(p).convert("RGB")
        imgs.append(transform(img))
    return torch.stack(imgs).to(device)  # [T,3,224,224]

@torch.no_grad()
def extract_cls_tokens(frame_paths):
    x = load_frames(frame_paths)
    feats = dino.get_intermediate_layers(x, n=1)[0]  # [T,197,768]
    cls_tokens = feats[:, 0, :]                       # [T,768]
    cls_tokens = F.normalize(cls_tokens, dim=-1)
    return cls_tokens  # [T,768]

def compute_video_consistency(frame_paths):
    feats = extract_cls_tokens(frame_paths)  # [T,768]
    sims = []
    for t in range(feats.shape[0] - 1):
        sim = torch.dot(feats[t], feats[t + 1])  # cosine sim (already normalized)
        sims.append(sim)
    return torch.stack(sims).mean().item()


def compute_dataset_consistency(root_dir):
    root = Path(root_dir)
    video_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    all_scores = []
    for video_dir in video_dirs:
        frame_paths = sorted(video_dir.glob("*.png"))
        frame_paths = [str(p) for p in frame_paths]
        if len(frame_paths) < 2:
            continue
        score = compute_video_consistency(frame_paths)
        print(f"{video_dir.name}: {score:.4f}")
        all_scores.append(score)
    dataset_mean = np.mean(all_scores)
    print("\n==============================")
    print(f"Dataset Average Consistency: {dataset_mean:.4f}")
    print("==============================")
    return dataset_mean


def compute_consistency(path):
    path = Path(path)
    png_files = list(path.glob("*.png"))
    if len(png_files) > 0:
        frame_paths = sorted(png_files)
        frame_paths = [str(p) for p in frame_paths]

        score = compute_video_consistency(frame_paths)
        print("==============================")
        print(f"Video: {path.name}")
        print(f"Video Consistency: {score:.4f}")
        print("==============================")
        return score
    else:
        return compute_dataset_consistency(path)


# Modify the paths if needed
if __name__ == "__main__":
    compute_consistency("./i2v_attack_dataset/frames")