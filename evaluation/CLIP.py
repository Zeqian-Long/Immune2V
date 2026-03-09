import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
import clip
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()


@torch.no_grad()
def encode_text(text):
    tokens = clip.tokenize([text]).to(device)
    feat = model.encode_text(tokens)
    feat = F.normalize(feat, dim=-1)
    return feat.squeeze(0)


@torch.no_grad()
def image_text_similarity(img_path, text_feat):
    img = Image.open(img_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    img_feat = model.encode_image(x)
    img_feat = F.normalize(img_feat, dim=-1)
    return torch.dot(img_feat.squeeze(0), text_feat).item()


def compute_video_clip(video_dir, text):
    text_feat = encode_text(text)
    frame_paths = [
        os.path.join(video_dir, f)
        for f in sorted(os.listdir(video_dir))
        if f.lower().endswith(".png")
    ]
    if len(frame_paths) == 0:
        return None
    sims = []
    for p in frame_paths:
        sims.append(image_text_similarity(p, text_feat))

    return np.mean(sims)



def compute_dataset_clip(root_dir, prompts_json_path):
    with open(prompts_json_path, "r") as f:
        prompts = json.load(f)
    prompt_dict = {
        item["img_path"]: item["good"]
        for item in prompts
    }
    video_scores = []

    for folder in sorted(os.listdir(root_dir)):
        video_path = os.path.join(root_dir, folder)
        if not os.path.isdir(video_path):
            continue
        key = folder + ".jpg"
        if key not in prompt_dict:
            print(f"Prompt not found for {folder}")
            continue
        text = prompt_dict[key]
        print(f"Processing video: {folder}")
        score = compute_video_clip(video_path, text)
        if score is not None:
            video_scores.append(score)
            print(f"Video CLIP alignment: {score:.4f}")
    if len(video_scores) == 0:
        print("No valid videos found.")
        return None
    dataset_mean = np.mean(video_scores)
    print("\n==============================")
    print(f"Dataset Average CLIP Alignment: {dataset_mean:.4f}")
    print("==============================")
    return dataset_mean


# Modify the paths if needed
if __name__ == "__main__":
    root_folder = "./i2v_attack_dataset/frames"
    prompts_json = "./data/prompts.json"
    compute_dataset_clip(root_folder, prompts_json)