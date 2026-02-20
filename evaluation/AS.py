import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from urllib.request import urlretrieve
from os.path import expanduser
import open_clip

def get_aesthetic_model(clip_model="vit_l_14"):

    home = expanduser("~")
    cache_folder = home + "/.cache/emb_reader"
    path_to_model = cache_folder + "/sa_0_4_" + clip_model + "_linear.pth"

    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_"
            + clip_model
            + "_linear.pth"
        )
        urlretrieve(url_model, path_to_model)

    if clip_model == "vit_l_14":
        m = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        m = nn.Linear(512, 1)
    else:
        raise ValueError("Unsupported CLIP model")

    s = torch.load(path_to_model, map_location="cpu")
    m.load_state_dict(s)
    m.eval()
    return m

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14",
    pretrained="openai"
)

clip_model = clip_model.to(device)
clip_model.eval()

aesthetic_model = get_aesthetic_model("vit_l_14").to(device)
aesthetic_model.eval()

def compute_aesthetic_score(image_path):

    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        score = aesthetic_model(image_features)

    return score.item()

def compute_video_aesthetic(video_dir):

    scores = []

    for file in sorted(os.listdir(video_dir)):
        if file.lower().endswith(".png"):
            image_path = os.path.join(video_dir, file)
            s = compute_aesthetic_score(image_path)
            scores.append(s)

    if len(scores) == 0:
        return None

    mean_score = np.mean(scores)

    print("==============================")
    print(f"Video: {video_dir}")
    print(f"Average Aesthetic Score: {mean_score:.4f}")
    print("==============================")

    return mean_score


def compute_dataset_aesthetic(root_dir):

    video_scores = []

    for folder in sorted(os.listdir(root_dir)):

        video_path = os.path.join(root_dir, folder)

        if not os.path.isdir(video_path):
            continue

        print(f"\nProcessing video: {folder}")
        score = compute_video_aesthetic(video_path)

        if score is not None:
            video_scores.append(score)

    if len(video_scores) == 0:
        print("No valid videos found.")
        return None

    dataset_mean = np.mean(video_scores)

    print("\n==============================")
    print(f"Dataset Average Aesthetic Score: {dataset_mean:.4f}")
    print("==============================")

    return dataset_mean



if __name__ == "__main__":
    # put the folder path here
    root_folder = "./frames_ddd"
    compute_video_aesthetic(root_folder)