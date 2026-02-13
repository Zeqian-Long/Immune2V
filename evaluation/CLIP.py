import torch
import torch.nn.functional as F
from PIL import Image
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

@torch.no_grad()
def encode_text(text):
    tokens = clip.tokenize([text]).to(device)
    feat = model.encode_text(tokens)
    feat = F.normalize(feat, dim=-1)
    return feat.squeeze(0)  # [D]

@torch.no_grad()
def image_text_similarity(img_path, text_feat):
    img = Image.open(img_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    img_feat = model.encode_image(x)
    img_feat = F.normalize(img_feat, dim=-1)
    return torch.dot(img_feat.squeeze(0), text_feat)

def video_clip_alignment(frame_paths, text):
    text_feat = encode_text(text)
    sims = []
    for p in frame_paths:
        sims.append(image_text_similarity(p, text_feat))
    return torch.stack(sims).mean().item()


# a list of image paths
frames = [
    "./frames_clean/0000.png",
    "./frames_clean/0001.png",
    "./frames_clean/0002.png",
    "./frames_clean/0003.png",
    "./frames_clean/0004.png",
    "./frames_clean/0005.png",
    "./frames_clean/0006.png",
    "./frames_clean/0007.png",
    "./frames_clean/0008.png",
    "./frames_clean/0009.png",
    "./frames_clean/0010.png",
    "./frames_clean/0011.png",
    "./frames_clean/0012.png",
    "./frames_clean/0013.png",
    "./frames_clean/0014.png",
    "./frames_clean/0015.png",
    "./frames_clean/0016.png",
]

text = "一辆银色越野车沿着蜿蜒的山间公路平稳行驶"

score = video_clip_alignment(frames, text)
print(f"CLIP alignment: {score:.4f}")