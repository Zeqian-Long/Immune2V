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
    "./blackswan/0000.png",
    "./blackswan/0001.png",
    "./blackswan/0002.png",
    "./blackswan/0003.png",
    "./blackswan/0004.png",
    "./blackswan/0005.png",
    "./blackswan/0006.png",
    "./blackswan/0007.png",
    "./blackswan/0008.png",
    "./blackswan/0009.png",
    "./blackswan/0010.png",
    "./blackswan/0011.png",
    "./blackswan/0012.png",
    "./blackswan/0013.png",
    "./blackswan/0014.png",
    "./blackswan/0015.png",
    "./blackswan/0016.png",
]

text = "A black swan glides gracefully across the calm water, its wings slightly spread as it paddles smoothly with each stroke. The swan's long neck curves elegantly as it turns its head to look around, creating gentle ripples in the water behind it."

score = video_clip_alignment(frames, text)
print(f"CLIP alignment: {score:.4f}")