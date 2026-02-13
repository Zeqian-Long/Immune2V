import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dino = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
dino.eval().to(device)


transform = transforms.Compose([
    transforms.Resize(256),          # short side -> 256 (keep aspect ratio)
    transforms.CenterCrop(224),      # fixed input size
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])

@torch.no_grad()
def extract_dino_cls(img_path):
    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)  # [1, 3, 224, 224]
    feat = dino(x)                              # [1, 768] (CLS token)
    feat = F.normalize(feat, dim=-1)
    return feat.squeeze(0)                      # [768]


def subject_consistency(frame_paths):
    feats = [extract_dino_cls(p) for p in frame_paths]
    sims = []
    for t in range(len(feats) - 1):
        sim = torch.dot(feats[t], feats[t + 1])  # cosine sim
        sims.append(sim)
    return torch.stack(sims).mean().item()


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

score = subject_consistency(frames)
print(f'Subject Consistency: {score:.4f}')
