import torch
from PIL import Image
from torchvision import transforms
import lpips


device = "cuda" if torch.cuda.is_available() else "cpu"


# Load LPIPS (Alex / VGG / Squeeze)
lpips_model = lpips.LPIPS(net='alex').to(device)
lpips_model.eval()
to_tensor = transforms.ToTensor()

def load_image(path):
    img = Image.open(path).convert("RGB")
    x = to_tensor(img)                  # [0,1]
    x = x * 2.0 - 1.0                   # [-1,1]
    return x.unsqueeze(0).to(device)    # [1,3,H,W]

@torch.no_grad()
def lpips_frame(img1_path, img2_path):
    x1 = load_image(img1_path)
    x2 = load_image(img2_path)
    dist = lpips_model(x1, x2)          # [1,1,1,1]
    return dist.squeeze()

def video_lpips(video1_frames, video2_frames):
    assert len(video1_frames) == len(video2_frames)
    dists = []
    for p1, p2 in zip(video1_frames, video2_frames):
        dists.append(lpips_frame(p1, p2))
    return torch.stack(dists).mean().item()



lpips_score = video_lpips(ref_frames, pred_frames)
print(f"Video LPIPS: {lpips_score:.4f}")
