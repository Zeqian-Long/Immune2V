import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

to_tensor = transforms.ToTensor()  # [0,1]
def load_image(path):
    img = Image.open(path).convert('RGB')
    return to_tensor(img)          # [3, H, W]

def psnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(max_val ** 2 / mse)


def video_psnr(video1_frames, video2_frames):
    assert len(video1_frames) == len(video2_frames)
    scores = []
    for p1, p2 in zip(video1_frames, video2_frames):
        img1 = load_image(p1)
        img2 = load_image(p2)
        scores.append(psnr(img1, img2))
    return torch.stack(scores).mean().item()

video1 = [
    'gt_0.png', 'gt_1.png', 'gt_2.png',
    'gt_3.png', 'gt_4.png', 'gt_5.png',
    'gt_6.png', 'gt_7.png', 'gt_8.png',
]

video2 = [
    'pred_0.png', 'pred_1.png', 'pred_2.png',
    'pred_3.png', 'pred_4.png', 'pred_5.png',
    'pred_6.png', 'pred_7.png', 'pred_8.png',
]

score = video_psnr(video1, video2)
print(f'Video PSNR: {score:.2f} dB')
