import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path


sigma = 40 / 255 
input_dir = "./data/images"
output_dir = "./data/random_gaussian"

os.makedirs(output_dir, exist_ok=True)

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

for file in sorted(os.listdir(input_dir)):
    if not file.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    img_path = os.path.join(input_dir, file)
    img = Image.open(img_path).convert("RGB")
    tensor = to_tensor(img)
    tensor = tensor * 2 - 1
    noise = torch.randn_like(tensor) * sigma
    noisy = tensor + noise
    noisy = torch.clamp(noisy, -1, 1)
    noisy = (noisy + 1) / 2
    noisy_img = to_pil(noisy)
    noisy_img.save(os.path.join(output_dir, file))
print("Done.")