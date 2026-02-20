import torch
import sys
import os
import json
from PIL import Image

DIFFSYNTH_PATH = "/workspace/Wan-I2V-Attack"
MODEL_DIR = "models/Wan-AI/Wan2.1-I2V-14B-480P"

IMAGE_DIR = "/workspace/Wan-I2V-Attack/data/images"
JSON_PATH = "/workspace/Wan-I2V-Attack/data/prompts.json"

OUT_VIDEO_DIR = "outputs/videos"
OUT_FRAME_DIR = "outputs/frames"

H = 480
W = 832
NUM_FRAMES = 17


sys.path.append(DIFFSYNTH_PATH)

from diffsynth.models.model_manager import ModelManager
from diffsynth.pipelines.wan_video import WanVideoPipeline
from diffsynth.data.video import save_video
from diffsynth.utils import setup_pipe_modules


# -------------------- Load Model --------------------

print("Loading models...")

model_manager = ModelManager(device="cpu")

model_manager.load_models(
    [f"{MODEL_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
    torch_dtype=torch.float16,
)

model_manager.load_models(
    [
        [
            f"{MODEL_DIR}/diffusion_pytorch_model-00001-of-00007.safetensors",
            f"{MODEL_DIR}/diffusion_pytorch_model-00002-of-00007.safetensors",
            f"{MODEL_DIR}/diffusion_pytorch_model-00003-of-00007.safetensors",
            f"{MODEL_DIR}/diffusion_pytorch_model-00004-of-00007.safetensors",
            f"{MODEL_DIR}/diffusion_pytorch_model-00005-of-00007.safetensors",
            f"{MODEL_DIR}/diffusion_pytorch_model-00006-of-00007.safetensors",
            f"{MODEL_DIR}/diffusion_pytorch_model-00007-of-00007.safetensors",
        ],
        f"{MODEL_DIR}/models_t5_umt5-xxl-enc-bf16.pth",
        f"{MODEL_DIR}/Wan2.1_VAE.pth",
    ],
    torch_dtype=torch.bfloat16,
)

pipe = WanVideoPipeline.from_model_manager(
    model_manager,
    torch_dtype=torch.bfloat16,
    device="cuda",
)

pipe = setup_pipe_modules(pipe)

print("Model loaded.")

# -------------------- Load JSON --------------------

with open(JSON_PATH, "r") as f:
    data = json.load(f)

os.makedirs(OUT_VIDEO_DIR, exist_ok=True)
os.makedirs(OUT_FRAME_DIR, exist_ok=True)

# -------------------- Batch Loop --------------------

for item in data:

    img_name = item["img_path"]
    prompt = item["good"]

    img_path = os.path.join(IMAGE_DIR, img_name)

    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        continue

    print(f"Processing {img_name}")

    image = Image.open(img_path).convert("RGB")
    image = image.resize((W, H))

    video = pipe(
        prompt=prompt,
        input_image=image,
        num_inference_steps=25,
        height=H,
        width=W,
        seed=0,
        tiled=False,
        num_frames=NUM_FRAMES,
        cfg_scale=5,
    )

    # ---------------- Save mp4 ----------------

    video_name = img_name.split(".")[0] + ".mp4"
    video_path = os.path.join(OUT_VIDEO_DIR, video_name)

    save_video(video, video_path, fps=15, quality=5)

    # ---------------- Save frames ----------------

    frame_subdir = os.path.join(
        OUT_FRAME_DIR,
        img_name.split(".")[0]
    )
    os.makedirs(frame_subdir, exist_ok=True)

    for i, frame in enumerate(video):
        frame.save(os.path.join(frame_subdir, f"{i:04d}.png"))

    print(f"Saved {video_name}")

print("All done.")
