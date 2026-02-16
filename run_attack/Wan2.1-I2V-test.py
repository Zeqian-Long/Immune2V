import torch
import sys
import os
diffsynth_path = "/workspace/Wan-I2V-Attack"
sys.path.append(diffsynth_path)
from diffsynth.models.model_manager import ModelManager
from diffsynth.pipelines.wan_video import WanVideoPipeline
from diffsynth.data.video import save_video, VideoData, LowMemoryImageFolder
from diffsynth.utils import setup_pipe_modules
from PIL import Image

# Load models
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    ["models/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
    torch_dtype=torch.float16, # Image Encoder is loaded with float16
)
model_manager.load_models(
    [
        [
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00001-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00002-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00003-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00004-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00005-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00006-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00007-of-00007.safetensors",
        ],
        "models/Wan-AI/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth",
        "models/Wan-AI/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth",
    ],
    torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
)
pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")


# --------------------------------------------- Testing ---------------------------------------------

h = 480
w = 832

image = Image.open("I_adv_final_hike.jpg")
# image = Image.open("data/images/dog.jpg")
image = image.resize((w, h))

# pipe.enable_vram_management(num_persistent_param_in_dit=6*10**9) # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.
pipe = setup_pipe_modules(pipe)


video = pipe(
    prompt="A golden retriever walks gracefully through a grassy field, sniffing the ground and exploring its surroundings. As it moves, its tail wags gently, reflecting its curiosity and excitement with each step.",
    input_image=image,
    num_inference_steps=25, height=h, width=w,
    seed=0, tiled=False, num_frames=17, cfg_scale=5,
)
save_video(video, "clean_attacked.mp4", fps=15, quality=5)


out_dir = "frames_attacked"
os.makedirs(out_dir, exist_ok=True)
for i, frame in enumerate(video):
    frame.save(os.path.join(out_dir, f"{i:04d}.png"))
