import torch
import sys
diffsynth_path = "/workspace/Wan-I2V-Attack"
sys.path.append(diffsynth_path)
from diffsynth.models.model_manager import ModelManager
from diffsynth.pipelines.wan_video import WanVideoPipeline, model_fn_wan_video
from diffsynth.utils import register_vae_hooks, setup_pipe_modules

from PIL import Image
from tqdm import tqdm

import copy
import yaml
import os


def load_all_models():
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        ["models/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
        torch_dtype=torch.float16, 
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
        torch_dtype=torch.bfloat16,
    )
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
    pipe = setup_pipe_modules(pipe, attack=False)
    return pipe


def prepare_data(pipe, image, target_image, prompt, h=480, w=832, num_frames=1):
    # Encode Prompt
    pipe.load_models_to_device(["text_encoder"])
    with torch.no_grad():
        prompt_emb_posi = pipe.encode_prompt(prompt=prompt, positive=True)      # [1, 512, 4096]

    # Modify if needed
    tiler_kwargs = {"tiled": False, "tile_size": (h / 16, w / 16), "tile_stride": (h / 32, w / 32)}

    # Encode Image
    pipe.load_models_to_device(["image_encoder", "vae"])
    with torch.no_grad():
        image_emb_src = pipe.encode_image(
            image, num_frames=num_frames, height=h, width=w, **tiler_kwargs
        )   # clip: [1, 1 + 256, 1280], y: [1, C (4+16), 1+T/4, 60, 104]

    # Register Hooks at Multiple Resolutions
    saved_features = register_vae_hooks(pipe)

    with torch.no_grad():
        image_emb_tgt = pipe.encode_image(
            target_image, num_frames=num_frames, height=h, width=w, **tiler_kwargs
        )

    target_features = copy.deepcopy(saved_features)
    saved_features = {}

    return prompt_emb_posi, image_emb_src, image_emb_tgt, saved_features, target_features



def obtain_latent_sequence(pipe, h, w, num_frames, prompt_emb, image_emb_src, num_inference_steps=25):

    noise = pipe.generate_noise(
        (1, 16, (num_frames - 1) // 4 + 1, h//8, w//8), 
        seed=0, device="cpu", dtype=torch.float32
    )
    noise = noise.to(dtype=pipe.torch_dtype, device=pipe.device)

    latents_list = []
    latents = noise
    extra_input = pipe.prepare_extra_input(latents)

    pipe.scheduler.set_timesteps(num_inference_steps=num_inference_steps, denoising_strength=1.0, shift=0.0)
    pipe.load_models_to_device(["dit"])

    with torch.no_grad():
        for progress_id, timestep in enumerate(tqdm(pipe.scheduler.timesteps)):
            latents_list.append(latents.detach().cpu())
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = model_fn_wan_video(
                pipe.dit, latents, timestep=timestep,
                **prompt_emb, **image_emb_src, **extra_input
            )
            latents = pipe.scheduler.step(noise_pred, pipe.scheduler.timesteps[progress_id], latents)

    return latents_list


def main():
    pipe = load_all_models()

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    h = cfg["video"]["height"]
    w = cfg["video"]["width"]

    num_frames = cfg["video"]["num_frames"]
    num_inference_steps = cfg["video"]["denoising_steps"]

    image = Image.open(cfg["data"]["image_path"]).resize((w, h))
    target_image = Image.open(cfg["data"]["target_image_path"]).resize((w, h))
    prompt = cfg["prompt"]["text"]

    prompt_emb, image_emb_src, image_emb_tgt, saved_features, target_features = prepare_data(pipe, image, target_image, prompt, 
                                                                                                h=h, w=w, num_frames=num_frames)

    latents_list = obtain_latent_sequence(pipe, h, w, num_frames, prompt_emb, image_emb_src, num_inference_steps=num_inference_steps)

    os.makedirs("cache", exist_ok=True)
    torch.save(prompt_emb, "cache/prompt_emb.pt")
    torch.save(image_emb_src, "cache/image_emb_src.pt")
    torch.save(image_emb_tgt, "cache/image_emb_tgt.pt")
    torch.save(latents_list, "cache/latents_list.pt")
    print("Saved to cache/")
                                                                                          

if __name__ == "__main__":
    main()
