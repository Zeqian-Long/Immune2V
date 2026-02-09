import torch
import sys
diffsynth_path = "/workspace/Wan-I2V-Attack"
sys.path.append(diffsynth_path)
from diffsynth.models.model_manager import ModelManager
from diffsynth.pipelines.wan_video import WanVideoPipeline, prompt_clip_attn_loss
from diffsynth.utils import setup_pipe_modules, plot_loss_curve, save_adv_result
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import random
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
    pipe = setup_pipe_modules(pipe, attack=True)
    return pipe



def init_adv_image(I, epsilon=0.03, value_range=(-1.0, 1.0), device=None):
    if not isinstance(I, torch.Tensor):
        raise TypeError("I must be a torch.Tensor")
    I = I.detach().clone()
    if device is not None:
        I = I.to(device)
    noise = torch.empty_like(I).uniform_(-epsilon, epsilon)
    I_adv = I + noise
    I_adv = torch.clamp(I_adv, value_range[0], value_range[1])
    I_adv.requires_grad_(True)
    return I_adv



def run_attack(pipe, image, h, w, num_frames, prompt_emb_src, prompt_emb_tgt, image_emb_src, image_emb_tgt, latents_list, num_steps=400, epsilon=20.0 / 255 * 2, step_size=2.0 / 255 * 2, info=None):

    tiler_kwargs = {"tiled": False, "tile_size": (h / 16, w / 16), "tile_stride": (h / 32, w / 32)}

    I_adv = pipe.preprocess_image(image).to(pipe.device).detach().requires_grad_(True)
    I_adv_before = I_adv.clone().detach()

    I_adv = init_adv_image(I_adv, epsilon=epsilon, value_range=(-1.0, 1.0))


    loss_history = []

    pbar = tqdm(range(num_steps), desc="Optimizing")

    for step in pbar:
        if I_adv.grad is not None:
            I_adv.grad.zero_()

        pipe.load_models_to_device(["vae", "image_encoder"])
        image_emb_adv = pipe.encode_image(I_adv, num_frames=num_frames, height=h, width=w, **tiler_kwargs)

        # print(F.mse_loss(image_emb_adv["y"][0, 4:, 0], image_emb_tgt[0, :, 0]))
        # print(F.mse_loss(image_emb_adv["y"][0, 4:, 1], image_emb_tgt[0, :, 1]))
        # print(F.mse_loss(image_emb_adv["y"][0, 4:, 2], image_emb_tgt[0, :, 2]))
        # print(F.mse_loss(image_emb_adv["y"][:, 4:, :], image_emb_tgt))

        # MSE Loss
        # enc_loss = F.mse_loss(image_emb_adv["y"][:, 4:, :], image_emb_tgt["y"][:, 4:, :])
        enc_loss = (F.mse_loss(image_emb_adv["y"][0, 4:, 0], image_emb_tgt[0, :, 0])
                    + F.mse_loss(image_emb_adv["y"][0, 4:, 1], image_emb_tgt[0, :, 1])
                    + F.mse_loss(image_emb_adv["y"][0, 4:, 2], image_emb_tgt[0, :, 2]))

        # Cosine Loss
        # z_adv = image_emb_adv["y"][:, 4:, :, :, :] 
        # z_tgt = image_emb_tgt["y"][:, 4:, :, :, :]
        # z_adv_v = z_adv.permute(0, 2, 3, 4, 1)     
        # z_tgt_v = z_tgt.permute(0, 2, 3, 4, 1)
        # enc_loss = 1 - F.cosine_similarity(z_adv_v, z_tgt_v, dim=-1).mean()

        pipe.scheduler.set_timesteps(num_inference_steps=25, denoising_strength=1.0, shift=5.0)
        idx = random.randrange(len(pipe.scheduler.timesteps))

        # noise = pipe.generate_noise((1, 16, (num_frames - 1) // 4 + 1, h//8, w//8), seed=None)
        # noise = noise.to(dtype=pipe.torch_dtype, device=pipe.device)

        # Sample the clean latent
        adv_latents = latents_list[idx].to(dtype=pipe.torch_dtype, device=pipe.device)
        timestep = pipe.scheduler.timesteps[idx].unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

        if info is not None:
            info['t'] = timestep.item()

        extra_input = pipe.prepare_extra_input(adv_latents)

        pipe.load_models_to_device(["dit"])


        noise_pred = prompt_clip_attn_loss(pipe.dit, adv_latents, timestep=timestep, **prompt_emb_src, **image_emb_adv, **extra_input, info=None)
        noise_pred_tar = prompt_clip_attn_loss(pipe.dit, adv_latents, timestep=timestep, **prompt_emb_tgt, **image_emb_src, **extra_input, info=None)  
        attn_loss = torch.norm(noise_pred - noise_pred_tar, dim=-1).mean()             

        # attn_loss = prompt_clip_attn_loss(pipe.dit, adv_latents, timestep=timestep, **prompt_emb, **image_emb_adv, **extra_input, info=info)

        # scale the loss if needed
        L = 1 * attn_loss + 1 * enc_loss

        pbar.set_postfix(loss=f"{L.item():.4f}", t=timestep.item())

        loss_history.append(L.item())
        L.backward()

        # if step == 100:
        #     step_size *= 0.5
        # if step == 200:
        #     step_size *= 0.5

        # PGD, Clamp
        sgn = I_adv.grad.data.sign()
        I_adv.data = I_adv.data - step_size * sgn
        delta = torch.clamp(I_adv - I_adv_before, min=-epsilon, max=epsilon)
        I_adv.data = torch.clamp(I_adv_before + delta, -1.0, 1.0)

    plot_loss_curve(loss_history)
    metrics = save_adv_result(I_adv, I_adv_before, save_path="I_adv_final_hike.jpg")



def main():
    pipe = load_all_models()

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    h = cfg["video"]["height"]
    w = cfg["video"]["width"]
    num_frames = cfg["video"]["num_frames"]

    image = Image.open(cfg["data"]["image_path"]).resize((w, h))
    target_image = Image.open(cfg["data"]["target_image_path"]).resize((w, h))


    cache_files = {
        'prompt_emb_src': 'cache/prompt_emb_src.pt',
        'prompt_emb_tgt': 'cache/prompt_emb_tgt.pt',
        'image_emb_src': 'cache/image_emb_src.pt',
        'image_emb_tgt': 'cache/image_emb_tgt.pt',
        'latents_list': 'cache/latents_list.pt',
        'info': 'cache/info.pt',
    }

    if all(os.path.exists(f) for f in cache_files.values()):
        print("Loading cached data...")
        prompt_emb_src = torch.load(cache_files['prompt_emb_src'])
        prompt_emb_tgt = torch.load(cache_files['prompt_emb_tgt'])
        image_emb_src = torch.load(cache_files['image_emb_src'])
        image_emb_tgt = torch.load(cache_files['image_emb_tgt'])
        latents_list = torch.load(cache_files['latents_list'])
        info = torch.load(cache_files['info'])
        print("Loaded successfully!")
    else:
        raise FileNotFoundError(
            "Data not found in cache/. Please prepare and preprocess data first by running preprocess_data.py!"
        )

                                                                                            
    num_steps = cfg["attack"]["num_steps"]
    epsilon = eval(cfg["attack"]["epsilon"])
    step_size = epsilon / 10

    info['attack'] = True
                                         
    run_attack(pipe, image, h, w, num_frames, prompt_emb_src, prompt_emb_tgt, image_emb_src, image_emb_tgt, latents_list,
                num_steps=num_steps, epsilon=epsilon, step_size=step_size, info=info)


if __name__ == "__main__":
    main()
