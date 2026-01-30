import torch
import sys
diffsynth_path = "/workspace/Wan-I2V-Attack"
sys.path.append(diffsynth_path)
from diffsynth.models.model_manager import ModelManager
from diffsynth.pipelines.wan_video import WanVideoPipeline, prompt_clip_attn_loss, model_fn_wan_video
from diffsynth.utils import crop_and_resize, register_vae_hooks, setup_pipe_modules, plot_loss_curve, save_adv_result

from PIL import Image
from tqdm import tqdm

import copy
import random
import yaml


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
    pipe = setup_pipe_modules(pipe)
    return pipe


def prepare_data(pipe, image, target_image, prompt, h=480, w=832, num_frames=1):
    # Encode Prompt
    pipe.load_models_to_device(["text_encoder"])
    with torch.no_grad():
        prompt_emb_posi = pipe.encode_prompt(prompt=prompt, positive=True)      # [1, 512, 4096]

    # Modify if needed
    tiler_kwargs = {"tiled": False, "tile_size": (h / 16, w / 16), "tile_stride": (h / 32, w / 32)}

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



def obtain_latent_sequence(pipe, h, w, num_frames, prompt_emb, image_emb_src):
    noise = pipe.generate_noise(
        (1, 16, (num_frames - 1) // 4 + 1, h//8, w//8), 
        seed=0, device="cpu", dtype=torch.float32
    )
    noise = noise.to(dtype=pipe.torch_dtype, device=pipe.device)

    latents_list = []
    latents = noise
    extra_input = pipe.prepare_extra_input(latents)

    pipe.scheduler.set_timesteps(num_inference_steps=50, denoising_strength=1.0, shift=5.0)
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

    # Save
    # torch.save(latents_list, 'latents_list.pt')

    return latents_list



def init_adv_image(I, epsilon=0.03, value_range=(-1.0, 1.0)):
    if not isinstance(I, torch.Tensor):
        raise TypeError("I must be a torch.Tensor")
    I_adv = I.clone()
    I_adv = I_adv + torch.empty_like(I_adv).uniform_(-epsilon, epsilon)
    I_adv = torch.clamp(I_adv, value_range[0], value_range[1]).detach()
    I_adv.requires_grad_(True)
    return I_adv



def run_attack(pipe, image, h, w, num_frames, prompt_emb, image_emb_src, image_emb_tgt, num_steps=400, epsilon=20.0 / 255 * 2, step_size=2.0 / 255 * 2):
    latents_list = obtain_latent_sequence(pipe, h, w, num_frames, prompt_emb, image_emb_src)

    tiler_kwargs = {"tiled": False, "tile_size": (h / 16, w / 16), "tile_stride": (h / 32, w / 32)}

    I_adv = pipe.preprocess_image(image).to(pipe.device).detach().requires_grad_(True)
    I_adv_before = I_adv.clone().detach()

    I_adv = init_adv_image(I_adv, epsilon=epsilon, value_range=(-1.0, 1.0))

    loss_history = []

    for step in tqdm(range(num_steps), desc="Optimizing"):
        if I_adv.grad is not None:
            I_adv.grad.zero_()

        pipe.load_models_to_device(["vae", "image_encoder"])
        image_emb_adv = pipe.encode_image(I_adv, num_frames=num_frames, height=h, width=w, **tiler_kwargs)

        L_enc_1 = torch.nn.functional.mse_loss(image_emb_adv["y"][:, 4:, :], image_emb_tgt["y"][:, 4:, :])


        pipe.scheduler.set_timesteps(num_inference_steps=50, denoising_strength=1.0, shift=5.0)
        idx = random.randrange(len(pipe.scheduler.timesteps))


        adv_latents = latents_list[idx].to(dtype=pipe.torch_dtype, device=pipe.device)
        timestep = pipe.scheduler.timesteps[idx].unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        extra_input = pipe.prepare_extra_input(adv_latents)

        pipe.load_models_to_device(["dit"])
        attn_loss = prompt_clip_attn_loss(
            pipe.dit, adv_latents, timestep=timestep,
            **prompt_emb, **image_emb_adv, **extra_input
        )

        L = attn_loss + L_enc_1
        print(f"Step {step+1}/{num_steps}, Loss: {L.item():.6f}")
        loss_history.append(L.item())
        L.backward()

        # if step == 100:
        #     step_size *= 0.5
        # if step == 200:
        #     step_size *= 0.5

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
    prompt = cfg["prompt"]["text"]

    prompt_emb, image_emb_src, image_emb_tgt, saved_features, target_features = prepare_data(pipe, image, target_image, prompt, 
                                                                                                h=h, w=w, num_frames=num_frames)

    num_steps = cfg["attack"]["num_steps"]
    epsilon = eval(cfg["attack"]["epsilon"])
    step_size = epsilon / 10
                                         
    run_attack(pipe, image, h, w, num_frames, prompt_emb, image_emb_src, image_emb_tgt, 
                num_steps=num_steps, epsilon=epsilon, step_size=step_size)


if __name__ == "__main__":
    main()
