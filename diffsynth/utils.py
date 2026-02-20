from PIL import Image
from torchvision.transforms.functional import to_pil_image

import torch
import matplotlib.pyplot as plt


def setup_pipe_modules(pipe, attack=False, enable_vram_management=False, num_persistent_param_in_dit=None):
    if attack:
        for module_name in ["dit", "vae", "image_encoder"]:
            if hasattr(pipe, module_name):
                getattr(pipe, module_name).to(pipe.device)
    else:
        for module_name in ["dit", "vae", "image_encoder", "text_encoder"]:
            if hasattr(pipe, module_name):
                getattr(pipe, module_name).to(pipe.device)

    # Optional VRAM management
    if enable_vram_management:
        if num_persistent_param_in_dit:
            pipe.enable_vram_management(num_persistent_param_in_dit=num_persistent_param_in_dit)
        else:
            pipe.enable_vram_management()

    return pipe


def make_hook(saved_features, name):
    def hook(module, inp, out):
        saved_features[name] = out
    return hook

def register_vae_hooks(pipe):
    saved_features = {}

    vae = pipe.vae.model
    encoder = vae.encoder
    decoder = vae.decoder

    # Encoder hooks
    encoder.conv1.register_forward_hook(make_hook(saved_features, "conv1"))
    for i, down in enumerate(encoder.downsamples):
        down.register_forward_hook(make_hook(saved_features, f"downsample_{i}"))
    list(encoder.middle)[-1].register_forward_hook(make_hook(saved_features, "middle"))

    # VAE latent layer hook
    vae.conv1.register_forward_hook(make_hook(saved_features, "mu_logvar"))

    # Decoder hooks
    for i, up in enumerate(decoder.upsamples):
        up.register_forward_hook(make_hook(saved_features, f"upsample_{i}"))

    return saved_features



def plot_loss_curve(loss_history, save_path="loss_curve.png"):
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history, label="Total Loss L")
    plt.xlabel("Step")
    plt.ylabel("Loss Value")
    plt.title("Attack Loss Curve during Optimization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()




def save_adv_result(I_adv, I_adv_before, save_path="I_adv_final.jpg"):
    # Detach and move to CPU for comparison
    I_adv_out = I_adv.detach().cpu().squeeze(0)
    I_adv_before = I_adv_before.detach().cpu()

    # Compute difference metrics
    diff = (I_adv_out - I_adv_before).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Max Diff: {max_diff:.6f}")
    print(f"Mean Diff: {mean_diff:.6f}")

    # Convert tensor from [-1, 1] → [0, 1] and save as image
    I_adv_out = (I_adv_out + 1.0) / 2.0
    I_adv_out = I_adv_out.to(torch.float32).clamp(0, 1)

    I_adv_pil = to_pil_image(I_adv_out)
    I_adv_pil.save(save_path)

    return {"max_diff": max_diff, "mean_diff": mean_diff}
