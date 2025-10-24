import sys

path_to_pip_installs = "/tmp/test_env"
if path_to_pip_installs not in sys.path:
    sys.path.insert(0, path_to_pip_installs)

"""
predict.py

This script generates predictions from a pre-trained two-stage diffusion model.

It takes a path to a specific data split (e.g., 'data/my_data2/test'),
loads the corresponding model checkpoint, and saves the synthetic images,
ground truth images, and ground truth masks to a structured output directory.

Example Usage:
python predict.py \
    --input_path data/my_data2 \
    --model_dir results/exp_brats2 \
    --output_dir predictions \
    --epoch 20 \
    --batch_size 4 \
    --phase test
"""

import os
import argparse
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as sk_psnr

# Assuming the following custom modules are in the PYTHONPATH or the same directory.
# If they are in a specific sub-directory, adjust the imports accordingly.
from dataset.dataset_dixon import CreateDatasetSynthesis, CreateDatasetSynthesis_masks
from backbones.ncsnpp_generator_adagn_feat import NCSNpp, NCSNpp_adaptive
from train_utils import (
    get_time_schedule,
    sample_from_model,
    Diffusion_Coefficients,
    Posterior_Coefficients,
)

# -------------------------
# Data loader
# -------------------------
def make_loader(args, phase, shuffle=False, input_path=None, batch_size=4, num_workers=4):
    """
    Build a DataLoader exactly like in training (DistributedSampler removed).
    """
    if input_path is None:
        input_path = args.input_path
    ds = CreateDatasetSynthesis(phase=phase, input_path=input_path)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader

def make_loader_with_masks(args, phase, shuffle=False, input_path=None, batch_size=4, num_workers=4):
    """
    Build a DataLoader with masks exactly like in training (DistributedSampler removed).
    """
    if input_path is None:
        input_path = args.input_path

    ds = CreateDatasetSynthesis_masks(phase=phase, input_path=input_path)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader

# -------------------------
# Model loading
# -------------------------
def _load_state_dict_flex(module, src, map_location=None, strict=True):
    """
    Load a state_dict into `module`, handling DDP/DataParallel prefixes ("module.")
    and also accepting either a path or a dict.
    """
    if isinstance(src, str):
        state = torch.load(src, map_location=map_location)
    else:
        state = src
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']

    try:
        module.load_state_dict(state, strict=strict)
    except RuntimeError:
        if isinstance(state, dict) and any(k.startswith('module.') for k in state.keys()):
            stripped = {k.replace('module.', '', 1): v for k, v in state.items()}
            module.load_state_dict(stripped, strict=strict)
        else:
            raise

def load_models(args, ckpt_dir, epoch, map_location=None, load_g2=True):
    """
    Load gen_diffusive_1 (and optionally gen_diffusive_2) + sampling helpers.
    """
    if map_location is None:
        map_location = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = map_location if isinstance(map_location, torch.device) else torch.device(map_location)

    gen1 = NCSNpp(args).to(device)
    gen2 = NCSNpp_adaptive(args).to(device) if load_g2 else None

    g1_path = os.path.join(ckpt_dir, f'gen_diffusive_1_{epoch}.pth')
    if not os.path.isfile(g1_path):
        raise FileNotFoundError(f"Missing generator 1 checkpoint: {g1_path}")
    _load_state_dict_flex(gen1, g1_path, device)

    if load_g2:
        g2_path = os.path.join(ckpt_dir, f'gen_diffusive_2_{epoch}.pth')
        if not os.path.isfile(g2_path):
            raise FileNotFoundError(f"Missing generator 2 checkpoint: {g2_path}")
        _load_state_dict_flex(gen2, g2_path, device)

    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    T = get_time_schedule(args, device)

    gen1.eval()
    if gen2 is not None:
        gen2.eval()
    torch.set_grad_enabled(False)

    return device, gen1, gen2, coeff, pos_coeff, T

def load_args_from_content(ckpt_dir, map_location='cpu'):
    """
    Load the saved args Namespace from a training content.pth file.
    """
    content_path = os.path.join(ckpt_dir, "content.pth")
    if not os.path.isfile(content_path):
        raise FileNotFoundError(f"Required 'content.pth' not found in {ckpt_dir}")
    content = torch.load(content_path, map_location=map_location, weights_only=False)
    if "args" not in content:
        raise KeyError("'args' object not found in content.pth. Was it saved during training?")
    return content["args"]

# ----------------------------------------
# Unified prediction function
# ----------------------------------------
def predict_sample(
    args, device, gen1, pos_coeff, T,
    *, gen2, cond1, cond2, cond3, num_timesteps=None, x_T=None
):
    """
    Generate a sample using the two-stage path.
    """
    if num_timesteps is None:
        num_timesteps = args.num_timesteps
    if x_T is None:
        x_T = torch.randn_like(cond1, device=device)

    out = sample_from_model(
        pos_coeff, gen1, cond1, gen2, cond2, cond3,
        num_timesteps, x_T, T, args
    )
    return out

# ----------------------------------------
# Saving utilities
# ----------------------------------------
def save_prediction(image_tensor, save_path):
    """
    Save a predicted image tensor (range [-1, 1]) as a uint8 PNG.
    """
    image_np = image_tensor.squeeze().cpu().numpy()
    image_np = np.clip(image_np, -1.0, 1.0)
    image_uint8 = ((image_np + 1) / 2.0 * 255).astype(np.uint8)
    cv2.imwrite(save_path, image_uint8)

def save_mask(mask_tensor, save_path):
    """
    Save a binary mask tensor (range [0, 1]) as a uint8 PNG.
    """
    mask_np = mask_tensor.squeeze().cpu().numpy()
    mask_uint8 = (mask_np * 255).astype(np.uint8)
    cv2.imwrite(save_path, mask_uint8)


# ----------------------------------------
# Main execution function
# ----------------------------------------
def run_predictions(cli_args):
    """
    Main function to load data, load models, and run prediction loop.
    """
    # --- Path and directory setup ---
    input_path = os.path.normpath(cli_args.input_path)
    phase = os.path.basename(cli_args.phase)
    data_folder_name = os.path.basename(os.path.dirname(input_path))
    model_name = os.path.basename(os.path.normpath(cli_args.model_dir))

    final_output_path = os.path.join(cli_args.output_dir, data_folder_name, model_name, phase)
    real_path = os.path.join(final_output_path, "real")
    synthetic_path = os.path.join(final_output_path, "synthetic")
    real_mask_path = os.path.join(final_output_path, "real_mask")

    print(f"[*] Creating output directories at: {final_output_path}")
    os.makedirs(real_path, exist_ok=True)
    os.makedirs(synthetic_path, exist_ok=True)
    os.makedirs(real_mask_path, exist_ok=True)

    # --- Set seed for reproducibility ---
    if cli_args.seed is not None:
        torch.manual_seed(cli_args.seed)
        np.random.seed(cli_args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cli_args.seed)

    # --- Load model and data ---
    print(f"[*] Loading training args from {cli_args.model_dir}")
    train_args = load_args_from_content(cli_args.model_dir)

    print(f"[*] Loading model from epoch {cli_args.epoch}")
    device, g1, g2, coeff, pos_coeff, T = load_models(
        train_args, ckpt_dir=cli_args.model_dir, epoch=cli_args.epoch
    )
    print(f"[*] Model loaded on device: {device}")

    print(f"[*] Loading '{phase}' data from: {input_path}")
    data_loader = make_loader(
        train_args, phase=phase, input_path=input_path, batch_size=cli_args.batch_size
    )
    mask_loader = make_loader_with_masks(
        train_args, phase=phase, input_path=input_path, batch_size=cli_args.batch_size
    )

    # --- Prediction loop ---
    print(f"[*] Starting prediction loop...")
    total_samples = 0
    zipped_loaders = zip(data_loader, mask_loader)
    
    progress_bar = tqdm(zipped_loaders, total=len(data_loader), desc="Predicting Batches")

    for batch, mask_batch in progress_bar:
        c1, c2, c3, target = batch
        _, _, _, target_mask = mask_batch

        # Move tensors to the computation device
        c1, c2, c3 = c1.to(device), c2.to(device), c3.to(device)

        # Generate prediction
        pred = predict_sample(
            train_args, device, g1, pos_coeff, T,
            gen2=g2, cond1=c1, cond2=c2, cond3=c3
        )

        # Save each item in the batch
        for j in range(pred.size(0)):
            sample_idx = total_samples + j
            
            # Save synthetic image
            save_prediction(pred[j], os.path.join(synthetic_path, f"sample_{sample_idx:04d}.png"))
            
            # Save ground truth image
            save_prediction(target[j], os.path.join(real_path, f"sample_{sample_idx:04d}.png"))
            
            # Save ground truth mask
            save_mask(target_mask[j], os.path.join(real_mask_path, f"sample_{sample_idx:04d}.png"))
            
        total_samples += pred.size(0)

    print(f"\n[+] Prediction complete. Saved {total_samples} samples to {final_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions from a diffusion model.")
    
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to the data directory for a specific split (e.g., "data/my_data2/test").')
    
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to the experiment directory containing model checkpoints and content.pth.')
                        
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Base directory to save the prediction outputs.')

    parser.add_argument('--epoch', type=int, required=True,
                        help='The model epoch number to load for inference.')

    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for inference.')

    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility.')
    
    parser.add_argument('--phase', type=str, default="test")

    args = parser.parse_args()
    run_predictions(args)