#!/usr/bin/env python3
"""
augment_mri.py

Intensity-only subject-level augmentation for aligned multi-contrast MRI.
- Applies the SAME random intensity transforms to all modalities for a given index.
- No rotations/translations/flips (keeps geometry and orientation unchanged).
- By default, includes the ORIGINAL images in the output (prepended before augmented copies).

Interfaces:
1) CLI usage (paths + args)
2) Notebook usage: call Python functions with a config dict (no CLI flags)

Inputs (expected default paths for CLI):
    data/my_data/train/DIXON.npy
    data/my_data/train/T1_mapping_fl2d.npy
    data/my_data/train/Diffusion.npy
    data/my_data/train/BOLD.npy

Outputs (default):
    data/my_data_aug/train/DIXON.npy
    data/my_data_aug/train/T1_mapping_fl2d.npy
    data/my_data_aug/train/Diffusion.npy
    data/my_data_aug/train/BOLD.npy

Example CLI:
    python augment_mri.py --save_root data/my_data --out_root data/my_data_aug \
        --num_aug 1 --seed 42 --include_original 1

Example notebook usage:
    from augment_mri import augment_and_save, augment_in_memory, default_config

    cfg = default_config()
    cfg.update({
        "save_root": "data/my_data",
        "out_root": "data/my_data_aug",
        "num_aug": 2,
        "seed": 123,
        "include_original": True,
    })
    augment_and_save(cfg)  # writes .npy files

    # Or, in-memory:
    dixon = np.load("data/my_data/train/DIXON.npy")
    t1map = np.load("data/my_data/train/T1_mapping_fl2d.npy")
    diffusion = np.load("data/my_data/train/Diffusion.npy")
    bold = np.load("data/my_data/train/BOLD.npy")
    out = augment_in_memory(dixon, t1map, diffusion, bold, cfg)
    dixon_out, t1map_out, diffusion_out, bold_out = out
"""

import argparse
import os
import numpy as np

# ----------------------- Helpers -----------------------

def robust_minmax(x, lower_p=1.0, upper_p=99.0, eps=1e-8):
    """Robust min-max normalize slice to [0,1] using percentiles."""
    lo = np.percentile(x, lower_p)
    hi = np.percentile(x, upper_p)
    if hi - lo < eps:
        return np.zeros_like(x, dtype=np.float32)
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + eps)
    return x.astype(np.float32)

def gamma_correction(x, gamma):
    """Apply gamma correction to [0,1] image."""
    x = np.clip(x, 0.0, 1.0)
    return np.power(x, gamma).astype(np.float32)

def make_bias_field(h, w, rng, magnitude=0.5, num_terms=3):
    """
    Create a smooth multiplicative bias field in [1 - magnitude, 1 + magnitude].
    Uses a small sum of 2D cosine bases with random frequencies/phases.
    Pure NumPy (no scipy dependency).
    """
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    field = np.zeros((h, w), dtype=np.float32)
    for _ in range(num_terms):
        fx = rng.uniform(0.5, 2.0)
        fy = rng.uniform(0.5, 2.0)
        phx = rng.uniform(0, 2*np.pi)
        phy = rng.uniform(0, 2*np.pi)
        field += np.cos(2*np.pi*fx*xx + phx) * np.cos(2*np.pi*fy*yy + phy)
    # normalize to [-1, 1] then to [1-magnitude, 1+magnitude]
    field -= field.min()
    denom = (field.max() - field.min()) if (field.max() > field.min()) else 1.0
    field = 2.0 * (field / denom) - 1.0
    field = 1.0 + magnitude * field
    return field.astype(np.float32)

def add_rician_noise(x, rng, sigma=0.02):
    """
    Add Rician noise to [0,1] image x: sqrt((x + n1)^2 + n2^2), n1,n2 ~ N(0, sigma^2)
    """
    n1 = rng.normal(0.0, sigma, size=x.shape).astype(np.float32)
    n2 = rng.normal(0.0, sigma, size=x.shape).astype(np.float32)
    return np.sqrt(np.maximum((x + n1)**2 + n2**2, 0.0)).astype(np.float32)

def apply_subject_intensity_aug(stacked_modalities, rng, cfg):
    """
    Apply the SAME intensity transforms to all modalities for the current subject index.
    stacked_modalities: list of 2D arrays [dixon, t1map, diffusion, bold] (H x W).
    Returns a new list of transformed arrays in [0,1].
    """
    # 1) robust normalize each modality to [0,1]
    imgs = [robust_minmax(im) for im in stacked_modalities]

    # Sample shared augmentation parameters
    do_scale = rng.random() < cfg['p_scale']
    scale = rng.uniform(1.0 - cfg['scale_range'], 1.0 + cfg['scale_range']) if do_scale else 1.0
    do_shift = rng.random() < cfg['p_shift']
    shift = rng.uniform(-cfg['shift_range'], cfg['shift_range']) if do_shift else 0.0
    do_gamma = rng.random() < cfg['p_gamma']
    gamma = rng.uniform(cfg['gamma_min'], cfg['gamma_max']) if do_gamma else 1.0
    do_bias = rng.random() < cfg['p_bias']
    if do_bias:
        bias = make_bias_field(imgs[0].shape[0], imgs[0].shape[1], rng,
                               magnitude=cfg['bias_magnitude'], num_terms=cfg['bias_terms'])
    else:
        bias = None
    do_noise = rng.random() < cfg['p_noise']
    sigma = rng.uniform(cfg['noise_min'], cfg['noise_max']) if do_noise else 0.0

    out = []
    for x in imgs:
        y = x.copy()
        y = y * scale + shift
        y = np.clip(y, 0.0, 1.0)
        y = gamma_correction(y, gamma)
        if bias is not None:
            y = np.clip(y * bias, 0.0, 1.0)
        if sigma > 0.0:
            y = np.clip(add_rician_noise(y, rng, sigma=sigma), 0.0, 1.0)
        out.append(y.astype(np.float32))
    return out

# ----------------------- Public API (Notebook-friendly) -----------------------

def default_config():
    """Return a dict with sensible defaults for augmentation + IO."""
    return {
        # IO
        "save_root": "data/my_data",
        "out_root": "data/my_data_aug",
        "subset": "train",                 # subfolder under roots
        "include_original": True,          # keep originals at the beginning of outputs
        "num_aug": 1,                      # number of augmented copies to generate
        "seed": 42,

        # Augmentation hyperparameters
        "p_scale": 0.7,
        "scale_range": 0.10,               # multiplicative gain in [0.9, 1.1]
        "p_shift": 0.7,
        "shift_range": 0.10,               # additive shift in [-0.1, 0.1]
        "p_gamma": 0.5,
        "gamma_min": 0.7,
        "gamma_max": 1.3,
        "p_bias": 0.4,
        "bias_magnitude": 0.4,
        "bias_terms": 3,
        "p_noise": 0.4,
        "noise_min": 0.01,
        "noise_max": 0.04,
    }

def augment_in_memory(dixon, t1map, diffusion, bold, cfg=None):
    """
    In-memory augmentation. Returns augmented arrays (and optionally originals).

    Args:
        dixon, t1map, diffusion, bold: np.ndarray of shape (N, H, W)
        cfg: dict from default_config(), optionally modified.

    Returns:
        (dixon_out, t1map_out, diffusion_out, bold_out): np.ndarray each of shape
            (N * num_aug + (N if include_original else 0), H, W)
            Originals are placed first if included.
    """
    if cfg is None:
        cfg = default_config()
    assert dixon.shape == t1map.shape == diffusion.shape == bold.shape, \
        f"Shape mismatch: {dixon.shape}, {t1map.shape}, {diffusion.shape}, {bold.shape}"

    n, h, w = dixon.shape
    total = n * cfg['num_aug'] + (n if cfg.get('include_original', True) else 0)

    dixon_out = np.empty((total, h, w), dtype=np.float32)
    t1map_out = np.empty((total, h, w), dtype=np.float32)
    diffusion_out = np.empty((total, h, w), dtype=np.float32)
    bold_out = np.empty((total, h, w), dtype=np.float32)

    rng_master = np.random.default_rng(cfg.get('seed', 42))

    idx_out = 0
    # 0) Optionally copy originals first (robust-normalized to [0,1] for consistency)
    if cfg.get('include_original', True):
        for i in range(n):
            dixon_out[idx_out] = robust_minmax(dixon[i])
            t1map_out[idx_out] = robust_minmax(t1map[i])
            diffusion_out[idx_out] = robust_minmax(diffusion[i])
            bold_out[idx_out] = robust_minmax(bold[i])
            idx_out += 1

    # 1) Augmented copies
    for copy_id in range(cfg['num_aug']):
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1, dtype=np.uint32))
        for i in range(n):
            d_aug, t_aug, diff_aug, b_aug = apply_subject_intensity_aug(
                [dixon[i], t1map[i], diffusion[i], bold[i]], rng, cfg
            )
            dixon_out[idx_out] = d_aug
            t1map_out[idx_out] = t_aug
            diffusion_out[idx_out] = diff_aug
            bold_out[idx_out] = b_aug
            idx_out += 1

    return dixon_out, t1map_out, diffusion_out, bold_out

def augment_and_save(cfg=None):
    """
    Disk IO wrapper: loads input arrays from cfg['save_root']/subset,
    applies augmentation (including originals if requested), and writes
    outputs to cfg['out_root']/subset.

    Args:
        cfg: dict from default_config(), optionally modified.

    Writes:
        DIXON.npy, T1_mapping_fl2d.npy, Diffusion.npy, BOLD.npy to out dir.
    """
    if cfg is None:
        cfg = default_config()

    in_dir = os.path.join(cfg['save_root'], cfg.get('subset', 'train'))
    out_dir = os.path.join(cfg['out_root'], cfg.get('subset', 'train'))
    os.makedirs(out_dir, exist_ok=True)

    dixon = np.load(os.path.join(in_dir, "DIXON.npy"))
    t1map = np.load(os.path.join(in_dir, "T1_mapping_fl2d.npy"))
    diffusion = np.load(os.path.join(in_dir, "Diffusion.npy"))
    bold = np.load(os.path.join(in_dir, "BOLD.npy"))

    dixon_out, t1map_out, diffusion_out, bold_out = augment_in_memory(
        dixon, t1map, diffusion, bold, cfg
    )

    np.save(os.path.join(out_dir, "DIXON.npy"), dixon_out)
    np.save(os.path.join(out_dir, "T1_mapping_fl2d.npy"), t1map_out)
    np.save(os.path.join(out_dir, "Diffusion.npy"), diffusion_out)
    np.save(os.path.join(out_dir, "BOLD.npy"), bold_out)

    return {
        "out_dir": out_dir,
        "shapes": {
            "DIXON": dixon_out.shape,
            "T1_mapping_fl2d": t1map_out.shape,
            "Diffusion": diffusion_out.shape,
            "BOLD": bold_out.shape,
        }
    }

# ----------------------- CLI -----------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_root", type=str, default="data/my_data")
    ap.add_argument("--out_root", type=str, default="data/my_data_aug")
    ap.add_argument("--subset", type=str, default="train")
    ap.add_argument("--num_aug", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include_original", type=int, default=1, help="1 to include originals, 0 to exclude")
    # Augmentation probabilities and ranges
    ap.add_argument("--p_scale", type=float, default=0.7)
    ap.add_argument("--scale_range", type=float, default=0.10)
    ap.add_argument("--p_shift", type=float, default=0.7)
    ap.add_argument("--shift_range", type=float, default=0.10)
    ap.add_argument("--p_gamma", type=float, default=0.5)
    ap.add_argument("--gamma_min", type=float, default=0.7)
    ap.add_argument("--gamma_max", type=float, default=1.3)
    ap.add_argument("--p_bias", type=float, default=0.4)
    ap.add_argument("--bias_magnitude", type=float, default=0.4)
    ap.add_argument("--bias_terms", type=int, default=3)
    ap.add_argument("--p_noise", type=float, default=0.4)
    ap.add_argument("--noise_min", type=float, default=0.01)
    ap.add_argument("--noise_max", type=float, default=0.04)
    return ap.parse_args()

def main_cli():
    args = parse_args()
    cfg = default_config()
    cfg.update({
        "save_root": args.save_root,
        "out_root": args.out_root,
        "subset": args.subset,
        "num_aug": args.num_aug,
        "seed": args.seed,
        "include_original": bool(args.include_original),
        "p_scale": args.p_scale,
        "scale_range": args.scale_range,
        "p_shift": args.p_shift,
        "shift_range": args.shift_range,
        "p_gamma": args.p_gamma,
        "gamma_min": args.gamma_min,
        "gamma_max": args.gamma_max,
        "p_bias": args.p_bias,
        "bias_magnitude": args.bias_magnitude,
        "bias_terms": args.bias_terms,
        "p_noise": args.p_noise,
        "noise_min": args.noise_min,
        "noise_max": args.noise_max,
    })
    info = augment_and_save(cfg)
    print("Saved to:", info["out_dir"])
    print("Output shapes:", info["shapes"])

if __name__ == "__main__":
    main_cli()
