#!/usr/bin/env python3
import os
import math
import time
import argparse
import json
import random
import numpy as np
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

from registration.models import RegNet2D 
from registration.losses import ncc_loss, grad_smooth_loss, dice_loss
from dataset.dataset_dixon import CreateDatasetSynthesis_single_with_masks 

# -----------------------
# Utilities
# -----------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_ddp(args):
    args.distributed = False
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ.get("LOCAL_RANK", 0))
        args.distributed = True
    else:
        # single GPU fallback
        args.rank = 0
        args.world_size = 1
        args.gpu = 0
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend="nccl", init_method="env://")
    return args

def cleanup_ddp(args):
    if args.distributed:
        dist.destroy_process_group()

def is_main(args):
    return (not args.distributed) or (args.rank == 0)

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path

# -----------------------
# Metrics
# -----------------------
def tensor_to_img(t: torch.Tensor) -> torch.Tensor:
    # t: (B,1,H,W), already normalized-ish; we’ll use as-is for PSNR/SSIM.
    return t

def psnr_torch(x, y, data_range=None):
    # x,y: (B,1,H,W)
    if data_range is None:
        # estimate max-min per batch
        mx = torch.max(torch.stack([x.max(dim=-1)[0].max(dim=-1)[0],
                                    y.max(dim=-1)[0].max(dim=-1)[0]], dim=0)[0], dim=1)[0]
        mn = torch.min(torch.stack([x.min(dim=-1)[0].min(dim=-1)[0],
                                    y.min(dim=-1)[0].min(dim=-1)[0]], dim=0)[0], dim=1)[0]
        data_range = (mx - mn + 1e-8).view(-1, 1, 1, 1)
    mse = torch.mean((x - y) ** 2, dim=[1,2,3]) + 1e-12
    psnr = 20 * torch.log10(data_range.view(-1) / torch.sqrt(mse))
    return psnr.mean()

def ssim_torch(x, y, window=11, K1=0.01, K2=0.03, data_range=1.0):
    # x,y: (B,1,H,W) in float; simplified SSIM with Gaussian window
    device = x.device
    # create gaussian 1D
    def gauss(w, sigma):
        coords = torch.arange(w, dtype=torch.float32, device=device) - w//2
        g = torch.exp(-(coords**2) / (2*sigma**2))
        g = g / g.sum()
        return g
    # 2D window
    sigma = 1.5
    g1d = gauss(window, sigma)
    w2d = (g1d[:, None] @ g1d[None, :]).unsqueeze(0).unsqueeze(0)  # (1,1,w,w)
    pad = window // 2

    mu_x = F.conv2d(x, w2d, padding=pad, groups=1)
    mu_y = F.conv2d(y, w2d, padding=pad, groups=1)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, w2d, padding=pad, groups=1) - mu_x2
    sigma_y2 = F.conv2d(y * y, w2d, padding=pad, groups=1) - mu_y2
    sigma_xy = F.conv2d(x * y, w2d, padding=pad, groups=1) - mu_xy

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + 1e-12)
    return ssim_map.mean()

def dice_coeff_torch(pred_mask, gt_mask, eps=1e-6):
    # expects {0,1} binary masks (B,1,H,W)
    inter = (pred_mask * gt_mask).sum(dim=[1,2,3])
    den = pred_mask.sum(dim=[1,2,3]) + gt_mask.sum(dim=[1,2,3]) + eps
    return (2.0 * inter / den).mean()

# -----------------------
# Normalization helpers
# -----------------------
def normalize_slice(x, clip_percentiles=(1, 99)):
    # x: float tensor (B,1,H,W) or (1,H,W)
    # percentile clip per-sample, then z-score
    if x.dim() == 3:
        x = x.unsqueeze(0)
    B = x.shape[0]
    x_out = []
    for b in range(B):
        xb = x[b:b+1]
        v = xb.flatten()
        p1 = torch.quantile(v, clip_percentiles[0] / 100.0)
        p99 = torch.quantile(v, clip_percentiles[1] / 100.0)
        xb = torch.clamp(xb, p1, p99)
        mean = xb.mean()
        std = xb.std() + 1e-8
        xb = (xb - mean) / std
        x_out.append(xb)
    return torch.cat(x_out, dim=0)

# -----------------------
# Saving fields & images
# -----------------------
def save_warp_and_outputs(save_dir: Path, idx_base: int, disp_norm, disp_pix, warped_img, warped_mask):
    """
    disp_norm: (B,2,H,W) normalized (-1..1 grid space)
    disp_pix : (B,2,H,W) pixel displacement (dy, dx)
    warped_img/mask: (B,1,H,W)
    """
    B = disp_norm.shape[0]
    for i in range(B):
        idx = idx_base + i
        np.save(save_dir / f"disp_norm_{idx:06d}.npy", disp_norm[i].detach().cpu().numpy())
        np.save(save_dir / f"disp_pix_{idx:06d}.npy",  disp_pix[i].detach().cpu().numpy())
        np.save(save_dir / f"warped_img_{idx:06d}.npy", warped_img[i].detach().cpu().numpy())
        if warped_mask is not None:
            np.save(save_dir / f"warped_mask_{idx:06d}.npy", warped_mask[i].detach().cpu().numpy())

def de_normalize_displacement(disp_norm, H, W):
    """
    Convert normalized disp (B,2,H,W) to pixel displacements (dy, dx).
    Normalization in grid_sample is [-1,1] across width/height.
    """
    vx = disp_norm[:, 1] * ((W - 1) / 2.0)
    vy = disp_norm[:, 0] * ((H - 1) / 2.0)
    return torch.stack([vy, vx], dim=1)  # (B,2,H,W) in pixels, (dy, dx)

# -----------------------
# Validation
# -----------------------
@torch.no_grad()
def validate(model, loader, device, save_outputs=False, save_dir: Path=None):
    model.eval()
    ssim_list, psnr_list, dice_list = [], [], []
    idx_base = 0

    for batch in loader:
        moving, fixed, mmask, fmask = batch
        # shapes: (B,H,W)
        moving = moving.float().unsqueeze(1).to(device)
        fixed  = fixed.float().unsqueeze(1).to(device)
        mmask  = mmask.float().unsqueeze(1).to(device)
        fmask  = fmask.float().unsqueeze(1).to(device)

        # normalize intensities per-slice
        moving_n = normalize_slice(moving)
        fixed_n  = normalize_slice(fixed)

        out = model(moving_n, fixed_n, moving_mask=mmask)
        warped = out['warped_image']
        disp_norm = out['disp_norm']  # (B,2,H,W)
        warped_mask = out.get('warped_mask', None)
        if warped_mask is not None:
            warped_mask = warped_mask.round().clamp(0,1)

        # Metrics
        ssim_val = ssim_torch(fixed_n, warped, data_range=2.0)  # z-scored ~ [-?, ?]; safe to use 2
        psnr_val = psnr_torch(fixed_n, warped, data_range=torch.tensor([2.0], device=device))
        ssim_list.append(ssim_val.item()); psnr_list.append(psnr_val.item())
        if warped_mask is not None:
            dice_val = dice_coeff_torch(warped_mask, fmask)
            dice_list.append(dice_val.item())

        if save_outputs and save_dir is not None:
            B, _, H, W = disp_norm.shape
            disp_pix = de_normalize_displacement(disp_norm, H, W)
            save_warp_and_outputs(save_dir, idx_base, disp_norm, disp_pix, warped, warped_mask)
            idx_base += B

    out = {
        "ssim": float(np.mean(ssim_list)) if ssim_list else 0.0,
        "psnr": float(np.mean(psnr_list)) if psnr_list else 0.0,
        "dice": float(np.mean(dice_list)) if dice_list else 0.0,
        "n_batches": len(ssim_list)
    }
    return out

# -----------------------
# Training
# -----------------------
def train_one_epoch(model, loader, optimizer, scaler, device, lambda_smooth=0.02, lambda_dice=0.2):
    model.train()
    running = {"loss": 0.0, "sim": 0.0, "smooth": 0.0, "dice": 0.0}
    n = 0

    for batch in loader:
        moving, fixed, mmask, fmask = batch
        moving = moving.float().unsqueeze(1).to(device)
        fixed  = fixed.float().unsqueeze(1).to(device)
        mmask  = mmask.float().unsqueeze(1).to(device)
        fmask  = fmask.float().unsqueeze(1).to(device)

        moving_n = normalize_slice(moving)
        fixed_n  = normalize_slice(fixed)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=True):
            out = model(moving_n, fixed_n, moving_mask=mmask)
            warped = out['warped_image']
            v = out['v']
            loss_sim = ncc_loss(warped, fixed_n)
            loss_smooth = grad_smooth_loss(v) * lambda_smooth

            loss = loss_sim + loss_smooth
            if 'warped_mask' in out:
                wmask = out['warped_mask']
                loss_dice = dice_loss(wmask, fmask) * lambda_dice
                loss = loss + loss_dice
            else:
                loss_dice = torch.zeros(1, device=device)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running["loss"] += loss.item()
        running["sim"]  += loss_sim.item()
        running["smooth"] += loss_smooth.item()
        running["dice"] += loss_dice.item()
        n += 1

    for k in running:
        running[k] /= max(n, 1)
    return running

def save_checkpoint(state, is_best, ckpt_dir: Path, filename="last.pth"):
    ensure_dir(ckpt_dir)
    torch.save(state, ckpt_dir / filename)
    if is_best:
        torch.save(state, ckpt_dir / "best.pth")

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Root data path, e.g., data/my_data2")
    parser.add_argument("--contrast1", type=str, required=True, help="Moving contrast, e.g., T1_mapping_fl2d")
    parser.add_argument("--contrast2", type=str, required=True, help="Fixed/target contrast, e.g., DIXON")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--n_steps", type=int, default=5, help="scaling-and-squaring steps for SVF integration")
    parser.add_argument("--smooth_w", type=float, default=0.02)
    parser.add_argument("--dice_w", type=float, default=0.2)
    parser.add_argument("--save_dir", type=str, default="runs/exp1")
    parser.add_argument("--save_val_outputs", action="store_true", help="Save displacements & warped images/masks for VAL each epoch")
    parser.add_argument("--save_test_outputs", action="store_true", help="Also run on TEST at the end and save outputs")
    parser.add_argument("--early_stop_patience", type=int, default=30)

    # DDP provided from launcher or single process
    args = parser.parse_args()
    args = setup_ddp(args)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if is_main(args):
        print("Args:", json.dumps(vars(args), indent=2))
    set_seed(args.seed + args.rank)

    # ---- Data ----
    dataset = CreateDatasetSynthesis_single_with_masks(
        phase="train", input_path=args.input_path, contrast=args.contrast1, target_contrast=args.contrast2
    )
    dataset_val = CreateDatasetSynthesis_single_with_masks(
        phase="val", input_path=args.input_path, contrast=args.contrast1, target_contrast=args.contrast2
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True, drop_last=True)
        val_sampler   = torch.utils.data.distributed.DistributedSampler(dataset_val, num_replicas=args.world_size, rank=args.rank, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=4, pin_memory=True, sampler=train_sampler, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, sampler=val_sampler, drop_last=False
    )

    # ---- Model ----
    model = RegNet2D(in_ch=2, n_steps=args.n_steps).to(device)
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=False)

    # ---- Optimizer & AMP ----
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # ---- Logging / Checkpoints ----
    save_root = ensure_dir(Path(args.save_dir))
    best_val = -1e9
    epochs_no_improve = 0

    # ---- Training loop ----
    for epoch in range(1, args.num_epoch + 1):
        if args.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        t0 = time.time()
        train_log = train_one_epoch(model, train_loader, optimizer, scaler, device,
                                    lambda_smooth=args.smooth_w, lambda_dice=args.dice_w)

        # validate (on rank 0 only collect logs)
        if args.distributed:
            dist.barrier()
        val_metrics = validate(model.module if isinstance(model, DDP) else model, val_loader, device,
                               save_outputs=args.save_val_outputs and is_main(args),
                               save_dir=ensure_dir(save_root / "val_outputs" / f"epoch_{epoch:04d}") if (args.save_val_outputs and is_main(args)) else None)

        if is_main(args):
            elapsed = time.time() - t0
            print(f"[Epoch {epoch:03d}] {elapsed:.1f}s | "
                  f"train loss {train_log['loss']:.4f} (sim {train_log['sim']:.4f}, smooth {train_log['smooth']:.4f}, dice {train_log['dice']:.4f}) | "
                  f"val SSIM {val_metrics['ssim']:.4f} PSNR {val_metrics['psnr']:.2f} Dice {val_metrics['dice']:.4f}")

            # use SSIM + Dice as a composite score to select best
            composite = val_metrics['ssim'] + val_metrics['dice']
            is_best = composite > best_val
            if is_best:
                best_val = composite
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            state = {
                "epoch": epoch,
                "model": (model.module if isinstance(model, DDP) else model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "best_val": best_val,
                "args": vars(args),
            }
            save_checkpoint(state, is_best, save_root)

            if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {epochs_no_improve} epochs).")
                break

    # ---- Optional TEST run & export ----
    if args.save_test_outputs and is_main(args):
        dataset_test = CreateDatasetSynthesis_single_with_masks(
            phase="test", input_path=args.input_path, contrast=args.contrast1, target_contrast=args.contrast2
        )
        test_loader = torch.utils.data.DataLoader(
            dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False
        )

        # Load best checkpoint
        best_ckpt = torch.load(save_root / "best.pth", map_location=device)
        (model.module if isinstance(model, DDP) else model).load_state_dict(best_ckpt["model"])
        out_dir = ensure_dir(save_root / "test_outputs")
        test_metrics = validate(model.module if isinstance(model, DDP) else model, test_loader, device,
                                save_outputs=True, save_dir=out_dir)
        print(f"[TEST] SSIM {test_metrics['ssim']:.4f} PSNR {test_metrics['psnr']:.2f} Dice {test_metrics['dice']:.4f}")
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)

    cleanup_ddp(args)

if __name__ == "__main__":
    main()
