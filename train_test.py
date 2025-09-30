import sys
path_to_pip_installs = "/tmp/test_env"
if path_to_pip_installs not in sys.path:
    sys.path.insert(0, path_to_pip_installs)

import os
import shutil
import socket
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from dotenv import load_dotenv
from skimage.metrics import peak_signal_noise_ratio as psnr
import wandb

from torch.multiprocessing import Process

from backbones.dense_layer import conv2d
from backbones.registration_network import RegistrationUNet
from dataset.dataset_dixon import CreateDatasetSynthesis_with_masks
from train_utils import (
    parse_arguments,
    copy_source,
    broadcast_params,
    get_time_schedule,
    _psnr_torch,
    _wandb_log,
    q_sample_pairs,
    sample_posterior,
    sample_from_model,
    Diffusion_Coefficients,
    Posterior_Coefficients,
    dice_loss,
    smoothness_loss
)

# Refactored Helper Function for Warping
def create_warp_grid(flow, base_grid_xy):
    """
    Creates a sampling grid for F.grid_sample from a flow field and a base coordinate grid.
    
    Args:
        flow (torch.Tensor): The optical flow field, shape [B, 2, H, W].
        base_grid_xy (torch.Tensor): The base coordinate grid, shape [B, 2, H, W].
        
    Returns:
        torch.Tensor: A grid suitable for F.grid_sample, shape [B, H, W, 2].
    """
    B, _, H, W = flow.shape
    
    # Add flow offsets to the base coordinates
    v_grid = base_grid_xy + flow

    # Scale grid to [-1, 1] for grid_sample
    v_grid_scaled_x = 2.0 * v_grid[:, 0, :, :] / max(W - 1, 1) - 1.0
    v_grid_scaled_y = 2.0 * v_grid[:, 1, :, :] / max(H - 1, 1) - 1.0
    v_grid_scaled = torch.stack([v_grid_scaled_x, v_grid_scaled_y], dim=1)

    # Permute to [B, H, W, 2] for grid_sample
    v_grid_final = v_grid_scaled.permute(0, 2, 3, 1)
    return v_grid_final

# -----------------------------------------------------------------------------------------
# %% ============================ DISCRIMINATOR TRAINING STEP ============================
# -----------------------------------------------------------------------------------------
def train_discriminator_step(models, optimizers, data, coeff, args, device, global_step):
    """Performs a single training step for the discriminator."""
    disc_diffusive_2 = models['disc_diffusive_2']
    gen_diffusive_1 = models['gen_diffusive_1']
    gen_diffusive_2 = models['gen_diffusive_2']
    pos_coeff = models['pos_coeff']
    optimizer_disc_diffusive_2 = optimizers['disc_diffusive_2']

    for p in disc_diffusive_2.parameters():
        p.requires_grad = True
    disc_diffusive_2.zero_grad()

    # Unpack data
    cond_data1, cond_data2, cond_data3, real_data = data['images']
    
    # Sample time and diffuse real data
    t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
    x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
    x_t.requires_grad = True

    # Train with real samples
    D_real, _ = disc_diffusive_2(x_t, t, x_tp1.detach())
    errD_real = F.softplus(-D_real).mean()
    errD_real.backward(retain_graph=True)

    # R1 regularization
    if args.lazy_reg is None or global_step % args.lazy_reg == 0:
        grad_real = torch.autograd.grad(outputs=D_real.sum(), inputs=x_t, create_graph=True)[0]
        grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
        grad_penalty = args.r1_gamma / 2 * grad_penalty
        grad_penalty.backward()

    # Train with fake samples
    with torch.no_grad():
        latent_z = torch.randn(args.batch_size, args.nz, device=device)
        x0_pred_g1 = gen_diffusive_1(x_tp1.detach(), cond_data1, cond_data2, cond_data3, t, latent_z)
        x0_pred_g2 = gen_diffusive_2(x_tp1.detach(), cond_data1, cond_data2, cond_data3, t, latent_z, x0_pred_g1[:, [0], :])
        x_pos_sample_g1 = sample_posterior(pos_coeff, x0_pred_g1[:, [0], :], x_tp1, t)
        x_pos_sample_g2 = sample_posterior(pos_coeff, x0_pred_g2[:, [0], :], x_tp1, t)

    D_fake_g1, _ = disc_diffusive_2(x_pos_sample_g1, t, x_tp1.detach())
    D_fake_g2, _ = disc_diffusive_2(x_pos_sample_g2, t, x_tp1.detach())
    
    errD_fake_g1 = F.softplus(D_fake_g1).mean()
    errD_fake_g2 = F.softplus(D_fake_g2).mean()
    
    errD_fake = errD_fake_g1 + errD_fake_g2
    errD_fake.backward()
    
    optimizer_disc_diffusive_2.step()
    
    losses = {
        "loss/D/real": errD_real.item(),
        "loss/D/fake_g1": errD_fake_g1.item(),
        "loss/D/fake_g2": errD_fake_g2.item(),
        "loss/D/total": errD_real.item() + errD_fake.item(),
        "lr/D": optimizer_disc_diffusive_2.param_groups[0]["lr"],
    }
    return losses

# -----------------------------------------------------------------------------------------
# %% ======================= GENERATOR & REGISTRATION TRAINING STEP ======================
# -----------------------------------------------------------------------------------------
def train_generator_and_registration_step(models, optimizers, data, coeff, args, device, base_grid_xy):
    """Performs a single training step for the generators and the registration network."""
    gen_diffusive_1 = models['gen_diffusive_1']
    gen_diffusive_2 = models['gen_diffusive_2']
    disc_diffusive_2 = models['disc_diffusive_2']
    reg_net = models['reg_net']
    att_conv = models['att_conv']
    pos_coeff = models['pos_coeff']
    
    optimizer_gen_diffusive_1 = optimizers['gen_diffusive_1']
    optimizer_gen_diffusive_2 = optimizers['gen_diffusive_2']
    optimizer_reg = optimizers['reg']
    optimizer_att = optimizers['att']
    
    # Set discriminator to not require gradients
    for p in disc_diffusive_2.parameters():
        p.requires_grad = False
        
    gen_diffusive_1.zero_grad()
    gen_diffusive_2.zero_grad()
    reg_net.zero_grad()
    att_conv.zero_grad()

    # Unpack data
    cond_data1, cond_data2, cond_data3, real_data = data['images']
    mask1, mask2, mask3, target_mask = data['masks']

    # === 1. REGISTRATION STEP ===
    reg_input = torch.cat([cond_data1, cond_data2, cond_data3, target_mask], dim=1)
    flow1, flow2, flow3 = reg_net(reg_input)

    # Create warping grids and warp inputs
    grid1 = create_warp_grid(flow1, base_grid_xy)
    grid2 = create_warp_grid(flow2, base_grid_xy)
    grid3 = create_warp_grid(flow3, base_grid_xy)
    
    cond1_warped = F.grid_sample(cond_data1, grid1, mode='bilinear', padding_mode='border', align_corners=True)
    cond2_warped = F.grid_sample(cond_data2, grid2, mode='bilinear', padding_mode='border', align_corners=True)
    cond3_warped = F.grid_sample(cond_data3, grid3, mode='bilinear', padding_mode='border', align_corners=True)
    
    mask1_warped = F.grid_sample(mask1, grid1, mode='nearest', align_corners=True)
    mask2_warped = F.grid_sample(mask2, grid2, mode='nearest', align_corners=True)
    mask3_warped = F.grid_sample(mask3, grid3, mode='nearest', align_corners=True)

    # === 2. DIFFUSION STEP ===
    t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
    x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
    latent_z = torch.randn(args.batch_size, args.nz, device=device)
    
    x0_pred_g1 = gen_diffusive_1(x_tp1.detach(), cond1_warped, cond2_warped, cond3_warped, t, latent_z)
    x0_pred_g2 = gen_diffusive_2(x_tp1.detach(), cond1_warped, cond2_warped, cond3_warped, t, latent_z, x0_pred_g1[:, [0], :])

    x_pos_sample_g1 = sample_posterior(pos_coeff, x0_pred_g1[:, [0], :], x_tp1, t)
    x_pos_sample_g2 = sample_posterior(pos_coeff, x0_pred_g2[:, [0], :], x_tp1, t)

    # === 3. LOSS CALCULATION ===
    # Adversarial losses
    output_g1, att_feat_g1 = disc_diffusive_2(x_pos_sample_g1, t, x_tp1.detach())
    output_g2, att_feat_g2 = disc_diffusive_2(x_pos_sample_g2, t, x_tp1.detach())
    errG_adv_g1 = F.softplus(-output_g1).mean()
    errG_adv_g2 = F.softplus(-output_g2).mean()
    errG_adv = errG_adv_g1 + errG_adv_g2

    # L1 losses
    errG_L1_g1 = F.l1_loss(x0_pred_g1[:, [0], :], real_data)
    errG_L1_g2 = F.l1_loss(x0_pred_g2[:, [0], :], real_data)
    errG_L1 = errG_L1_g1 + errG_L1_g2
    
    # Attention mask losses
    critic_criterian = nn.BCEWithLogitsLoss(reduction='none')
    att_map_g1 = torch.sigmoid(F.interpolate(att_conv(att_feat_g1), size=(256, 256), mode='bilinear', align_corners=False))
    att_map_g2 = torch.sigmoid(F.interpolate(att_conv(att_feat_g2), size=(256, 256), mode='bilinear', align_corners=False))
    mask_loss_1 = (att_map_g2 * critic_criterian(x_pos_sample_g1, torch.sigmoid(x_pos_sample_g2))).mean()
    mask_loss_2 = (att_map_g1 * critic_criterian(x_pos_sample_g2, torch.sigmoid(x_pos_sample_g1))).mean()
    mask_loss = mask_loss_1 + mask_loss_2

    # Registration losses
    loss_reg_dice = (dice_loss(mask1_warped, target_mask) + dice_loss(mask2_warped, target_mask) + dice_loss(mask3_warped, target_mask)) / 3.0
    loss_reg_smooth = (smoothness_loss(flow1) + smoothness_loss(flow2) + smoothness_loss(flow3)) / 3.0
    
    # Total loss
    errG_diffusion = errG_adv + (args.lambda_l1_loss * errG_L1) + (args.lambda_mask_loss * mask_loss)
    errG_total = errG_diffusion + (args.lambda_dice * loss_reg_dice) + (args.lambda_smooth * loss_reg_smooth)
    
    # === 4. BACKPROPAGATION & OPTIMIZER STEP ===
    errG_total.backward()
    optimizer_gen_diffusive_1.step()
    optimizer_gen_diffusive_2.step()
    optimizer_reg.step()
    optimizer_att.step()
    
    # --- Compute training metrics for logging ---
    to_range_0_1 = lambda x: (x + 1.) / 2.
    pred1_01 = torch.clamp(to_range_0_1(x0_pred_g1[:, [0], :].detach()), 0, 1)
    pred2_01 = torch.clamp(to_range_0_1(x0_pred_g2[:, [0], :].detach()), 0, 1)
    gt_01 = torch.clamp(to_range_0_1(real_data.detach()), 0, 1)

    psnr_g1 = _psnr_torch(pred1_01, gt_01)
    psnr_g2 = _psnr_torch(pred2_01, gt_01)
    
    results = {
        "losses": {
            "loss/G/adv_g1": errG_adv_g1.item(), "loss/G/adv_g2": errG_adv_g2.item(),
            "loss/G/adv_total": errG_adv.item(), "loss/G/L1_g1": errG_L1_g1.item(),
            "loss/G/L1_g2": errG_L1_g2.item(), "loss/G/L1_total": errG_L1.item(),
            "loss/G/mask": mask_loss.item(), "loss/G/diffusion_total": errG_diffusion.item(),
            "loss/Reg/dice": loss_reg_dice.item(), "loss/Reg/smooth": loss_reg_smooth.item(),
            "loss/G/total": errG_total.item(),
            "lr/G1": optimizer_gen_diffusive_1.param_groups[0]["lr"],
            "lr/G2": optimizer_gen_diffusive_2.param_groups[0]["lr"],
            "lr/Reg": optimizer_reg.param_groups[0]["lr"],
        },
        "metrics": {
            "metric/train/psnr_g1": psnr_g1, "metric/train/psnr_g2": psnr_g2,
        },
        "predictions": {
            "pred_g1": x0_pred_g1, "pred_g2": x0_pred_g2
        }
    }
    return results

# -----------------------------------------------------------------------------------------
# %% =============================== VALIDATION STEP =====================================
# -----------------------------------------------------------------------------------------
def validate_epoch(models, data_loader, pos_coeff, T, args, device, base_grid_xy, epoch):
    """Performs a full validation loop for one epoch."""
    gen_diffusive_1 = models['gen_diffusive_1']
    gen_diffusive_2 = models['gen_diffusive_2']
    reg_net = models['reg_net']
    
    # Arrays to store metrics for this epoch
    val_l1, val_psnr, val_dice, val_smooth = [], [], [], []
    val_dice_comps, val_smooth_comps = {1:[], 2:[], 3:[]}, {1:[], 2:[], 3:[]}
    
    to_range_0_1 = lambda x: (x + 1.) / 2.

    with torch.no_grad():
        for x1, x2, x3, x4, m1, m2, m3, m4 in data_loader:
            cond1, cond2, cond3, real_data = x1.to(device), x2.to(device), x3.to(device), x4.to(device)
            mask1, mask2, mask3, target_mask = m1.to(device), m2.to(device), m3.to(device), m4.to(device)

            # === 1. REGISTRATION ===
            reg_input = torch.cat([cond1, cond2, cond3, target_mask], dim=1)
            flow1, flow2, flow3 = reg_net(reg_input)
            
            grid1 = create_warp_grid(flow1, base_grid_xy)
            grid2 = create_warp_grid(flow2, base_grid_xy)
            grid3 = create_warp_grid(flow3, base_grid_xy)
            
            cond1_warped = F.grid_sample(cond1, grid1, mode='bilinear', padding_mode='border', align_corners=True)
            cond2_warped = F.grid_sample(cond2, grid2, mode='bilinear', padding_mode='border', align_corners=True)
            cond3_warped = F.grid_sample(cond3, grid3, mode='bilinear', padding_mode='border', align_corners=True)
            
            mask1_warped = F.grid_sample(mask1, grid1, mode='nearest', align_corners=True)
            mask2_warped = F.grid_sample(mask2, grid2, mode='nearest', align_corners=True)
            mask3_warped = F.grid_sample(mask3, grid3, mode='nearest', align_corners=True)

            # === 2. REGISTRATION METRICS ===
            dce1, dce2, dce3 = dice_loss(mask1_warped, target_mask), dice_loss(mask2_warped, target_mask), dice_loss(mask3_warped, target_mask)
            val_dice.append(((dce1 + dce2 + dce3) / 3.0).item())
            val_dice_comps[1].append(dce1.item()); val_dice_comps[2].append(dce2.item()); val_dice_comps[3].append(dce3.item())

            sm1, sm2, sm3 = smoothness_loss(flow1), smoothness_loss(flow2), smoothness_loss(flow3)
            val_smooth.append(((sm1 + sm2 + sm3) / 3.0).item())
            val_smooth_comps[1].append(sm1.item()); val_smooth_comps[2].append(sm2.item()); val_smooth_comps[3].append(sm3.item())
            
            # === 3. DIFFUSION SAMPLING ===
            x_t = torch.randn_like(real_data)
            fake_sample = sample_from_model(pos_coeff, gen_diffusive_1, cond1_warped, gen_diffusive_2, cond2_warped, cond3_warped, args.num_timesteps, x_t, T, args)
            
            # === 4. SYNTHESIS METRICS ===
            fake_sample_01 = to_range_0_1(fake_sample)
            real_data_01 = to_range_0_1(real_data)
            
            val_l1.append(F.l1_loss(fake_sample_01, real_data_01).item())
            
            # Use skimage psnr for consistency with original code
            fake_np = fake_sample_01.detach().cpu().numpy()
            real_np = real_data_01.detach().cpu().numpy()
            val_psnr.append(psnr(real_np, fake_np, data_range=real_np.max()))
    
    # Aggregate and return metrics
    metrics = {
        "metric/val/psnr_mean": float(np.nanmean(val_psnr)),
        "metric/val/l1_mean": float(np.nanmean(val_l1)),
        "metric/val/dice_mean": float(np.nanmean(val_dice)),
        "metric/val/smooth_mean": float(np.nanmean(val_smooth)),
        "metric/val/dice_mean1": float(np.nanmean(val_dice_comps[1])),
        "metric/val/dice_mean2": float(np.nanmean(val_dice_comps[2])),
        "metric/val/dice_mean3": float(np.nanmean(val_dice_comps[3])),
        "metric/val/smooth_mean1": float(np.nanmean(val_smooth_comps[1])),
        "metric/val/smooth_mean2": float(np.nanmean(val_smooth_comps[2])),
        "metric/val/smooth_mean3": float(np.nanmean(val_smooth_comps[3])),
        "epoch": epoch,
    }
    return metrics

# -----------------------------------------------------------------------------------------
# %% =============================== MAIN TRAINING FUNCTION ==============================
# -----------------------------------------------------------------------------------------
def train_mudiff(rank, gpu, args):
    from backbones.discriminator import Discriminator_large
    from backbones.ncsnpp_generator_adagn_feat import NCSNpp, NCSNpp_adaptive
    from utils.EMA import EMA

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device(f'cuda:{gpu}')
    is_master = (rank == 0)

    # W&B init
    if is_master:
        load_dotenv()
        if os.getenv("WANDB_API_KEY", ""): wandb.login(key=os.getenv("WANDB_API_KEY"))
        else: os.environ["WANDB_MODE"] = os.environ.get("WANDB_MODE", "offline")
        wandb.init(project=os.environ.get("WANDB_PROJECT", "mudiff"), name=f"{args.exp}-{time.strftime('%Y%m%d-%H%M%S')}", config=vars(args))

    # DataLoaders
    dataset = CreateDatasetSynthesis_with_masks(phase="train", input_path=args.input_path)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=args.world_size, rank=rank)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    dataset_val = CreateDatasetSynthesis_with_masks(phase="val", input_path=args.input_path)
    val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val, num_replicas=args.world_size, rank=rank)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, sampler=val_sampler, drop_last=True)

    # Networks
    gen_diffusive_1 = NCSNpp(args).to(device)
    gen_diffusive_2 = NCSNpp_adaptive(args).to(device)
    disc_diffusive_2 = Discriminator_large(nc=2, ngf=args.ngf, t_emb_dim=args.t_emb_dim, act=nn.LeakyReLU(0.2)).to(device)
    reg_net = RegistrationUNet().to(device)
    att_conv = conv2d(64 * 8, 1, 1, padding=0).to(device)
    
    broadcast_params(gen_diffusive_1.parameters()); broadcast_params(gen_diffusive_2.parameters())
    broadcast_params(disc_diffusive_2.parameters()); broadcast_params(reg_net.parameters())

    # Optimizers
    optimizer_gen_diffusive_1 = optim.Adam(gen_diffusive_1.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    optimizer_gen_diffusive_2 = optim.Adam(gen_diffusive_2.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    optimizer_disc_diffusive_2 = optim.Adam(disc_diffusive_2.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    optimizer_reg = optim.Adam(reg_net.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    optimizer_att = optim.Adam(att_conv.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))

    if args.use_ema:
        optimizer_gen_diffusive_1 = EMA(optimizer_gen_diffusive_1, ema_decay=args.ema_decay)
        optimizer_gen_diffusive_2 = EMA(optimizer_gen_diffusive_2, ema_decay=args.ema_decay)

    # Schedulers
    scheduler_gen_diffusive_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gen_diffusive_1, args.num_epoch, eta_min=1e-5)
    scheduler_gen_diffusive_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gen_diffusive_2, args.num_epoch, eta_min=1e-5)
    scheduler_disc_diffusive_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_disc_diffusive_2, args.num_epoch, eta_min=1e-5)
    # NOTE: No scheduler for reg_net in original code, maintaining that behavior.

    # DDP
    gen_diffusive_1 = nn.parallel.DistributedDataParallel(gen_diffusive_1, device_ids=[gpu])
    gen_diffusive_2 = nn.parallel.DistributedDataParallel(gen_diffusive_2, device_ids=[gpu])
    disc_diffusive_2 = nn.parallel.DistributedDataParallel(disc_diffusive_2, device_ids=[gpu])
    reg_net = nn.parallel.DistributedDataParallel(reg_net, device_ids=[gpu])

    # Setup Paths and Diffusion Coefficients
    exp_path = os.path.join(args.output_path, args.exp)
    if is_master and not os.path.exists(exp_path):
        os.makedirs(exp_path); copy_source(__file__, exp_path)
        shutil.copytree('./backbones', os.path.join(exp_path, 'backbones'))

    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    T = get_time_schedule(args, device)

    # Resume from checkpoint
    global_step, epoch, init_epoch = 0, 0, 0
    if args.resume:
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        
        gen_diffusive_1.load_state_dict(checkpoint['gen_diffusive_1_dict'])
        optimizer_gen_diffusive_1.load_state_dict(checkpoint['optimizer_gen_diffusive_1'])
        scheduler_gen_diffusive_1.load_state_dict(checkpoint['scheduler_gen_diffusive_1'])
        
        gen_diffusive_2.load_state_dict(checkpoint['gen_diffusive_2_dict'])
        optimizer_gen_diffusive_2.load_state_dict(checkpoint['optimizer_gen_diffusive_2'])
        scheduler_gen_diffusive_2.load_state_dict(checkpoint['scheduler_gen_diffusive_2'])

        disc_diffusive_2.load_state_dict(checkpoint['disc_diffusive_2_dict'])
        optimizer_disc_diffusive_2.load_state_dict(checkpoint['optimizer_disc_diffusive_2'])
        scheduler_disc_diffusive_2.load_state_dict(checkpoint['scheduler_disc_diffusive_2'])
        
        # *** ADDED: Load Registration Network and its Optimizer ***
        reg_net.load_state_dict(checkpoint['reg_net_dict'])
        optimizer_reg.load_state_dict(checkpoint['optimizer_reg'])

        global_step = checkpoint['global_step']
        if is_master: print(f"=> loaded checkpoint (epoch {init_epoch})")

    # Prepare models and optimizers dictionaries for passing to functions
    models = {'gen_diffusive_1': gen_diffusive_1, 'gen_diffusive_2': gen_diffusive_2,
              'disc_diffusive_2': disc_diffusive_2, 'reg_net': reg_net,
              'att_conv': att_conv, 'pos_coeff': pos_coeff}
    optimizers = {'gen_diffusive_1': optimizer_gen_diffusive_1, 'gen_diffusive_2': optimizer_gen_diffusive_2,
                  'disc_diffusive_2': optimizer_disc_diffusive_2, 'reg': optimizer_reg, 'att': optimizer_att}

    # Pre-calculate base grid for warping (assuming constant image size)
    B, C, H, W = args.batch_size, 1, 256, 256 # Example size, adjust if dynamic
    mesh_x, mesh_y = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing='ij')
    base_grid_xy = torch.stack([mesh_x, mesh_y], dim=0).float().permute(0, 2, 1)
    base_grid_xy = base_grid_xy.unsqueeze(0).repeat(B, 1, 1, 1)

    # ============================= TRAINING LOOP =============================
    for epoch in range(init_epoch, args.num_epoch + 1):
        train_sampler.set_epoch(epoch)

        for iteration, (x1, x2, x3, x4, m1, m2, m3, m4) in enumerate(data_loader):
            # Move data to device
            images = [d.to(device, non_blocking=True) for d in [x1, x2, x3, x4]]
            masks = [d.to(device, non_blocking=True) for d in [m1, m2, m3, m4]]
            data = {'images': images, 'masks': masks}

            # --- Discriminator Step ---
            d_losses = train_discriminator_step(models, optimizers, data, coeff, args, device, global_step)

            # --- Generator and Registration Step ---
            g_reg_results = train_generator_and_registration_step(models, optimizers, data, coeff, args, device, base_grid_xy)
            g_reg_losses = g_reg_results['losses']
            g_reg_metrics = g_reg_results['metrics']
            preds = g_reg_results['predictions']

            # --- Logging ---
            if is_master:
                _wandb_log(d_losses, step=global_step)
                _wandb_log(g_reg_losses, step=global_step)
                _wandb_log(g_reg_metrics, step=global_step)

            global_step += 1

            if iteration % 100 == 0 and is_master:
                print(f"Epoch {epoch}, Iter {iteration}: G-Total Loss: {g_reg_losses['loss/G/total']:.4f}, D-Total Loss: {d_losses['loss/D/total']:.4f}")
                # Log image strip to W&B
                panel = torch.cat([torch.clamp((d + 1)/2, 0, 1) for d in [images[0][:1], images[1][:1], images[2][:1], preds['pred_g1'][:1,:1], preds['pred_g2'][:1,:1], images[3][:1]]], dim=-1)
                wandb.log({"train/strip": wandb.Image(panel.cpu(), caption="cond1|cond2|cond3|pred_g1|pred_g2|gt")}, step=global_step)

        # --- End of Epoch ---
        if not args.no_lr_decay:
            scheduler_gen_diffusive_1.step(); scheduler_gen_diffusive_2.step(); scheduler_disc_diffusive_2.step()
            
        # --- Validation ---
        val_metrics = validate_epoch(models, data_loader_val, pos_coeff, T, args, device, base_grid_xy, epoch)
        if is_master:
            _wandb_log(val_metrics, step=global_step)
            print(f"Epoch {epoch} Validation: PSNR={val_metrics['metric/val/psnr_mean']:.2f}, L1={val_metrics['metric/val/l1_mean']:.4f}, Dice={val_metrics['metric/val/dice_mean']:.4f}")

        # --- Saving Checkpoints and Samples (Master Only) ---
        if is_master:
            if epoch % args.save_content_every == 0 and args.save_content:
                print('Saving content...')
                content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                           'gen_diffusive_1_dict': gen_diffusive_1.state_dict(),
                           'optimizer_gen_diffusive_1': optimizer_gen_diffusive_1.state_dict(),
                           'scheduler_gen_diffusive_1': scheduler_gen_diffusive_1.state_dict(),
                           'gen_diffusive_2_dict': gen_diffusive_2.state_dict(),
                           'optimizer_gen_diffusive_2': optimizer_gen_diffusive_2.state_dict(),
                           'scheduler_gen_diffusive_2': scheduler_gen_diffusive_2.state_dict(),
                           'disc_diffusive_2_dict': disc_diffusive_2.state_dict(),
                           'optimizer_disc_diffusive_2': optimizer_disc_diffusive_2.state_dict(),
                           'scheduler_disc_diffusive_2': scheduler_disc_diffusive_2.state_dict(),
                           'reg_net_dict': reg_net.state_dict(),
                           'optimizer_reg': optimizer_reg.state_dict()
                           }
                torch.save(content, os.path.join(exp_path, 'content.pth'))
            # Other saving logic for samples etc. can go here
    
    if is_master: wandb.finish()


def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.port_num
    torch.cuda.set_device(args.local_rank)
    gpu = args.local_rank
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(rank, gpu, args)
    dist.barrier()
    dist.destroy_process_group()

# %%
if __name__ == '__main__':
    args, size = parse_arguments()
    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            p = Process(target=init_processes, args=(global_rank, global_size, train_mudiff, args))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        init_processes(0, 1, train_mudiff, args)