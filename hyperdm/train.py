import sys
path_to_pip_installs = "/tmp/test_env"
if path_to_pip_installs not in sys.path:
    sys.path.insert(0, path_to_pip_installs)

import os
import shutil
import socket
import time
import argparse

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from skimage.metrics import peak_signal_noise_ratio as psnr

from torch.multiprocessing import Process

# --- Imports from MU-Diff Project ---
from backbones.dense_layer import conv2d
from dataset.dataset_dixon import CreateDatasetSynthesis
from train_utils import (
    copy_source,
    broadcast_params,
    get_time_schedule,
    q_sample_pairs,
    sample_posterior,
    Diffusion_Coefficients,
    Posterior_Coefficients,
)
from backbones.discriminator import Discriminator_large
from backbones.ncsnpp_generator_adagn_feat import NCSNpp
from utils.EMA import EMA

# NEW: Import the simplified HyperDM wrapper
from hyperdm.hyperdm import HyperDM_MU_Diff_Single

# --- Helper function for validation sampling ---
def sample_from_hyper_model(hyperdm_model, pos_coeff, cond_data1, cond_data2, cond_data3, x_t, T, args):
    """
    A helper function to generate a sample from the HyperDM model for validation.
    It samples one network from the hyper-network and then generates one image.
    """
    with torch.no_grad():
        # DDP unwraps the model to access its methods
        func_net = hyperdm_model.module.sample_networks(device=x_t.device)
        
        # This is an adapted version of the original sample_from_model
        x = x_t
        for i in reversed(range(args.num_timesteps)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
            latent_z = torch.randn(x.size(0), args.nz, device=x.device)
            
            x0_predict_diff = func_net((x, cond_data1, cond_data2, cond_data3, t, latent_z))
            x_new = sample_posterior(pos_coeff, x0_predict_diff[:, [0], :], x, t)
            x = x_new.detach()
    return x

def train_hyperdm_mudiff_single(rank, gpu, args):
    """
    Simplified training function for a single-generator HyperDM model.
    """
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device('cuda:{}'.format(gpu))

    is_master = (rank == 0)

    batch_size = args.batch_size
    nz = args.nz
    dataset = CreateDatasetSynthesis(phase="train", input_path=args.input_path)
    dataset_val = CreateDatasetSynthesis(phase="val", input_path=args.input_path)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=args.world_size, rank=rank)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, sampler=train_sampler, drop_last=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val, num_replicas=args.world_size, rank=rank)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, sampler=val_sampler, drop_last=True)
    
    # Validation metric arrays
    val_l1_loss = np.zeros([1, args.num_epoch, len(data_loader_val)])
    val_psnr_values = np.zeros([1, args.num_epoch, len(data_loader_val)])

    if is_master: 
        print('train data size:' + str(len(data_loader)))
        print('val data size:' + str(len(data_loader_val)))
    
    primary_net = NCSNpp(args).to(device)
    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)

    hyperdm_model = HyperDM_MU_Diff_Single(
        primary_net=primary_net,
        hyper_net_dims=args.hyper_net_dims,
        diffusion_args=args,
        pos_coeff=pos_coeff
    ).to(device)
    if is_master: hyperdm_model.print_stats()

    disc = Discriminator_large(nc=2, ngf=args.ngf, t_emb_dim=args.t_emb_dim, act=nn.LeakyReLU(0.2)).to(device)
    
    broadcast_params(hyperdm_model.parameters())
    broadcast_params(disc.parameters())

    optimizer_disc = optim.Adam(disc.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    optimizer_hyperdm = optim.Adam(hyperdm_model.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))

    if args.use_ema:
        optimizer_hyperdm = EMA(optimizer_hyperdm, ema_decay=args.ema_decay)

    scheduler_hyperdm = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_hyperdm, args.num_epoch, eta_min=1e-5)
    scheduler_disc = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_disc, args.num_epoch, eta_min=1e-5)

    hyperdm_model = nn.parallel.DistributedDataParallel(hyperdm_model, device_ids=[gpu], find_unused_parameters=False)
    disc = nn.parallel.DistributedDataParallel(disc, device_ids=[gpu])
    
    exp_path = os.path.join(args.output_path, args.exp)
    if is_master and not os.path.exists(exp_path): os.makedirs(exp_path)
        
    global_step = 0
    init_epoch = 0
    if args.resume:
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        hyperdm_model.load_state_dict(checkpoint['hyperdm_model_dict'])
        optimizer_hyperdm.load_state_dict(checkpoint['optimizer_hyperdm'])
        scheduler_hyperdm.load_state_dict(checkpoint['scheduler_hyperdm'])
        disc.load_state_dict(checkpoint['disc_dict'])
        optimizer_disc.load_state_dict(checkpoint['optimizer_disc'])
        scheduler_disc.load_state_dict(checkpoint['scheduler_disc'])
        global_step = checkpoint['global_step']
        if is_master: print(f"=> loaded checkpoint (epoch {init_epoch})")

    for epoch in range(init_epoch, args.num_epoch + 1):
        train_sampler.set_epoch(epoch)
        for iteration, (x1, x2, x3, x4) in enumerate(data_loader):
            # === Train Discriminator ===
            for p in disc.parameters():
                p.requires_grad = True
            disc.zero_grad()

            cond_data1, cond_data2, cond_data3, real_data = [d.to(device, non_blocking=True) for d in (x1, x2, x3, x4)]
            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
            x_t.requires_grad = True
            
            D_real, _ = disc(x_t, t, x_tp1.detach())
            errD_real = F.softplus(-D_real).mean()
            errD_real.backward(retain_graph=True)
            
            if global_step % args.lazy_reg == 0:
                grad_real = torch.autograd.grad(outputs=D_real.sum(), inputs=x_t, create_graph=True)[0]
                grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                grad_penalty = args.r1_gamma / 2 * grad_penalty
                grad_penalty.backward()
            
            with torch.no_grad():
                x0_predict_diff = hyperdm_model(x_tp1.detach(), cond_data1, cond_data2, cond_data3, t)
                x_pos_sample = sample_posterior(pos_coeff, x0_predict_diff[:, [0], :], x_tp1, t)

            output, _ = disc(x_pos_sample, t, x_tp1.detach())
            errD_fake = F.softplus(output).mean()
            errD_fake.backward()
            
            optimizer_disc.step()

            # === Train Generator (Hyper-network) ===
            for p in disc.parameters():
                p.requires_grad = False
            hyperdm_model.zero_grad()
            
            x0_predict_diff = hyperdm_model(x_tp1.detach(), cond_data1, cond_data2, cond_data3, t)
            x_pos_sample = sample_posterior(pos_coeff, x0_predict_diff[:, [0], :], x_tp1, t)
            output, _ = disc(x_pos_sample, t, x_tp1.detach())
            
            errG_adv = F.softplus(-output).mean()
            errG_L1 = F.l1_loss(x0_predict_diff[:, [0], :], real_data)
            errG = errG_adv + (args.lambda_l1_loss * errG_L1)

            errG.backward()
            optimizer_hyperdm.step()
            
            if is_master and iteration % 100 == 0:
                print(f'Epoch {epoch}, Iter {iteration}, G-Total: {errG.item():.4f}, D-Total: {(errD_real + errD_fake).item():.4f}')
            
            global_step += 1

        if not args.no_lr_decay:
            scheduler_hyperdm.step()
            scheduler_disc.step()
        
        if is_master:
            # --- Image Saving (master only) ---
            if epoch % 1 == 0:
                torchvision.utils.save_image(x_pos_sample, os.path.join(exp_path, f'xpos_epoch_{epoch}.png'), normalize=True)
                
                # Generate a full sample for visualization
                with torch.no_grad():
                    # Use the last batch from the dataloader for consistency in visualization
                    x_t_val = torch.randn_like(real_data[:1])
                    fake_sample_val = sample_from_hyper_model(hyperdm_model, pos_coeff, cond_data1[:1], cond_data2[:1], cond_data3[:1], x_t_val, None, args)
                
                sample_panel = torch.cat((real_data[:1], fake_sample_val), axis=-1)
                torchvision.utils.save_image(sample_panel, os.path.join(exp_path, f'sample_discrete_epoch_{epoch}.png'), normalize=True)

            # --- Checkpointing (master only) ---
            if args.save_content and (epoch % args.save_content_every == 0):
                print('Saving content.')
                content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                            'hyperdm_model_dict': hyperdm_model.state_dict(),
                            'optimizer_hyperdm': optimizer_hyperdm.state_dict(),
                            'scheduler_hyperdm': scheduler_hyperdm.state_dict(),
                            'disc_dict': disc.state_dict(),
                            'optimizer_disc': optimizer_disc.state_dict(),
                            'scheduler_disc': scheduler_disc.state_dict()}
                torch.save(content, os.path.join(exp_path, 'content.pth'))

        # --- Validation Loop ---
        if is_master: # Only master process should run validation and save results
            hyperdm_model.eval()
            for val_iter, (x1_val, x2_val, x3_val, x4_val) in enumerate(data_loader_val):
                cond_data1_val = x1_val.to(device, non_blocking=True)
                cond_data2_val = x2_val.to(device, non_blocking=True)
                cond_data3_val = x3_val.to(device, non_blocking=True)
                real_data_val = x4_val.to(device, non_blocking=True)

                x_t = torch.randn_like(real_data_val)
                
                fake_sample_val = sample_from_hyper_model(hyperdm_model, pos_coeff, cond_data1_val, cond_data2_val, cond_data3_val, x_t, None, args)

                to_range_0_1 = lambda x: (x + 1.) / 2.
                fake_sample_val = to_range_0_1(fake_sample_val)
                real_data_val = to_range_0_1(real_data_val)
                
                fake_sample_val_np = fake_sample_val.detach().cpu().numpy()
                real_data_val_np = real_data_val.detach().cpu().numpy()
                
                val_l1_loss[0, epoch, val_iter] = np.abs(fake_sample_val_np - real_data_val_np).mean()
                val_psnr_values[0, epoch, val_iter] = psnr(real_data_val_np.squeeze(), fake_sample_val_np.squeeze(), data_range=1.0) # Data range is 1.0 after normalization

            val_psnr_mean = float(np.nanmean(val_psnr_values[0, epoch, :]))
            val_l1_mean = float(np.nanmean(val_l1_loss[0, epoch, :]))
            print(f"--- Epoch {epoch} Validation ---")
            print(f"  Avg PSNR: {val_psnr_mean:.4f}")
            print(f"  Avg L1:   {val_l1_mean:.4f}")
            
            np.save(f'{exp_path}/val_l1_loss.npy', val_l1_loss)
            np.save(f'{exp_path}/val_psnr_values.npy', val_psnr_values)
            hyperdm_model.train()


def parse_arguments_hyperdm_single():
    parser = argparse.ArgumentParser('mudiff-hyperdm-single parameters')
    # General arguments
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--exp', default='mudiff-hyperdm-single-exp')
    parser.add_argument('--input_path', default='/data/BRATS/')
    parser.add_argument('--output_path', default='results/')
    # Data arguments
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--num_channels', type=int, default=1)
    # Diffusion arguments
    parser.add_argument('--num_timesteps', type=int, default=4)
    parser.add_argument('--beta_min', type=float, default=0.1)
    parser.add_argument('--beta_max', type=float, default=20.)
    # Model arguments
    parser.add_argument('--centered', action='store_false', default=True)
    parser.add_argument('--use_geometric', action='store_true', default=False)
    parser.add_argument('--num_channels_dae', type=int, default=64)
    parser.add_argument('--n_mlp', type=int, default=3)
    parser.add_argument('--ch_mult', nargs='+', type=int, default=[1, 2, 4])
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--attn_resolutions', default=(16,))
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--resamp_with_conv', action='store_false', default=True)
    parser.add_argument('--conditional', action='store_false', default=True)
    parser.add_argument('--fir', action='store_false', default=True)
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1])
    parser.add_argument('--skip_rescale', action='store_false', default=True)
    parser.add_argument('--resblock_type', default='biggan')
    parser.add_argument('--progressive', type=str, default='none')
    parser.add_argument('--progressive_input', type=str, default='residual')
    parser.add_argument('--progressive_combine', type=str, default='sum')
    parser.add_argument('--embedding_type', type=str, default='positional')
    parser.add_argument('--fourier_scale', type=float, default=16.)
    parser.add_argument('--not_use_tanh', action='store_true', default=False)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--ngf', type=int, default=64)
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epoch', type=int, default=1200)
    parser.add_argument('--lr_g', type=float, default=1.6e-4)
    parser.add_argument('--lr_d', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--no_lr_decay', action='store_true', default=False)
    parser.add_argument('--use_ema', action='store_true', default=False)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--r1_gamma', type=float, default=0.05)
    parser.add_argument('--lazy_reg', type=int, default=10)
    parser.add_argument('--lambda_l1_loss', type=float, default=1.0)
    # Checkpointing arguments
    parser.add_argument('--save_content', action='store_true', default=True)
    parser.add_argument('--save_content_every', type=int, default=50)
    # DDP arguments
    parser.add_argument('--num_proc_node', type=int, default=1)
    parser.add_argument('--num_process_per_node', type=int, default=1)
    parser.add_argument('--node_rank', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--master_address', type=str, default='127.0.0.1')
    parser.add_argument('--port_num', type=str, default='6021')
    # HyperDM arguments
    parser.add_argument("--hyper_net_dims", type=int, nargs="+", default=[128, 256, 512], help="Dimensions for the hyper-network's MLP.")

    args = parser.parse_args()
    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node
    return args, size

def cleanup():
    dist.destroy_process_group()

def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.port_num
    torch.cuda.set_device(args.local_rank)
    gpu = args.local_rank
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(rank, gpu, args)
    dist.barrier()
    cleanup()

if __name__ == '__main__':
    args, size = parse_arguments_hyperdm_single()

    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            args.global_rank = global_rank
            print(f'Node rank {args.node_rank}, local proc {rank}, global proc {global_rank}')
            p = Process(target=init_processes, args=(global_rank, size, train_hyperdm_mudiff_single, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        init_processes(0, 1, train_hyperdm_mudiff_single, args)

