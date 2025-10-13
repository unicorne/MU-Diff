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
from dotenv import load_dotenv
from skimage.metrics import peak_signal_noise_ratio as psnr
import wandb

from torch.multiprocessing import Process
import torch.multiprocessing as mp

# --- Imports from MU-Diff Project ---
from backbones.dense_layer import conv2d
from dataset.dataset_dixon import CreateDatasetSynthesis
from train_utils import (
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
)
from backbones.discriminator import Discriminator_large
from backbones.ncsnpp_generator_adagn_feat import NCSNpp
from backbones.ncsnpp_generator_adagn_feat import NCSNpp_adaptive
from utils.EMA import EMA

from hyperdm.hyperdm import HyperDM_MU_Diff

# %%
def train_hyperdm_mudiff(rank, gpu, args):
    """
    Modified training function to train the HyperDM model.
    """
    # rank = args.node_rank * args.num_process_per_node + gpu

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device('cuda:{}'.format(gpu))

    # ------------------ W&B init (only on rank 0) ------------------
    is_master = (rank == 0)
    if is_master:
        load_dotenv()  # read .env
        api_key = os.getenv("WANDB_API_KEY", "")
        if api_key:
            try:
                wandb.login(key=api_key)
            except Exception:
                pass
        else:
            os.environ["WANDB_MODE"] = os.environ.get("WANDB_MODE", "offline")
        run_name = f"{args.exp}-rank0-{time.strftime('%Y%m%d-%H%M%S')}"
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "mudiff-hyperdm"),
            name=run_name,
            config=vars(args),
            tags=["train", "DDP", "HyperDM"],
            notes=f"Host: {socket.gethostname()}",
        )
    # ---------------------------------------------------------------

    batch_size = args.batch_size
    nz = args.nz  # latent dimension

    dataset = CreateDatasetSynthesis(phase="train", input_path=args.input_path)
    dataset_val = CreateDatasetSynthesis(phase="val", input_path=args.input_path)

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4,
                                              pin_memory=True,
                                              sampler=train_sampler,
                                              drop_last=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val,
                                                                  num_replicas=args.world_size,
                                                                  rank=rank)
    data_loader_val = torch.utils.data.DataLoader(dataset_val,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=4,
                                                  pin_memory=True,
                                                  sampler=val_sampler,
                                                  drop_last=True)

    if is_master:
        print('train data size:' + str(len(data_loader)))
        print('val data size:' + str(len(data_loader_val)))

    to_range_0_1 = lambda x: (x + 1.) / 2.
    critic_criterian = nn.BCEWithLogitsLoss(reduction='none')

    # --- Model Initialization ---
    # NEW: Instantiate the primary networks first, then wrap them in the HyperDM model.
    primary_net1 = NCSNpp(args).to(device)
    primary_net2 = NCSNpp_adaptive(args).to(device)

    # These coefficients are still needed for the diffusion math (q_sample, posterior sampling)
    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)

    hyperdm_model = HyperDM_MU_Diff(
        primary_net1=primary_net1,
        primary_net2=primary_net2,
        hyper_net1_dims=args.hyper_net1_dims,
        hyper_net2_dims=args.hyper_net2_dims,
        diffusion_args=args,
        pos_coeff=pos_coeff
    ).to(device)
    
    if is_master:
        hyperdm_model.print_stats()

    args.num_channels = 1
    att_conv = conv2d(64 * 8, 1, 1, padding=0).to(device)
    optimizer_att = optim.Adam(att_conv.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))

    disc_diffusive_2 = Discriminator_large(nc=2, ngf=args.ngf,
                                           t_emb_dim=args.t_emb_dim,
                                           act=nn.LeakyReLU(0.2)).to(device)
    
    # --- DDP and Optimizer Setup ---
    # NEW: Broadcast hyperdm_model parameters. The primary networks are frozen and don't need broadcasting.
    broadcast_params(hyperdm_model.parameters())
    broadcast_params(disc_diffusive_2.parameters())

    optimizer_disc_diffusive_2 = optim.Adam(disc_diffusive_2.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    # NEW: Create a single optimizer for the hyper-networks within the HyperDM model.
    optimizer_hyperdm = optim.Adam(hyperdm_model.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))

    if args.use_ema:
        optimizer_hyperdm = EMA(optimizer_hyperdm, ema_decay=args.ema_decay)

    scheduler_hyperdm = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_hyperdm, args.num_epoch, eta_min=1e-5)
    scheduler_disc_diffusive_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_disc_diffusive_2, args.num_epoch, eta_min=1e-5)

    # NEW: Wrap the hyperdm_model in DDP
    hyperdm_model = nn.parallel.DistributedDataParallel(hyperdm_model, device_ids=[gpu], find_unused_parameters=False)
    disc_diffusive_2 = nn.parallel.DistributedDataParallel(disc_diffusive_2, device_ids=[gpu])

    # --- Path and Checkpoint Setup ---
    exp = args.exp
    output_path = args.output_path
    exp_path = os.path.join(output_path, exp)
    if rank == 0:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            copy_source(__file__, exp_path)
            shutil.copytree('./backbones', os.path.join(exp_path, 'backbones'))

    # NEW: Checkpoint loading needs to be adapted for the hyperdm_model
    if args.resume:
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        hyperdm_model.load_state_dict(checkpoint['hyperdm_model_dict'])
        optimizer_hyperdm.load_state_dict(checkpoint['optimizer_hyperdm'])
        scheduler_hyperdm.load_state_dict(checkpoint['scheduler_hyperdm'])
        
        disc_diffusive_2.load_state_dict(checkpoint['disc_diffusive_2_dict'])
        optimizer_disc_diffusive_2.load_state_dict(checkpoint['optimizer_disc_diffusive_2'])
        scheduler_disc_diffusive_2.load_state_dict(checkpoint['scheduler_disc_diffusive_2'])

        global_step = checkpoint['global_step']
        if is_master:
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    else:
        global_step, epoch, init_epoch = 0, 0, 0

    # ... (helper functions like _log_images remain the same) ...
    def _log_images(tag, cond1, cond2, cond3, pred1, pred2, gt, step):
        if not is_master:
            return
        try:
            panel = torch.cat([
                torch.clamp(to_range_0_1(cond1), 0, 1), torch.clamp(to_range_0_1(cond2), 0, 1),
                torch.clamp(to_range_0_1(cond3), 0, 1), torch.clamp(to_range_0_1(pred1), 0, 1),
                torch.clamp(to_range_0_1(pred2), 0, 1), torch.clamp(to_range_0_1(gt), 0, 1),
            ], dim=-1)
            grid = torchvision.utils.make_grid(panel, nrow=1, normalize=False)
            wandb.log({tag: wandb.Image(grid.cpu(), caption="cond1|cond2|cond3|pred_g1|pred_g2|gt")}, step=step)
        except Exception as e:
            print(f"W&B image logging failed: {e}")

    # --- Main Training Loop ---
    for epoch in range(init_epoch, args.num_epoch + 1):
        train_sampler.set_epoch(epoch)

        for iteration, (x1, x2, x3, x4) in enumerate(data_loader):
            # === Train Discriminator ===
            for p in disc_diffusive_2.parameters():
                p.requires_grad = True
            disc_diffusive_2.zero_grad()

            cond_data1 = x1.to(device, non_blocking=True)
            cond_data2 = x2.to(device, non_blocking=True)
            cond_data3 = x3.to(device, non_blocking=True)
            real_data = x4.to(device, non_blocking=True)
            t2 = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
            x2_t, x2_tp1 = q_sample_pairs(coeff, real_data, t2)
            x2_t.requires_grad = True

            D2_real, _ = disc_diffusive_2(x2_t, t2, x2_tp1.detach())
            errD_real2 = F.softplus(-D2_real).mean()
            errD_real2.backward(retain_graph=True)

            # ... (r1 regularization remains the same) ...
            if global_step % args.lazy_reg == 0:
                grad2_real = torch.autograd.grad(outputs=D2_real.sum(), inputs=x2_t, create_graph=True)[0]
                grad2_penalty = (grad2_real.view(grad2_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                grad_penalty2 = args.r1_gamma / 2 * grad2_penalty
                grad_penalty2.backward()
            
            # NEW: Generate fake samples using the hyperdm_model in no_grad context for discriminator training
            with torch.no_grad():
                x2_0_predict_diff_g1, x2_0_predict_diff_g2 = hyperdm_model(x2_tp1.detach(), cond_data1, cond_data2, cond_data3, t2)
                x2_pos_sample_g1 = sample_posterior(pos_coeff, x2_0_predict_diff_g1[:, [0], :], x2_tp1, t2)
                x2_pos_sample_g2 = sample_posterior(pos_coeff, x2_0_predict_diff_g2[:, [0], :], x2_tp1, t2)

            output2_g1, _ = disc_diffusive_2(x2_pos_sample_g1, t2, x2_tp1.detach())
            output2_g2, _ = disc_diffusive_2(x2_pos_sample_g2, t2, x2_tp1.detach())
            errD2_fake2_g1 = F.softplus(output2_g1).mean()
            errD2_fake2_g2 = F.softplus(output2_g2).mean()
            errD_fake2 = errD2_fake2_g1 + errD2_fake2_g2
            errD_fake2.backward()
            optimizer_disc_diffusive_2.step()

            # ... (logging for D remains the same) ...
            if is_master:
                _wandb_log({
                    "loss/D/total": errD_real2 + errD_fake2,
                }, step=global_step)


            # === Train Generator (Hyper-networks) ===
            for p in disc_diffusive_2.parameters():
                p.requires_grad = False
            
            # NEW: Zero gradients for the hyperdm model's optimizer
            hyperdm_model.zero_grad()
            att_conv.zero_grad()

            t2 = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
            x2_t, x2_tp1 = q_sample_pairs(coeff, real_data, t2)

            # NEW: The forward call now samples networks and returns predictions
            x2_0_predict_diff_g1, x2_0_predict_diff_g2 = hyperdm_model(x2_tp1.detach(), cond_data1, cond_data2, cond_data3, t2)

            x2_pos_sample_g1 = sample_posterior(pos_coeff, x2_0_predict_diff_g1[:, [0], :], x2_tp1, t2)
            x2_pos_sample_g2 = sample_posterior(pos_coeff, x2_0_predict_diff_g2[:, [0], :], x2_tp1, t2)
            
            output2_g1, att_feat_g1 = disc_diffusive_2(x2_pos_sample_g1, t2, x2_tp1.detach())
            output2_g2, att_feat_g2 = disc_diffusive_2(x2_pos_sample_g2, t2, x2_tp1.detach())

            # ... (rest of G loss calculation is the same) ...
            att_map_g1 = torch.sigmoid(att_conv(att_feat_g1))
            att_map_g1 = F.interpolate(att_map_g1, size=(256, 256), mode='bilinear', align_corners=False)
            att_map_g2 = torch.sigmoid(att_conv(att_feat_g2))
            att_map_g2 = F.interpolate(att_map_g2, size=(256, 256), mode='bilinear', align_corners=False)
            mask_loss_1 = (att_map_g2 * critic_criterian(x2_pos_sample_g1, torch.sigmoid(x2_pos_sample_g2))).mean()
            mask_loss_2 = (att_map_g1 * critic_criterian(x2_pos_sample_g2, torch.sigmoid(x2_pos_sample_g1))).mean()
            mask_loss = mask_loss_1 + mask_loss_2
            errG2 = F.softplus(-output2_g1).mean()
            errG4 = F.softplus(-output2_g2).mean()
            errG_adv = errG2 + errG4
            errG1_2_L1 = F.l1_loss(x2_0_predict_diff_g1[:, [0], :], real_data)
            errG2_2_L1 = F.l1_loss(x2_0_predict_diff_g2[:, [0], :], real_data)
            errG_L1 = errG1_2_L1 + errG2_2_L1
            errG = errG_adv + (args.lambda_l1_loss * errG_L1) + (args.lambda_mask_loss * mask_loss)
            
            errG.backward()
            # NEW: Step the hyperdm optimizer
            optimizer_hyperdm.step()
            optimizer_att.step()

            # ... (logging for G remains the same) ...
            if is_master and iteration % 100 == 0:
                print(f'epoch {epoch} iteration {iteration}, G-Total: {errG.item():.4f}, G-Adv: {errG_adv.item():.4f}')
                _log_images("train/strip", cond_data1, cond_data2, cond_data3, 
                            x2_0_predict_diff_g1, x2_0_predict_diff_g2, real_data, step=global_step)
            
            global_step += 1

        # --- Schedulers and Checkpointing ---
        # NEW: Step the new scheduler
        if not args.no_lr_decay:
            scheduler_hyperdm.step()
            scheduler_disc_diffusive_2.step()
        
        # NEW: Checkpointing needs to be updated to save the hyperdm_model state
        if is_master and args.save_content and (epoch % args.save_content_every == 0):
            print('Saving content.')
            content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                        'hyperdm_model_dict': hyperdm_model.state_dict(),
                        'optimizer_hyperdm': optimizer_hyperdm.state_dict(),
                        'scheduler_hyperdm': scheduler_hyperdm.state_dict(),
                        'disc_diffusive_2_dict': disc_diffusive_2.state_dict(),
                        'optimizer_disc_diffusive_2': optimizer_disc_diffusive_2.state_dict(),
                        'scheduler_disc_diffusive_2': scheduler_disc_diffusive_2.state_dict(),
                        }
            torch.save(content, os.path.join(exp_path, 'content.pth'))

        # NEW: Also update saving of EMA weights if used
        if is_master and (epoch % args.save_ckpt_every == 0):
            if args.use_ema:
                optimizer_hyperdm.swap_parameters_with_ema(store_params_in_ema=True)
            
            torch.save(hyperdm_model.state_dict(), os.path.join(exp_path, f'hyperdm_model_{epoch}.pth'))

            if args.use_ema:
                optimizer_hyperdm.swap_parameters_with_ema(store_params_in_ema=True)
        
        # ... (Validation loop can remain mostly the same, as it uses the full sampling function) ...

    # --- W&B Finish ---
    if is_master:
        try:
            wandb.finish()
        except Exception:
            pass

# NEW: Add arguments for hyper-network dimensions
def parse_arguments_hyperdm():
    parser = argparse.ArgumentParser('mudiff-hyperdm parameters')
    # ... (all previous arguments from your train_utils.py)
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--num_channels', type=int, default=1)
    parser.add_argument('--centered', action='store_false', default=True)
    parser.add_argument('--use_geometric', action='store_true', default=False)
    parser.add_argument('--beta_min', type=float, default=0.1)
    parser.add_argument('--beta_max', type=float, default=20.)
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
    parser.add_argument('--exp', default='mudiff-hyperdm-exp')
    parser.add_argument('--input_path', default='/data/BRATS/')
    parser.add_argument('--output_path', default='/results')
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)
    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epoch', type=int, default=1200)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--lr_g', type=float, default=1.6e-4)
    parser.add_argument('--lr_d', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--no_lr_decay', action='store_true', default=False)
    parser.add_argument('--use_ema', action='store_true', default=True)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--r1_gamma', type=float, default=0.05)
    parser.add_argument('--lazy_reg', type=int, default=10)
    parser.add_argument('--save_content', action='store_true', default=True)
    parser.add_argument('--save_content_every', type=int, default=50)
    parser.add_argument('--save_ckpt_every', type=int, default=50)
    parser.add_argument('--lambda_l1_loss', type=float, default=0.5)
    parser.add_argument('--lambda_mask_loss', type=float, default=0.1)
    parser.add_argument('--num_proc_node', type=int, default=1)
    parser.add_argument('--num_process_per_node', type=int, default=1)
    parser.add_argument('--node_rank', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--master_address', type=str, default='127.0.0.1')
    parser.add_argument('--port_num', type=str, default='6021')

    # NEW: Add hyper-network dimensions arguments
    parser.add_argument("--hyper_net1_dims", type=int, nargs="+", default=[128, 256, 512], help="Dimensions for the first hyper-network's MLP.")
    parser.add_argument("--hyper_net2_dims", type=int, nargs="+", default=[128, 256, 512], help="Dimensions for the second hyper-network's MLP.")

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


# --- Main execution block ---
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True) 
    args, size = parse_arguments_hyperdm()

    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print(f'Node rank {args.node_rank}, local proc {rank}, global proc {global_rank}')
            p = mp.Process(target=init_processes, args=(global_rank, global_size, train_hyperdm_mudiff, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        # HACK for non-DDP execution
        def init_processes(rank, size, fn, args):
            fn(rank, rank, args)
        init_processes(0, 1, train_hyperdm_mudiff, args)



### Key Changes and How to Run

#1.  **Model Initialization**: The script now initializes `NCSNpp` and `NCSNpp_adaptive` and immediately passes them to the `HyperDM_MU_Diff` wrapper. This wrapper is then the main model that gets trained.
#2.  **Optimizer**: The optimizers for the individual generators have been replaced with a single `optimizer_hyperdm` that targets the parameters of the hyper-networks inside the wrapper.
#3.  **Training Loop**:
#    * In the **discriminator training phase**, fake samples are generated using `hyperdm_model` within a `torch.no_grad()` block. This is crucial because you don't want to update the hyper-networks when training the discriminator.
#    * In the **generator training phase**, a call to `hyperdm_model(...)` performs the core logic: it samples new weights from the hyper-networks and computes the denoised predictions. The subsequent loss calculation and backpropagation now update the weights of the hyper-networks.
#4.  **Configuration**: I added `--hyper_net1_dims` and `--hyper_net2_dims` arguments so you can define the architecture of your hyper-networks from the command line. The default values `[128, 256, 512]` define an MLP with a 128-dimensional input noise vector and two hidden layers.
#5.  **Checkpointing**: The saving and loading logic has been updated to handle the `hyperdm_model`'s state dictionary.
#
#To run this script, you would execute it from your terminal like this:
#
#```bash
#python train_hyperdm.py --exp my_hyperdm_experiment --batch_size 2 --hyper_net1_dims 128 256 --hyper_net2_dims 128 256 ... [other args]
#