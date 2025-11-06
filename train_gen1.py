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

from backbones.dense_layer import conv2d
from dataset.dataset_dixon import CreateDatasetSynthesis
from train_utils import (
    parse_arguments,
    copy_source,
    broadcast_params,
    get_time_schedule,
    _wandb_log,
    q_sample_pairs,
    sample_posterior,
    Diffusion_Coefficients,
    Posterior_Coefficients,
)


def sample_from_model(coefficients, generator, cond1, cond2, cond3, n_time, x_init, T, opt):
    x = x_init
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            x_0 = generator(x, cond1, cond2, cond3, t_time, latent_z)
            x_new = sample_posterior(coefficients, x_0[:, [0], :], x, t)
            x = x_new.detach()
    return x


# %%
def train_mudiff(rank, gpu, args):
    from backbones.discriminator import Discriminator_large
    from backbones.ncsnpp_generator_adagn_feat import NCSNpp
    from utils.EMA import EMA

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
            project=os.environ.get("WANDB_PROJECT", "mudiff"),
            name=run_name,
            config=vars(args),
            tags=["train", "DDP", "single-generator"],
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

    val_l1_loss = np.zeros([1, args.num_epoch, len(data_loader_val)])
    val_psnr_values = np.zeros([1, args.num_epoch, len(data_loader_val)])

    if is_master:
        print('train data size:' + str(len(data_loader)))
        print('val data size:' + str(len(data_loader_val)))
    to_range_0_1 = lambda x: (x + 1.) / 2.

    # networks performing reverse denoising
    gen_diffusive = NCSNpp(args).to(device)
    disc_diffusive = Discriminator_large(nc=2, ngf=args.ngf,
                                         t_emb_dim=args.t_emb_dim,
                                         act=nn.LeakyReLU(0.2)).to(device)

    broadcast_params(gen_diffusive.parameters())
    broadcast_params(disc_diffusive.parameters())

    optimizer_disc = optim.Adam(disc_diffusive.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    optimizer_gen = optim.Adam(gen_diffusive.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))

    if args.use_ema:
        optimizer_gen = EMA(optimizer_gen, ema_decay=args.ema_decay)

    scheduler_gen = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gen, args.num_epoch, eta_min=1e-5)
    scheduler_disc = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_disc, args.num_epoch, eta_min=1e-5)

    # ddp
    gen_diffusive = nn.parallel.DistributedDataParallel(gen_diffusive, device_ids=[gpu])
    disc_diffusive = nn.parallel.DistributedDataParallel(disc_diffusive, device_ids=[gpu])

    exp_path = os.path.join(args.output_path, args.exp)
    if is_master:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            copy_source(__file__, exp_path)
            shutil.copytree('./backbones', os.path.join(exp_path, 'backbones'))

    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    T = get_time_schedule(args, device)

    if args.resume:
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        gen_diffusive.load_state_dict(checkpoint['gen_diffusive_dict'])
        optimizer_gen.load_state_dict(checkpoint['optimizer_gen'])
        scheduler_gen.load_state_dict(checkpoint['scheduler_gen'])
        disc_diffusive.load_state_dict(checkpoint['disc_diffusive_dict'])
        optimizer_disc.load_state_dict(checkpoint['optimizer_disc'])
        scheduler_disc.load_state_dict(checkpoint['scheduler_disc'])
        global_step = checkpoint['global_step']
        if is_master:
            print(f"=> loaded checkpoint (epoch {init_epoch})")
    else:
        global_step, init_epoch = 0, 0

    def _log_images(tag, cond1, cond2, cond3, pred, gt, step):
        if not is_master:
            return
        try:
            panel = torch.cat([
                torch.clamp(to_range_0_1(cond1), 0, 1),
                torch.clamp(to_range_0_1(cond2), 0, 1),
                torch.clamp(to_range_0_1(cond3), 0, 1),
                torch.clamp(to_range_0_1(pred), 0, 1),
                torch.clamp(to_range_0_1(gt), 0, 1),
            ], dim=-1)
            grid = torchvision.utils.make_grid(panel, nrow=1, normalize=False)
            wandb.log({tag: wandb.Image(grid.cpu(), caption="cond1|cond2|cond3|pred|gt")}, step=step)
        except Exception as e:
            print(f"W&B image logging failed: {e}")

    for epoch in range(init_epoch, args.num_epoch + 1):
        train_sampler.set_epoch(epoch)

        for iteration, (x1, x2, x3, x4) in enumerate(data_loader):
            # ------------------- TRAIN DISCRIMINATOR -------------------
            for p in disc_diffusive.parameters():
                p.requires_grad = True
            disc_diffusive.zero_grad()

            cond_data1, cond_data2, cond_data3, real_data = [d.to(device, non_blocking=True) for d in (x1, x2, x3, x4)]
            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
            x_t.requires_grad = True

            # Train with real
            D_real, _ = disc_diffusive(x_t, t, x_tp1.detach())
            errD_real = F.softplus(-D_real).mean()
            errD_real.backward(retain_graph=True)

            if args.lazy_reg is None or global_step % args.lazy_reg == 0:
                grad_real = torch.autograd.grad(outputs=D_real.sum(), inputs=x_t, create_graph=True)[0]
                grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                grad_penalty = args.r1_gamma / 2 * grad_penalty
                grad_penalty.backward()

            # Train with fake
            with torch.no_grad():
                latent_z = torch.randn(batch_size, nz, device=device)
                x_0_predict = gen_diffusive(x_tp1.detach(), cond_data1, cond_data2, cond_data3, t, latent_z)
                x_pos_sample = sample_posterior(pos_coeff, x_0_predict[:, [0], :], x_tp1, t)

            output, _ = disc_diffusive(x_pos_sample, t, x_tp1.detach())
            errD_fake = F.softplus(output).mean()
            errD_fake.backward()

            optimizer_disc.step()

            # --- log D losses
            if is_master:
                _wandb_log({
                    "loss/D/real": errD_real.item(),
                    "loss/D/fake": errD_fake.item(),
                    "loss/D/total": errD_real.item() + errD_fake.item(),
                    "loss/D/grad_penalty": grad_penalty.item() if 'grad_penalty' in locals() else 0,
                    "lr/D": optimizer_disc.param_groups[0]["lr"],
                }, step=global_step)

            # ------------------- TRAIN GENERATOR -------------------
            for p in disc_diffusive.parameters():
                p.requires_grad = False
            gen_diffusive.zero_grad()

            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
            latent_z = torch.randn(batch_size, nz, device=device)
            x_0_predict = gen_diffusive(x_tp1.detach(), cond_data1, cond_data2, cond_data3, t, latent_z)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict[:, [0], :], x_tp1, t)

            output, _ = disc_diffusive(x_pos_sample, t, x_tp1.detach())
            errG_adv = F.softplus(-output).mean()
            errG_L1 = F.l1_loss(x_0_predict[:, [0], :], real_data)
            errG = errG_adv + (args.lambda_l1_loss * errG_L1)

            errG.backward()
            optimizer_gen.step()

            # --- log G losses and metrics
            if is_master:
                pred_01 = torch.clamp(to_range_0_1(x_0_predict[:, [0], :].detach()), 0, 1)
                gt_01 = torch.clamp(to_range_0_1(real_data.detach()), 0, 1)
                psnr_train = psnr(gt_01.cpu().numpy(), pred_01.cpu().numpy(), data_range=1.0)
                l1_train = F.l1_loss(pred_01, gt_01).item()

                _wandb_log({
                    "loss/G/adv": errG_adv.item(),
                    "loss/G/L1": errG_L1.item(),
                    "loss/G/total": errG.item(),
                    "metric/train/psnr": psnr_train,
                    "metric/train/l1": l1_train,
                    "lr/G": optimizer_gen.param_groups[0]["lr"],
                }, step=global_step)

            if iteration % 100 == 0 and is_master:
                print(f'epoch {epoch} iter {iteration}, G-Adv: {errG_adv.item():.4f}, G-L1: {errG_L1.item():.4f}')
                _log_images("train/strip",
                            cond_data1[:1].detach(), cond_data2[:1].detach(), cond_data3[:1].detach(),
                            x_0_predict[:1, :1].detach(), real_data[:1].detach(),
                            step=global_step)

            global_step += 1

        # ------------------- END OF EPOCH -------------------
        if not args.no_lr_decay:
            scheduler_gen.step()
            scheduler_disc.step()

        if is_master and epoch % 10 == 0:
            x_t = torch.randn_like(real_data[:1])
            fake_sample = sample_from_model(pos_coeff, gen_diffusive,
                                            cond_data1[:1], cond_data2[:1], cond_data3[:1],
                                            args.num_timesteps, x_t, T, args)
            panel = torch.cat((real_data[:1], fake_sample), axis=-1)
            save_path = os.path.join(exp_path, f'sample_epoch_{epoch}.png')
            torchvision.utils.save_image(panel, save_path, normalize=True)
            try:
                wandb.log({"train/sample": wandb.Image(
                    torchvision.utils.make_grid(panel, normalize=True).cpu(),
                    caption=f"gt | sample @ epoch {epoch}"
                )}, step=global_step)
            except Exception as e:
                print(f"W&B image logging failed: {e}")

        # --- Validation loop
        if is_master:
            for i, (x1_val, x2_val, x3_val, x4_val) in enumerate(data_loader_val):
                cond1, cond2, cond3, real = [d.to(device) for d in (x1_val, x2_val, x3_val, x4_val)]
                x_t = torch.randn_like(real)
                fake_sample = sample_from_model(pos_coeff, gen_diffusive, cond1, cond2, cond3,
                                                args.num_timesteps, x_t, T, args)
                
                fake_sample_val = to_range_0_1(fake_sample.detach())
                fake_sample_val_div_mean = fake_sample_val / fake_sample_val.mean()
                real_data_val = to_range_0_1(real.detach())
                real_data_val_div_mean = real_data_val / real_data_val.mean()

                fake_sample_val_np = fake_sample_val_div_mean.cpu().numpy()
                real_data_val_np = real_data_val_div_mean.cpu().numpy()
                
                val_l1_loss[0, epoch, i] = abs(fake_sample_val_np - real_data_val_np).mean()
                val_psnr_values[0, epoch, i] = psnr(real_data_val_np, fake_sample_val_np, data_range=real_data_val_np.max())

            val_psnr_mean = np.nanmean(val_psnr_values[0, epoch, :])
            val_l1_mean = np.nanmean(val_l1_loss[0, epoch, :])

            print(f"Epoch {epoch} Val PSNR: {val_psnr_mean:.4f}, Val L1: {val_l1_mean:.4f}")
            _wandb_log({
                "metric/val/psnr_mean": val_psnr_mean,
                "metric/val/l1_mean": val_l1_mean,
                "epoch": epoch,
            }, step=global_step)
            
            _log_images("val/strip",
                        cond1[:1].detach(), cond2[:1].detach(), cond3[:1].detach(),
                        fake_sample[:1].detach(), real[:1].detach(),
                        step=global_step)
        
        np.save(f'{exp_path}/val_l1_loss.npy', val_l1_loss)
        np.save(f'{exp_path}/val_psnr_values.npy', val_psnr_values)

        if is_master and args.save_content and (epoch % args.save_content_every == 0):
            print('Saving content.')
            content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                       'gen_diffusive_dict': gen_diffusive.state_dict(),
                       'optimizer_gen': optimizer_gen.state_dict(),
                       'scheduler_gen': scheduler_gen.state_dict(),
                       'disc_diffusive_dict': disc_diffusive.state_dict(),
                       'optimizer_disc': optimizer_disc.state_dict(),
                       'scheduler_disc': scheduler_disc.state_dict()}
            torch.save(content, os.path.join(exp_path, 'content.pth'))

        if is_master and (epoch % args.save_ckpt_every == 0):
            if args.use_ema:
                optimizer_gen.swap_parameters_with_ema(store_params_in_ema=True)
            torch.save(gen_diffusive.state_dict(), os.path.join(exp_path, f'gen_{epoch}.pth'))
            if args.use_ema:
                optimizer_gen.swap_parameters_with_ema(store_params_in_ema=True)

    if is_master:
        try:
            wandb.finish()
        except Exception:
            pass


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


if __name__ == '__main__':
    args, size = parse_arguments()
    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            args.global_rank = global_rank
            print(f'Node rank {args.node_rank}, local proc {rank}, global proc {global_rank}')
            p = Process(target=init_processes, args=(global_rank, args.world_size, train_mudiff, args))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        init_processes(0, 1, train_mudiff, args)