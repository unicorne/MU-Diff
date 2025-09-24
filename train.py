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
from dataset_dixon import CreateDatasetSynthesis
from train_utils import (
    parse_arguments,
    copy_source,
    broadcast_params,
    get_time_schedule,
    _var_func_vp,
    _psnr_torch,
    _wandb_log,
    q_sample_pairs,
    sample_posterior,
    sample_from_model,
    Diffusion_Coefficients,
    Posterior_Coefficients,
)



# %%
def train_mudiff(rank, gpu, args):
    from backbones.discriminator import Discriminator_large

    from backbones.ncsnpp_generator_adagn_feat import NCSNpp
    from backbones.ncsnpp_generator_adagn_feat import NCSNpp_adaptive

    from utils.EMA import EMA

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
            project=os.environ.get("WANDB_PROJECT", "mudiff"),
            name=run_name,
            config=vars(args),
            tags=["train", "DDP"],
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

    val_l1_loss = np.zeros([2, args.num_epoch, len(data_loader_val)])
    val_psnr_values = np.zeros([2, args.num_epoch, len(data_loader_val)])
    if is_master:
        print('train data size:' + str(len(data_loader)))
        print('val data size:' + str(len(data_loader)))
    to_range_0_1 = lambda x: (x + 1.) / 2.
    critic_criterian = nn.BCEWithLogitsLoss(reduction='none')

    # networks performing reverse denoising
    gen_diffusive_1 = NCSNpp(args).to(device)
    gen_diffusive_2 = NCSNpp_adaptive(args).to(device)

    args.num_channels = 1
    att_conv = conv2d(64 * 8, 1, 1, padding=0).cuda()

    disc_diffusive_2 = Discriminator_large(nc=2, ngf=args.ngf,
                                           t_emb_dim=args.t_emb_dim,
                                           act=nn.LeakyReLU(0.2)).to(device)

    broadcast_params(gen_diffusive_1.parameters())
    broadcast_params(gen_diffusive_2.parameters())

    broadcast_params(disc_diffusive_2.parameters())

    optimizer_disc_diffusive_2 = optim.Adam(disc_diffusive_2.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    optimizer_gen_diffusive_1 = optim.Adam(gen_diffusive_1.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))

    optimizer_gen_diffusive_2 = optim.Adam(gen_diffusive_2.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))

    if args.use_ema:
        optimizer_gen_diffusive_1 = EMA(optimizer_gen_diffusive_1, ema_decay=args.ema_decay)
        optimizer_gen_diffusive_2 = EMA(optimizer_gen_diffusive_2, ema_decay=args.ema_decay)

    scheduler_gen_diffusive_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gen_diffusive_1, args.num_epoch,
                                                                           eta_min=1e-5)
    scheduler_gen_diffusive_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gen_diffusive_2, args.num_epoch,
                                                                           eta_min=1e-5)

    scheduler_disc_diffusive_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_disc_diffusive_2, args.num_epoch,
                                                                            eta_min=1e-5)

    # ddp
    gen_diffusive_1 = nn.parallel.DistributedDataParallel(gen_diffusive_1, device_ids=[gpu])
    gen_diffusive_2 = nn.parallel.DistributedDataParallel(gen_diffusive_2, device_ids=[gpu])

    disc_diffusive_2 = nn.parallel.DistributedDataParallel(disc_diffusive_2, device_ids=[gpu])

    exp = args.exp

    output_path = args.output_path

    exp_path = os.path.join(output_path, exp)
    if rank == 0:
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
        epoch = init_epoch
        gen_diffusive_1.load_state_dict(checkpoint['gen_diffusive_1_dict'])
        gen_diffusive_2.load_state_dict(checkpoint['gen_diffusive_2_dict'])

        # load G

        optimizer_gen_diffusive_1.load_state_dict(checkpoint['optimizer_gen_diffusive_1'])
        scheduler_gen_diffusive_1.load_state_dict(checkpoint['scheduler_gen_diffusive_1'])
        optimizer_gen_diffusive_2.load_state_dict(checkpoint['optimizer_gen_diffusive_2'])
        scheduler_gen_diffusive_2.load_state_dict(checkpoint['scheduler_gen_diffusive_2'])

        # load D

        disc_diffusive_2.load_state_dict(checkpoint['disc_diffusive_2_dict'])
        optimizer_disc_diffusive_2.load_state_dict(checkpoint['optimizer_disc_diffusive_2'])
        scheduler_disc_diffusive_2.load_state_dict(checkpoint['scheduler_disc_diffusive_2'])

        global_step = checkpoint['global_step']
        if is_master:
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    else:
        global_step, epoch, init_epoch = 0, 0, 0

    # --- small helper to log a panel of images to W&B (master only)
    def _log_images(tag, cond1, cond2, cond3, pred1, pred2, gt, step):
        if not is_master:
            return
        try:
            # build a horizontal strip: cond1 | cond2 | cond3 | pred1 | pred2 | gt
            # clamp to [0,1] for nicer viewing
            panel = torch.cat([
                torch.clamp((cond1 + 1) / 2, 0, 1),
                torch.clamp((cond2 + 1) / 2, 0, 1),
                torch.clamp((cond3 + 1) / 2, 0, 1),
                torch.clamp((pred1 + 1) / 2, 0, 1),
                torch.clamp((pred2 + 1) / 2, 0, 1),
                torch.clamp((gt + 1) / 2, 0, 1),
            ], dim=-1)
            grid = torchvision.utils.make_grid(panel, nrow=1, normalize=False)
            wandb.log({tag: wandb.Image(grid.cpu(), caption="cond1|cond2|cond3|pred_g1|pred_g2|gt")}, step=step)
        except Exception:
            pass

    for epoch in range(init_epoch, args.num_epoch + 1):
        # train_sampler.set_epoch(epoch)

        for iteration, (x1, x2, x3, x4) in enumerate(data_loader):
            for p in disc_diffusive_2.parameters():
                p.requires_grad = True

            disc_diffusive_2.zero_grad()

            # sample from p(x_0)
            cond_data1 = x1.to(device, non_blocking=True)
            cond_data2 = x2.to(device, non_blocking=True)
            cond_data3 = x3.to(device, non_blocking=True)
            real_data = x4.to(device, non_blocking=True)

            t2 = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)

            x2_t, x2_tp1 = q_sample_pairs(coeff, real_data, t2)
            x2_t.requires_grad = True

            # train discriminator with real
            D2_real, _ = disc_diffusive_2(x2_t, t2, x2_tp1.detach())

            errD2_real2 = F.softplus(-D2_real)
            errD2_real2 = errD2_real2.mean()
            errD_real2 = errD2_real2
            errD_real2.backward(retain_graph=True)

            if args.lazy_reg is None:

                grad2_real = torch.autograd.grad(
                    outputs=D2_real.sum(), inputs=x2_t, create_graph=True
                )[0]
                grad2_penalty = (
                        grad2_real.view(grad2_real.size(0), -1).norm(2, dim=1) ** 2
                ).mean()

                grad_penalty2 = args.r1_gamma / 2 * grad2_penalty
                grad_penalty2.backward()
            else:
                if global_step % args.lazy_reg == 0:
                    grad2_real = torch.autograd.grad(
                        outputs=D2_real.sum(), inputs=x2_t, create_graph=True
                    )[0]
                    grad2_penalty = (
                            grad2_real.view(grad2_real.size(0), -1).norm(2, dim=1) ** 2
                    ).mean()

                    grad_penalty2 = args.r1_gamma / 2 * grad2_penalty
                    grad_penalty2.backward()

            # train with fake

            latent_z2 = torch.randn(batch_size, nz, device=device)

            x2_0_predict_diff_g1 = gen_diffusive_1(x2_tp1.detach(), cond_data1, cond_data2, cond_data3, t2, latent_z2)

            x2_0_predict_diff_g2 = gen_diffusive_2(x2_tp1.detach(), cond_data1, cond_data2, cond_data3, t2, latent_z2,
                                                   x2_0_predict_diff_g1[:, [0], :])

            x2_pos_sample_g1 = sample_posterior(pos_coeff, x2_0_predict_diff_g1[:, [0], :], x2_tp1, t2)

            x2_pos_sample_g2 = sample_posterior(pos_coeff, x2_0_predict_diff_g2[:, [0], :], x2_tp1, t2)

            # D output for fake sample x_pos_sample

            output2_g1, _ = disc_diffusive_2(x2_pos_sample_g1, t2, x2_tp1.detach())

            output2_g2, _ = disc_diffusive_2(x2_pos_sample_g2, t2, x2_tp1.detach())

            errD2_fake2_g1 = (F.softplus(output2_g1)).mean()

            errD2_fake2_g2 = (F.softplus(output2_g2)).mean()

            errD_fake2 = errD2_fake2_g1 + errD2_fake2_g2
            errD_fake2.backward()

            optimizer_disc_diffusive_2.step()

            # --- log D losses (master only)
            if is_master:
                _wandb_log({
                    "loss/D/real": errD_real2,
                    "loss/D/fake_g1": errD2_fake2_g1,
                    "loss/D/fake_g2": errD2_fake2_g2,
                    "loss/D/total": errD_real2 + errD2_fake2_g1 + errD2_fake2_g2,
                    "lr/D": optimizer_disc_diffusive_2.param_groups[0]["lr"],
                }, step=global_step)

            for p in disc_diffusive_2.parameters():
                p.requires_grad = False

            cond_data1 = x1.to(device, non_blocking=True)
            cond_data2 = x2.to(device, non_blocking=True)
            cond_data3 = x3.to(device, non_blocking=True)
            real_data = x4.to(device, non_blocking=True)

            gen_diffusive_1.zero_grad()
            gen_diffusive_2.zero_grad()

            t2 = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)

            # sample x_t and x_tp1
            x2_t, x2_tp1 = q_sample_pairs(coeff, real_data, t2)

            latent_z2 = torch.randn(batch_size, nz, device=device)

            x2_0_predict_diff_g1 = gen_diffusive_1(x2_tp1.detach(), cond_data1, cond_data2, cond_data3, t2, latent_z2)

            x2_0_predict_diff_g2 = gen_diffusive_2(x2_tp1.detach(), cond_data1, cond_data2, cond_data3, t2, latent_z2,
                                                   x2_0_predict_diff_g1[:, [0], :])

            # sampling q(x_t | x_0_predict, x_t+1)
            x2_pos_sample_g1 = sample_posterior(pos_coeff, x2_0_predict_diff_g1[:, [0], :], x2_tp1, t2)

            x2_pos_sample_g2 = sample_posterior(pos_coeff, x2_0_predict_diff_g2[:, [0], :], x2_tp1, t2)

            # D output for fake sample x_pos_sample
            output2_g1, att_feat_g1 = disc_diffusive_2(x2_pos_sample_g1, t2, x2_tp1.detach())

            output2_g2, att_feat_g2 = disc_diffusive_2(x2_pos_sample_g2, t2, x2_tp1.detach())

            att_map_g1 = torch.sigmoid(att_conv(att_feat_g1))
            att_map_g1 = F.interpolate(att_map_g1, size=(256, 256), mode='bilinear', align_corners=False)

            att_map_g2 = torch.sigmoid(att_conv(att_feat_g2))
            att_map_g2 = F.interpolate(att_map_g2, size=(256, 256), mode='bilinear', align_corners=False)

            mask_loss_1 = (att_map_g2 * critic_criterian(x2_pos_sample_g1, torch.sigmoid(x2_pos_sample_g2))).mean()
            mask_loss_2 = (att_map_g1 * critic_criterian(x2_pos_sample_g2, torch.sigmoid(x2_pos_sample_g1))).mean()

            mask_loss = mask_loss_1 + mask_loss_2

            errG2 = F.softplus(-output2_g1)
            errG2 = errG2.mean()

            errG4 = F.softplus(-output2_g2)
            errG4 = errG4.mean()

            errG_adv = errG2 + errG4

            errG1_2_L1 = F.l1_loss(x2_0_predict_diff_g1[:, [0], :], real_data)

            errG2_2_L1 = F.l1_loss(x2_0_predict_diff_g2[:, [0], :], real_data)

            errG_L1 = errG1_2_L1 + errG2_2_L1

            errG = errG_adv + (args.lambda_l1_loss * errG_L1) + (args.lambda_mask_loss * mask_loss)

            errG.backward()
            optimizer_gen_diffusive_1.step()
            optimizer_gen_diffusive_2.step()

            # --- compute & log train PSNR/L1 (on predicted x0 vs GT), master only
            if is_master:
                # bring to [0,1] for PSNR stability
                pred1_01 = torch.clamp(to_range_0_1(x2_0_predict_diff_g1[:, [0], :].detach()), 0, 1)
                pred2_01 = torch.clamp(to_range_0_1(x2_0_predict_diff_g2[:, [0], :].detach()), 0, 1)
                gt_01    = torch.clamp(to_range_0_1(real_data.detach()), 0, 1)

                psnr_g1 = _psnr_torch(pred1_01, gt_01)
                psnr_g2 = _psnr_torch(pred2_01, gt_01)
                l1_g1   = F.l1_loss(pred1_01, gt_01).item()
                l1_g2   = F.l1_loss(pred2_01, gt_01).item()

                _wandb_log({
                    "loss/G/adv_g1": errG2,
                    "loss/G/adv_g2": errG4,
                    "loss/G/adv_total": errG_adv,
                    "loss/G/L1_g1": errG1_2_L1,
                    "loss/G/L1_g2": errG2_2_L1,
                    "loss/G/L1_total": errG_L1,
                    "loss/G/mask": mask_loss,
                    "loss/G/total": errG,
                    "metric/train/psnr_g1": psnr_g1,
                    "metric/train/psnr_g2": psnr_g2,
                    "metric/train/l1_g1": l1_g1,
                    "metric/train/l1_g2": l1_g2,
                    "lr/G1": optimizer_gen_diffusive_1.param_groups[0]["lr"],
                    "lr/G2": optimizer_gen_diffusive_2.param_groups[0]["lr"],
                }, step=global_step)

            global_step += 1
            if iteration % 100 == 0:
                if rank == 0:
                    print('epoch {} iteration{},  G-Adv: {}, G-Sum: {}'.format(epoch, iteration,
                                                                               errG_adv.item(), errG.item()))
                    # also push a quick image strip
                    _log_images("train/strip",
                                cond_data1[:1].detach(),
                                cond_data2[:1].detach(),
                                cond_data3[:1].detach(),
                                x2_0_predict_diff_g1[:1, :1].detach(),
                                x2_0_predict_diff_g2[:1, :1].detach(),
                                real_data[:1].detach(),
                                step=global_step)

        if not args.no_lr_decay:
            scheduler_gen_diffusive_1.step()
            scheduler_gen_diffusive_2.step()

            scheduler_disc_diffusive_2.step()

        if rank == 0:
            if epoch % 10 == 0:
                torchvision.utils.save_image(x2_pos_sample_g1,
                                             os.path.join(exp_path, 'xposg1_epoch_{}.png'.format(epoch)),
                                             normalize=True)
                torchvision.utils.save_image(x2_pos_sample_g2,
                                             os.path.join(exp_path, 'xposg2_epoch_{}.png'.format(epoch)),
                                             normalize=True)
                # log these to W&B too
                try:
                    wandb.log({
                        "train/xpos_g1": wandb.Image(
                            torchvision.utils.make_grid(x2_pos_sample_g1, normalize=True).cpu(),
                            caption=f"xpos_g1 epoch {epoch}"
                        ),
                        "train/xpos_g2": wandb.Image(
                            torchvision.utils.make_grid(x2_pos_sample_g2, normalize=True).cpu(),
                            caption=f"xpos_g2 epoch {epoch}"
                        ),
                    }, step=global_step)
                except Exception:
                    pass

            # concatenate noise and source contrast
            x2_t = torch.randn_like(real_data)
            fake_sample = sample_from_model(pos_coeff, gen_diffusive_1, cond_data1, gen_diffusive_2, cond_data2,
                                            cond_data3,
                                            args.num_timesteps, x2_t, T, args)

            fake_sample = torch.cat((real_data, fake_sample), axis=-1)

            torchvision.utils.save_image(fake_sample,
                                         os.path.join(exp_path, 'sample_discrete_epoch_{}.png'.format(epoch)),
                                         normalize=True)

            # also log a sample panel
            try:
                wandb.log({
                    "train/sample_discrete": wandb.Image(
                        torchvision.utils.make_grid(fake_sample, normalize=True).cpu(),
                        caption=f"sample_discrete epoch {epoch}"
                    )
                }, step=global_step)
            except Exception:
                pass

            if args.save_content:
                if epoch % args.save_content_every == 0:
                    print('Saving content.')
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
                               }

                    torch.save(content, os.path.join(exp_path, 'content.pth'))

            if epoch % args.save_ckpt_every == 0:
                if args.use_ema:
                    optimizer_gen_diffusive_1.swap_parameters_with_ema(store_params_in_ema=True)
                    optimizer_gen_diffusive_2.swap_parameters_with_ema(store_params_in_ema=True)

                torch.save(gen_diffusive_1.state_dict(), os.path.join(exp_path, 'gen_diffusive_1_{}.pth'.format(epoch)))
                torch.save(gen_diffusive_2.state_dict(), os.path.join(exp_path, 'gen_diffusive_2_{}.pth'.format(epoch)))

                if args.use_ema:
                    optimizer_gen_diffusive_1.swap_parameters_with_ema(store_params_in_ema=True)
                    optimizer_gen_diffusive_2.swap_parameters_with_ema(store_params_in_ema=True)

        # ---------------- Validation loop (per-epoch) ----------------
        for iteration, (x1_val, x2_val, x3_val, x4_val) in enumerate(data_loader_val):
            cond_data1_val = x1_val.to(device, non_blocking=True)
            cond_data2_val = x2_val.to(device, non_blocking=True)
            cond_data3_val = x3_val.to(device, non_blocking=True)
            real_data_val = x4_val.to(device, non_blocking=True)

            x_t = torch.randn_like(real_data)

            fake_sample_val = sample_from_model(pos_coeff, gen_diffusive_1, cond_data1_val, gen_diffusive_2,
                                                cond_data2_val,
                                                cond_data3_val,
                                                args.num_timesteps, x_t, T, args)

            # diffusion steps
            fake_sample_val = to_range_0_1(fake_sample_val)
            fake_sample_val = fake_sample_val / fake_sample_val.mean()
            real_data_val = to_range_0_1(real_data_val)
            real_data_val = real_data_val / real_data_val.mean()

            fake_sample_val_np = fake_sample_val.detach().cpu().numpy()
            real_data_val_np = real_data_val.detach().cpu().numpy()
            val_l1_loss[0, epoch, iteration] = abs(fake_sample_val_np - real_data_val_np).mean()

            val_psnr_values[0, epoch, iteration] = psnr(real_data_val_np, fake_sample_val_np, data_range=real_data_val_np.max())

        # reduce/log val metrics (master only)
        val_psnr_mean = float(np.nanmean(val_psnr_values[0, epoch, :]))
        val_l1_mean = float(np.nanmean(val_l1_loss[0, epoch, :]))
        if is_master:
            print(val_psnr_mean)
            print(val_l1_mean)
            _wandb_log({
                "metric/val/psnr_mean": val_psnr_mean,
                "metric/val/l1_mean": val_l1_mean,
                "epoch": epoch,
            }, step=global_step)

        np.save('{}/val_l1_loss.npy'.format(exp_path), val_l1_loss)
        np.save('{}/val_psnr_values.npy'.format(exp_path), val_psnr_values)
        # -------------------------------------------------------------

    # finish W&B
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
    cleanup()


def cleanup():
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
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, train_mudiff, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:

        init_processes(0, size, train_mudiff, args)
