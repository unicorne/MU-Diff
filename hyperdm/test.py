import sys
path_to_pip_installs = "/tmp/test_env"
if path_to_pip_installs not in sys.path:
    sys.path.insert(0, path_to_pip_installs)

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from matplotlib.backends.backend_pdf import PdfPages

from backbones.ncsnpp_generator_adagn_feat import NCSNpp
from hyperdm.hyperdm import HyperDM_MU_Diff_Single
from train_utils import (
    Posterior_Coefficients,
    Diffusion_Coefficients,
)
from dataset.dataset_dixon import CreateDatasetSynthesis

def parse_arguments_hyperdm_single():
    parser = argparse.ArgumentParser('mudiff-hyperdm-single-test parameters')
    # --- NEW: Added argument for number of test images ---
    parser.add_argument('--num_test_images', type=int, default=None, help="Number of test images to process.")

    # (The rest of the arguments are the same as before)
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--exp', default='mudiff-hyperdm-single-exp')
    parser.add_argument('--input_path', default='/data/BRATS/')
    parser.add_argument('--output_path', default='results/')
    parser.add_argument('--results_path', default='results/')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--num_channels', type=int, default=1)
    parser.add_argument('--num_timesteps', type=int, default=4)
    parser.add_argument('--beta_min', type=float, default=0.1)
    parser.add_argument('--beta_max', type=float, default=20.)
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
    parser.add_argument("--M", type=int, default=20)
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--hyper_net_dims", type=int, nargs="+", default=[32, 64], help="Dimensions for the hyper-network's MLP.")
    parser.add_argument('--checkpoint', type=str, default="content.pth")
    parser.add_argument('--output_file_base', type=str, default="hyperdm_mudiff_full_analysis", help="Base name for the output PDF file.")
    
    args = parser.parse_args()
    return args

def generate_plots_for_pdf(pdf_pages, index, real_data, mean_prediction, aleatoric_uncertainty, epistemic_uncertainty):
    """Generates a figure and saves it to the PDF object."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f'HyperDM MU-Diff Uncertainty Analysis - Image {index}', fontsize=16)

    to_range_0_1 = lambda x: (x + 1.) / 2.

    # --- Ground Truth and Mean Prediction ---
    axs[0, 0].imshow(to_range_0_1(real_data).squeeze().cpu().numpy(), cmap='gray')
    axs[0, 0].set_title("Ground Truth")
    axs[0, 0].axis('off')

    axs[0, 1].imshow(to_range_0_1(mean_prediction).squeeze().cpu().numpy(), cmap='gray')
    axs[0, 1].set_title("Mean Prediction")
    axs[0, 1].axis('off')

    # --- Uncertainty Maps ---
    im_au = axs[1, 0].imshow(aleatoric_uncertainty.squeeze().cpu().numpy(), cmap='jet')
    axs[1, 0].set_title("Aleatoric Uncertainty (AU)")
    axs[1, 0].axis('off')
    fig.colorbar(im_au, ax=axs[1, 0], fraction=0.046, pad=0.04)

    im_eu = axs[1, 1].imshow(epistemic_uncertainty.squeeze().cpu().numpy(), cmap='jet')
    axs[1, 1].set_title("Epistemic Uncertainty (EU)")
    axs[1, 1].axis('off')
    fig.colorbar(im_eu, ax=axs[1, 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf_pages.savefig(fig)
    plt.close(fig)

def generate_summary_plot(pdf_pages, all_real, all_preds, all_au, all_eu):
    """Generates a summary plot and saves it to the PDF object."""
    mean_real = torch.mean(torch.stack(all_real), dim=0)
    mean_preds = torch.mean(torch.stack(all_preds), dim=0)
    mean_au = torch.mean(torch.stack(all_au), dim=0)
    mean_eu = torch.mean(torch.stack(all_eu), dim=0)

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('HyperDM MU-Diff Summary Statistics', fontsize=16)
    
    to_range_0_1 = lambda x: (x + 1.) / 2.

    # --- Average Ground Truth and Prediction ---
    axs[0, 0].imshow(to_range_0_1(mean_real).squeeze().cpu().numpy(), cmap='gray')
    axs[0, 0].set_title("Average Ground Truth")
    axs[0, 0].axis('off')

    axs[0, 1].imshow(to_range_0_1(mean_preds).squeeze().cpu().numpy(), cmap='gray')
    axs[0, 1].set_title("Average Mean Prediction")
    axs[0, 1].axis('off')

    # --- Average Uncertainty ---
    im_au = axs[1, 0].imshow(mean_au.squeeze().cpu().numpy(), cmap='jet')
    axs[1, 0].set_title("Mean Aleatoric Uncertainty (AU)")
    axs[1, 0].axis('off')
    fig.colorbar(im_au, ax=axs[1, 0], fraction=0.046, pad=0.04)

    im_eu = axs[1, 1].imshow(mean_eu.squeeze().cpu().numpy(), cmap='jet')
    axs[1, 1].set_title("Mean Epistemic Uncertainty (EU)")
    axs[1, 1].axis('off')
    fig.colorbar(im_eu, ax=axs[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf_pages.savefig(fig)
    plt.close(fig)

def era5_test(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    output_dir = os.path.join(args.output_path, args.exp)
    os.makedirs(output_dir, exist_ok=True)
    pdf_output_path = os.path.join(output_dir, f"{args.output_file_base}.pdf")
    
    with PdfPages(pdf_output_path) as pdf:
        # --- Load Model and Coefficients ---
        primary_net = NCSNpp(args).to(device)
        pos_coeff = Posterior_Coefficients(args, device)
        
        hyperdm_model = HyperDM_MU_Diff_Single(
            primary_net=primary_net,
            hyper_net_dims=args.hyper_net_dims,
            diffusion_args=args,
            pos_coeff=pos_coeff
        ).to(device)

        checkpoint_path = os.path.join(args.results_path, args.exp, args.checkpoint)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            state_dict = checkpoint['hyperdm_model_dict']
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            hyperdm_model.load_state_dict(new_state_dict)
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print(f"WARNING: Checkpoint not found at {checkpoint_path}. Using randomly initialized weights.")

        hyperdm_model.eval()

        # --- Load Data ---
        dataset = CreateDatasetSynthesis(phase="test", input_path=args.input_path)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
        
        all_real_data, all_mean_predictions, all_aleatoric, all_epistemic = [], [], [], []

        # --- Loop over test images ---
        if args.num_test_images is None:
            args.num_test_images = len(data_loader)
        else:
            args.num_test_images = min(args.num_test_images, len(data_loader))
        for i, (x1, x2, x3, real_data) in enumerate(tqdm(data_loader, desc="Processing Test Images")):
            if i >= args.num_test_images:
                break
            if i % 10 == 0:
                print(f"Processing image {i+1}/{args.num_test_images}")

            cond_data1, cond_data2, cond_data3, real_data = [d.to(device) for d in (x1, x2, x3, real_data)]
            x_init = torch.randn_like(real_data)

            epistemic_uncertainty, aleatoric_uncertainty, mean_prediction = hyperdm_model.get_mean_variance(
                M=args.M, N=args.N, cond1=cond_data1, cond2=cond_data2, cond3=cond_data3,
                x_init=x_init, device=device, progress=False
            )

            all_real_data.append(real_data.cpu())
            all_mean_predictions.append(mean_prediction.cpu())
            all_aleatoric.append(aleatoric_uncertainty.cpu())
            all_epistemic.append(epistemic_uncertainty.cpu())

            generate_plots_for_pdf(pdf, i, real_data, mean_prediction, aleatoric_uncertainty, epistemic_uncertainty)

        # --- Generate and save the summary plot ---
        if all_real_data:
            generate_summary_plot(pdf, all_real_data, all_mean_predictions, all_aleatoric, all_epistemic)
        else:
            print("No images were processed.")

    print(f"Saved comprehensive analysis to {pdf_output_path}")

    # --- Save uncertainty arrays ---
    if all_real_data:
        aleatoric_array = torch.stack(all_aleatoric).numpy()
        epistemic_array = torch.stack(all_epistemic).numpy()

        au_path = os.path.join(output_dir, "aleatoric_uncertainty.npy")
        eu_path = os.path.join(output_dir, "epistemic_uncertainty.npy")
        
        np.save(au_path, aleatoric_array)
        np.save(eu_path, epistemic_array)
        
        print(f"Saved aleatoric uncertainty array to {au_path}")
        print(f"Saved epistemic uncertainty array to {eu_path}")

if __name__ == "__main__":
    args = parse_arguments_hyperdm_single()
    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    era5_test(args)