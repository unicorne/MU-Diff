import sys
path_to_pip_installs = "/tmp/test_env"
if path_to_pip_installs not in sys.path:
    sys.path.insert(0, path_to_pip_installs)


import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
from typing import List

# --- Assuming your model definitions are in these files ---
from backbones.ncsnpp_generator_adagn_feat import NCSNpp
from hyperdm.hyperdm import HyperDM_MU_Diff_Single 

# --- A simple MLP for the Hyper-network ---
class MLP(nn.Module):
    def __init__(self, layer_channels: List[int]):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(layer_channels) - 2):
            layers.append(nn.Linear(layer_channels[i], layer_channels[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_channels[-2], layer_channels[-1]))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

def pretrain_hypernetwork(args):
    """
    Pre-trains the MLP to output the weights of a pre-trained generator
    by loading them from a 'content.pth' checkpoint file.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load the pre-trained primary network (gen_diffusive_1) from the content.pth file
    primary_net = NCSNpp(args).to(device)
    
    try:
        # Load the entire checkpoint dictionary
        content_checkpoint = torch.load(args.content_path, map_location=device, weights_only=False)
        
        # Extract the state dictionary for the first generator
        gen1_state_dict = content_checkpoint['gen_diffusive_1_dict']
        
        # The MU-Diff checkpoint might be saved with a 'module.' prefix from DDP, so we strip it
        if any(key.startswith('module.') for key in gen1_state_dict.keys()):
             gen1_state_dict = {k.replace('module.', ''): v for k, v in gen1_state_dict.items()}

        primary_net.load_state_dict(gen1_state_dict)
        print(f"Successfully loaded pre-trained generator weights from '{args.content_path}'.")

    except FileNotFoundError:
        print(f"Error: Pre-trained content file not found at '{args.content_path}'")
        print("Please provide the correct path via the --content_path argument.")
        return
    except KeyError:
        print(f"Error: Key 'gen_diffusive_1_dict' not found in '{args.content_path}'.")
        print("Please ensure you are using a valid 'content.pth' file from MU-Diff training.")
        return

    # 2. Flatten the pre-trained weights to create the target vector
    target_weights = torch.cat([p.data.flatten() for p in primary_net.parameters()]).to(device)
    primary_params_count = target_weights.numel()
    
    print(f"Total parameters in primary net: {primary_params_count}")

    # 3. Initialize the Hyper-network (MLP)
    hyper_net_dims = args.hyper_net_dims + [primary_params_count]
    hyper_net_input_dim = hyper_net_dims[0]
    hyper_net = MLP(hyper_net_dims).to(device)
    
    optimizer = optim.Adam(hyper_net.parameters(), lr=args.pretrain_lr)
    loss_fn = nn.MSELoss()

    print("Starting hyper-network pre-training...")
    
    # 4. Pre-training loop
    for epoch in range(args.pretrain_epochs):
        optimizer.zero_grad()
        
        # Generate a random noise vector `z` as input
        z = torch.randn(1, hyper_net_input_dim, device=device)
        
        # The hyper-network outputs the predicted weights
        predicted_weights = hyper_net(z)
        
        # Calculate the loss between predicted and target weights
        loss = loss_fn(predicted_weights.squeeze(), target_weights)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            tqdm.write(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    print("Pre-training finished.")
    
    # 5. Save the pre-trained hyper-network weights
    save_path = 'hyperdm/pretrained_hypernet_exp.pth'
    torch.save(hyper_net.state_dict(), save_path)
    print(f"Pre-trained hyper-network saved to '{save_path}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('HyperNet Pre-training')
    
    # --- Path to the main checkpoint file ---
    parser.add_argument('--content_path', type=str, required=True, help="Path to the 'content.pth' file from MU-Diff training.")

    # --- Pre-training specific arguments ---
    parser.add_argument("--hyper_net_dims", type=int, nargs="+", default=[32, 64, 128, 256], help="Dimensions for the hyper-network's MLP.")
    parser.add_argument('--pretrain_epochs', type=int, default=2000, help="Number of epochs for pre-training.")
    parser.add_argument('--pretrain_lr', type=float, default=1e-4, help="Learning rate for pre-training.")

    parser.add_argument('--num_channels_dae', type=int, default=32)
    parser.add_argument('--n_mlp', type=int, default=3)
    parser.add_argument('--ch_mult', nargs='+', type=int, default=[1, 1, 2])
    parser.add_argument('--num_res_blocks', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--num_channels', type=int, default=1)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--not_use_tanh', action='store_true', default=False)
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

    args = parser.parse_args()
    
    pretrain_hypernetwork(args)
    # python -m hyperdm.pretrain_mlp --content_path results/exp_brats_hyperdm/content.pth --pretrain_epochs 100 --pretrain_lr 1e-4 --hyper_net_dims 32 64