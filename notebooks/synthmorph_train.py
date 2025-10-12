import sys
path_to_pip_installs = "/tmp/test_env"
if path_to_pip_installs not in sys.path:
    sys.path.insert(0, path_to_pip_installs)

import random
import argparse
import voxelmorph as vxm
import tqdm
import numpy as np
import torch
import neurite as ne
import matplotlib.pyplot as plt
from PIL import Image

# import SpatialTransformer
from voxelmorph.nn.modules import SpatialTransformer
import voxelmorph.nn.losses as vxm_losses

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter, zoom

def draw_perlin(
    out_shape,
    scales,
    min_std=0,
    max_std=1,
    dtype=torch.float32,
    seed=None,
    device=None
):
    """
    Generate Perlin-like noise in PyTorch by summing Gaussian noise at different scales.

    This function is a PyTorch translation of the provided TensorFlow version.
    It samples noise at different resolutions (defined by scales), upsamples it to the
    target output shape, and accumulates it.

    Parameters:
        out_shape (tuple or list): The desired output shape. For N-dimensional spatial
            data, this should have N+1 elements, with the last being the channel/feature
            dimension (e.g., [H, W, C]).
        scales (list or int): A list of relative resolutions for noise sampling. A scale
            of 2 means sampling at half the output resolution.
        min_std (float): Minimum standard deviation for the Gaussian noise.
        max_std (float): Maximum standard deviation for the Gaussian noise.
        dtype (torch.dtype): The data type for the output tensor.
        seed (int, optional): A seed for reproducible randomization.
        device (str or torch.device, optional): The device (e.g., 'cpu' or 'cuda')
            on which to create the tensor.
    """
    if seed is not None:
        torch.manual_seed(seed)

    out_shape = np.asarray(out_shape, dtype=np.int32)
    if np.isscalar(scales):
        scales = [scales]

    # Get the number of spatial dimensions (e.g., 2 for [H, W, C]).
    n_dims = len(out_shape) - 1

    # Initialize the output tensor with zeros.
    out = torch.zeros(*out_shape, dtype=dtype, device=device)

    for scale in scales:
        # 1. Calculate the shape of the low-resolution noise.
        sample_shape = np.ceil(out_shape[:-1] / scale)
        sample_shape = tuple(np.int32((*sample_shape, out_shape[-1])))

        # 2. Draw a random standard deviation.
        std = (torch.rand(1, dtype=dtype, device=device) * (max_std - min_std)) + min_std

        # 3. Generate the low-resolution Gaussian noise.
        gauss = torch.randn(*sample_shape, dtype=dtype, device=device) * std

        if scale == 1:
            # If scale is 1, the shape already matches, so no resize is needed.
            out += gauss
        else:
            # 4. Upsample the noise to the full output shape.
            
            # Reshape for torch.nn.functional.interpolate:
            # It expects (B, C, H, W, ...). We move the channel dim to the front
            # and add a batch dimension of 1.
            # e.g., from (H, W, C) to (1, C, H, W).
            gauss_reshaped = gauss.permute(n_dims, *range(n_dims)).unsqueeze(0)

            # Define the target spatial shape for upsampling.
            target_spatial_shape = tuple(out_shape[:-1])

            # Choose interpolation mode based on dimensionality.
            if n_dims == 1:
                mode = 'linear'
            elif n_dims == 2:
                mode = 'bilinear'
            elif n_dims == 3:
                mode = 'trilinear'
            else:
                # 'nearest' is a safe fallback for dimensions > 3.
                mode = 'nearest'

            # Perform the upsampling.
            upsampled = F.interpolate(
                gauss_reshaped,
                size=target_spatial_shape,
                mode=mode,
                align_corners=False if mode != 'nearest' else None
            )

            # Reshape back to the original layout:
            # e.g., from (1, C, H, W) to (H, W, C).
            upsampled_reshaped = upsampled.squeeze(0).permute(*range(1, n_dims + 1), 0)

            out += upsampled_reshaped

    return out


# =============================================================================
# == 1. MODEL AND HELPER DEFINITIONS
# =============================================================================

class ConvBlock(nn.Module):
    """A convolutional block consisting of a convolution and activation."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.activation(self.conv(x))

class Unet(nn.Module):
    """A U-Net architecture for image-to-image translation."""
    def __init__(self, in_channels=2, nb_features=None):
        super().__init__()
        enc_nf, dec_nf = nb_features
        
        self.pool = nn.MaxPool2d(2, 2)
        self.encoders = nn.ModuleList([
            ConvBlock(in_channels, enc_nf[0]),
            ConvBlock(enc_nf[0], enc_nf[1]),
            ConvBlock(enc_nf[1], enc_nf[2]),
            ConvBlock(enc_nf[2], enc_nf[3])
        ])
        
        self.upsamples = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Upsample(scale_factor=2, mode='nearest')
        ])
        
        self.decoders = nn.ModuleList([
            ConvBlock(enc_nf[3] + enc_nf[2], dec_nf[0]),
            ConvBlock(dec_nf[0] + enc_nf[1], dec_nf[1]),
            ConvBlock(dec_nf[1] + enc_nf[0], dec_nf[2]),
            # Final layers from the original SynthMorph demo
            ConvBlock(dec_nf[2], dec_nf[3]),
            ConvBlock(dec_nf[3], dec_nf[4]),
            ConvBlock(dec_nf[4], dec_nf[5]),
            ConvBlock(dec_nf[5], dec_nf[6]),
            ConvBlock(dec_nf[6], dec_nf[7])
        ])

    def forward(self, x):
        skip_connections = []
        
        x = self.encoders[0](x); skip_connections.append(x); x = self.pool(x)
        x = self.encoders[1](x); skip_connections.append(x); x = self.pool(x)
        x = self.encoders[2](x); skip_connections.append(x); x = self.pool(x)
        x = self.encoders[3](x)

        x = self.upsamples[0](x); x = torch.cat([x, skip_connections.pop()], dim=1); x = self.decoders[0](x)
        x = self.upsamples[1](x); x = torch.cat([x, skip_connections.pop()], dim=1); x = self.decoders[1](x)
        x = self.upsamples[2](x); x = torch.cat([x, skip_connections.pop()], dim=1); x = self.decoders[2](x)
        
        x = self.decoders[3](x)
        x = self.decoders[4](x)
        x = self.decoders[5](x)
        x = self.decoders[6](x)
        x = self.decoders[7](x)
                
        return x

class VecInt(nn.Module):
    """Integrates a stationary velocity field (SVF) using scaling and squaring."""
    def __init__(self, in_shape, n_steps=7):
        super().__init__()
        self.n_steps = n_steps
        self.scale = 1.0 / (2**self.n_steps)
        self.transformer = SpatialTransformer(in_shape)

    def forward(self, vel):
        flow = vel * self.scale
        for _ in range(self.n_steps):
            flow = flow + self.transformer(flow, flow)
        return flow

class VxmDense(nn.Module):
    """VoxelMorph network for unsupervised nonlinear registration."""
    def __init__(self, in_shape, nb_unet_features, int_steps=7, int_resolution=2):
        super().__init__()
        ndims = len(in_shape)
        self.int_resolution = int_resolution
        self.int_steps = int_steps

        self.unet = Unet(in_channels=2, nb_features=nb_unet_features)
        self.flow_conv = nn.Conv2d(nb_unet_features[-1][-1], ndims, kernel_size=3, padding=1)
        self.flow_conv.weight = nn.Parameter(torch.randn_like(self.flow_conv.weight) * 1e-5)
        self.flow_conv.bias = nn.Parameter(torch.zeros_like(self.flow_conv.bias))

        if self.int_steps > 0:
            down_shape = [dim // int_resolution for dim in in_shape]
            self.vec_int = VecInt(down_shape, self.int_steps)

        self.transformer = SpatialTransformer(in_shape)

    def forward(self, source, target):
        x = torch.cat([source, target], dim=1)
        unet_out = self.unet(x)
        svf = self.flow_conv(unet_out)
        
        pos_flow = svf
        if self.int_steps > 0:
            if self.int_resolution > 1:
                pos_flow = F.interpolate(svf, scale_factor=1./self.int_resolution, mode='bilinear', align_corners=True, recompute_scale_factor=True)
            
            pos_flow = self.vec_int(pos_flow)
            
            if self.int_resolution > 1:
                pos_flow = F.interpolate(pos_flow, scale_factor=self.int_resolution, mode='bilinear', align_corners=True, recompute_scale_factor=True)
        
        moved_image = self.transformer(source, pos_flow)
        return moved_image, pos_flow

class DiceLoss(nn.Module):
    """Dice loss for segmentation tasks."""
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_true, y_pred):
        assert y_true.shape == y_pred.shape
        dims = tuple(range(2, len(y_true.shape)))
        
        intersect = torch.sum(y_pred * y_true, dims)
        cardinality = torch.sum(y_pred + y_true, dims)
        
        dice_score = (2. * intersect + self.smooth) / (cardinality + self.smooth)
        return 1. - torch.mean(dice_score)

def labels_to_image_pytorch(label_map, num_label, in_shape, warp_max=4, device='cpu'):
    """PyTorch implementation of the labels_to_image logic."""
    if isinstance(label_map, np.ndarray):
        label_map = torch.from_numpy(label_map).float()
        
    label_map = label_map.to(device)
    if len(label_map.shape) == 2:
        label_map = label_map.unsqueeze(0).unsqueeze(0)
    
    warp = draw_perlin(
        out_shape=(1, *in_shape, len(in_shape)),
        scales=(16, 32), max_std=warp_max,
    )
    warp_tensor = warp.float().permute(0, 3, 1, 2).to(device)

    transform_layer = SpatialTransformer(in_shape).to(device)
    warped_label_map = transform_layer(label_map, warp_tensor)

    warped_one_hot = F.one_hot(warped_label_map.long().squeeze(1), num_classes=num_label).permute(0, 3, 1, 2).float()

    rand_intensities = (torch.rand(1, num_label, 1, 1, device=device) * 0.8 + 0.1)
    image = torch.sum(warped_one_hot * rand_intensities, dim=1, keepdim=True)
    
    image_np = image.squeeze(0).squeeze(0).cpu().numpy()
    blurred_np = gaussian_filter(image_np, sigma=np.random.uniform(1, 3))
    image = torch.from_numpy(blurred_np).unsqueeze(0).unsqueeze(0).float().to(device)
    
    noise = torch.randn(*image.shape, device=device) * np.random.uniform(0, 0.05)
    image += noise
    
    return image, warped_one_hot

# =============================================================================
# == 2. TRAINING FUNCTION
# =============================================================================

def train(epochs, steps_per_epoch, model_save_path, lr, loss_mult):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- Data Generation Setup ---
    in_shape = (256, 256)
    num_label = 16
    num_maps = 40
    
    print("Generating base label maps for training...")
    label_maps = []
    transform_layer = SpatialTransformer(in_shape).to(device)
    for _ in tqdm.tqdm(range(num_maps)):
        im = draw_perlin(out_shape=(*in_shape, num_label), scales=(32, 64), max_std=1, device=device)
        warp = draw_perlin(out_shape=(*in_shape, len(in_shape)), scales=(16, 32, 64), max_std=16, device=device)
        im_tensor = im.float().permute(2, 0, 1).unsqueeze(0)
        warp_tensor = warp.float().permute(2, 0, 1).unsqueeze(0)
        im_warped = transform_layer(im_tensor, warp_tensor)
        lab = torch.argmax(im_warped, dim=1).squeeze().cpu().numpy() # Move to CPU for numpy
        label_maps.append(np.uint8(lab))
    
    # --- Model, Loss, and Optimizer Setup ---
    nb_unet_features = ([256] * 4, [256] * 8)
    model = VxmDense(in_shape, nb_unet_features, int_steps=7, int_resolution=2).to(device)
    
    similarity_loss = DiceLoss()
    regularization_loss = vxm_losses.Grad('l2', loss_mult=loss_mult)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    print(f'Starting training...')
    for epoch in range(epochs):
        epoch_loss = []
        model.train()
        for step in tqdm.tqdm(range(steps_per_epoch), desc=f'Epoch {epoch+1}/{epochs}'):
            optimizer.zero_grad()
            
            moving_label = random.choice(label_maps)
            fixed_image, fixed_map = labels_to_image_pytorch(moving_label, num_label, in_shape, device=device)
            moving_image, moving_map = labels_to_image_pytorch(moving_label, num_label, in_shape, device=device)
            
            moved_image, warp = model(moving_image, fixed_image)
            
            transform_layer_map = SpatialTransformer(in_shape, interpolation_mode='nearest').to(device)
            moved_map = transform_layer_map(moving_map, warp)

            sim_loss = similarity_loss(fixed_map, moved_map)
            reg_loss = regularization_loss.loss(None, warp)
            total_loss = sim_loss + reg_loss
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss.append(total_loss.item())
            
        avg_epoch_loss = np.mean(epoch_loss)
        losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_epoch_loss:.4f}")

    # --- Save Model and Loss Plot ---
    print(f"Training finished. Saving model to {model_save_path}")
    torch.save(model.state_dict(), model_save_path)

    plt.plot(range(1, epochs + 1), losses, '.-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.savefig('training_loss_curve.png')
    print("Saved training loss plot to training_loss_curve.png")

# =============================================================================
# == 3. EVALUATION FUNCTIONS
# =============================================================================

def load_and_conform(path, in_shape=(256, 256), is_mask=False):
    """Loads a 2D image, conforms it, and converts it to a tensor."""
    img = Image.open(path).convert('L')
    x = np.array(img, dtype=np.float32)

    if is_mask:
        x[x > 0] = 1
    else:
        x = (x - x.min()) / (x.max() - x.min())
        
    zoom_order = 0 if is_mask else 1
    x = zoom(x, [o / i for o, i in zip(in_shape, x.shape)], order=zoom_order)
    return torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float()

def calculate_dice_score(y_pred, y_true, smooth=1e-5):
    """Calculates the Dice Similarity Coefficient for binary masks."""
    y_pred = y_pred > 0.5; y_true = y_true > 0.5
    intersection = torch.sum(y_pred & y_true)
    union = torch.sum(y_pred) + torch.sum(y_true)
    return ((2. * intersection + smooth) / (union + smooth)).item()

def evaluate(model_path, moving_img_path, fixed_img_path, moving_mask_path, fixed_mask_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    in_shape = (256, 256)
    enc_nf = [16, 32, 32, 32]
    dec_nf = [32, 32, 32, 32, 32, 16, 16]
    nb_unet_features = (enc_nf, dec_nf)
    
    # --- Load Model ---
    model = VxmDense(in_shape, nb_unet_features, int_steps=7, int_resolution=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")

    # --- Load and Conform Data ---
    moving_image = load_and_conform(moving_img_path).to(device)
    fixed_image = load_and_conform(fixed_img_path).to(device)
    moving_mask = load_and_conform(moving_mask_path, is_mask=True).to(device)
    fixed_mask = load_and_conform(fixed_mask_path, is_mask=True).to(device)

    # --- Run Inference ---
    with torch.no_grad():
        moved_image, warp = model(moving_image, fixed_image)

    transform_layer = SpatialTransformer(in_shape, mode='nearest').to(device)
    warped_mask = transform_layer(moving_mask, warp)

    # --- Calculate and Print Dice ---
    dice_before = calculate_dice_score(moving_mask, fixed_mask)
    dice_after = calculate_dice_score(warped_mask, fixed_mask)
    print(f"\nDice Score Before Registration: {dice_before:.4f}")
    print(f"Dice Score After Registration:  {dice_after:.4f}")

    # --- Visualization ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    images = [
        moving_image, fixed_image, moved_image,
        moving_mask, fixed_mask, warped_mask
    ]
    titles = [
        'Moving Image', 'Fixed Image', 'Moved Image',
        'Moving Mask', 'Fixed Mask', f'Warped Mask (Dice: {dice_after:.3f})'
    ]
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].cpu().squeeze().numpy(), cmap='gray')
        ax.set_title(titles[i]); ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('evaluation_result.png')
    print("Saved evaluation visualization to evaluation_result.png")

# =============================================================================
# == 4. MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SynthMorph Training and Evaluation Script")
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Mode: 'train' or 'evaluate'")

    # --- Training Arguments ---
    train_parser = subparsers.add_parser('train', help="Train a new model")
    train_parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    train_parser.add_argument('--steps-per-epoch', type=int, default=100, help="Number of steps per epoch")
    train_parser.add_argument('--model-save-path', type=str, default='vxm_model.pth', help="Path to save the trained model")
    train_parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument('--loss-mult', type=float, default=0.02, help="Regularization loss multiplier")

    # --- Evaluation Arguments ---
    eval_parser = subparsers.add_parser('evaluate', help="Evaluate a trained model")
    eval_parser.add_argument('--model-path', type=str, required=True, help="Path to the saved model (.pth file)")
    eval_parser.add_argument('--moving-img', type=str, required=True, help="Path to the moving image")
    eval_parser.add_argument('--fixed-img', type=str, required=True, help="Path to the fixed image")
    eval_parser.add_argument('--moving-mask', type=str, required=True, help="Path to the moving mask")
    eval_parser.add_argument('--fixed-mask', type=str, required=True, help="Path to the fixed mask")

    args = parser.parse_args()

    if args.mode == 'train':
        train(args.epochs, args.steps_per_epoch, args.model_save_path, args.lr, args.loss_mult)
    elif args.mode == 'evaluate':
        evaluate(args.model_path, args.moving_img, args.fixed_img, args.moving_mask, args.fixed_mask)

# python notebooks/synthmorph_train.py train --model-save-path test.pth --epochs 1 --steps-per-epoch 1