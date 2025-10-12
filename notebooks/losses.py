#ssim, LocalNCC_MIND, GradientSim, FeatureNCC   

import torch
import torch.nn as nn
import torch.nn.functional as F

#################### Local NCC loss #####################################
# This section was already correctly handling the device, no changes needed.
def box_filter2d(x, k=3):
    """Fast local averaging (separable box) used for SSD smoothing."""
    pad = k // 2
    # This line correctly creates 'w' on the same device as 'x'
    w = torch.ones((x.shape[1], 1, k, k), device=x.device, dtype=x.dtype) / (k * k)
    return F.conv2d(x, w, padding=pad, groups=x.shape[1])

def mind_descriptor_2d(img, radius=1, dilations=(1,), smooth_k=3, eps=1e-6):
    """
    img: [B, 1, H, W]
    Returns: MIND features [B, C, H, W] with C = len(offsets) * len(dilations)
    """
    B, C, H, W = img.shape
    assert C == 1, "Expect single channel for raw intensities"

    base_offsets = [(1,0), (-1,0), (0,1), (0,-1)]
    feats = []

    mu = box_filter2d(img, k=smooth_k)
    var = box_filter2d((img - mu)**2, k=smooth_k)

    for d in dilations:
        for dy, dx in base_offsets:
            shifted = F.pad(img, (d, d, d, d), mode='reflect')
            shifted = shifted[:, :, d+dy*d: d+dy*d+H, d+dx*d: d+dx*d+W]
            ssd = box_filter2d((img - shifted)**2, k=smooth_k)
            feat = torch.exp(-ssd / (var + eps))
            feats.append(feat)

    f = torch.cat(feats, dim=1)
    f = f / (torch.sqrt((f**2).sum(dim=1, keepdim=True)) + eps)
    return f

def local_ncc_loss(pred, target, win=9, epsilon=1e-5):
    """
    A numerically stable local NCC loss implementation.
    """
    pool = nn.AvgPool2d(kernel_size=win, stride=1, padding=win // 2)
    mu_pred = pool(pred)
    mu_target = pool(target)
    var_pred = pool(pred**2) - mu_pred**2
    var_target = pool(target**2) - mu_target**2
    cross_corr = pool(pred * target) - mu_pred * mu_target
    ncc_numerator = cross_corr
    ncc_denominator = torch.sqrt(var_pred * var_target + epsilon)
    ncc = ncc_numerator / ncc_denominator
    return 1 - torch.mean(ncc)

def local_ncc_loss_complete(x1, x2):
    mind_pred   = mind_descriptor_2d(x1, dilations=(1,2), smooth_k=5)
    mind_target = mind_descriptor_2d(x2, dilations=(1,2), smooth_k=5)
    mind_ncc_loss_val = local_ncc_loss(mind_pred, mind_target, win=9)
    return mind_ncc_loss_val

#################### Gradient loss #####################################

class GradientSimilarityLoss(torch.nn.Module):
    """
    Calculates the L2 loss (MSE) on the spatial gradients of the images.
    """
    def __init__(self):
        super().__init__()
        # Define Sobel filters. They are created on the CPU by default.
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        # register_buffer allows these tensors to be moved with .to(device)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, prediction, target):
        # **FIX:** Ensure the Sobel filters are on the same device as the input tensor.
        # This makes the module device-agnostic.
        sobel_x_on_device = self.sobel_x.to(prediction.device)
        sobel_y_on_device = self.sobel_y.to(prediction.device)

        # Calculate gradients for the prediction (warped image)
        pred_grad_x = F.conv2d(prediction, sobel_x_on_device, padding='same')
        pred_grad_y = F.conv2d(prediction, sobel_y_on_device, padding='same')

        # Calculate gradients for the target (fixed image)
        target_grad_x = F.conv2d(target, sobel_x_on_device, padding='same')
        target_grad_y = F.conv2d(target, sobel_y_on_device, padding='same')

        # Calculate the MSE loss between the gradient maps
        return self.mse_loss(pred_grad_x, target_grad_x) + self.mse_loss(pred_grad_y, target_grad_y)


############################# Feature Space ################################
class SmallEncoder2D(nn.Module):
    def __init__(self, in_ch=1, feat_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.InstanceNorm2d(16), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 3, padding=1), nn.InstanceNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, feat_ch, 3, padding=1), nn.InstanceNorm2d(feat_ch)
        )
    def forward(self, x):
        f = self.net(x)
        f = f / (torch.sqrt((f**2).sum(dim=1, keepdim=True)) + 1e-6)
        return f

# Example similarity on embeddings:
def feature_space_ncc_loss(encoder, x, y, win=9):
    fx = encoder(x)
    fy = encoder(y)
    return local_ncc_loss(fx, fy, win=win)

# **FIX:** The encoder should be passed as an argument, not created here.
def feature_space_ncc_loss_complete(x1, x2):
    """
    Calculates feature-space NCC loss.
    Args:
        encoder (nn.Module): The pre-initialized and device-placed encoder model.
        x1 (torch.Tensor): The first image tensor.
        x2 (torch.Tensor): The second image tensor.
    """
    encoder = SmallEncoder2D(in_ch=1, feat_ch=32)
    feat_ncc_loss_val = feature_space_ncc_loss(encoder, x1, x2, win=9)
    return feat_ncc_loss_val
