import sys
path_to_pip_installs = "/tmp/test_env"
if path_to_pip_installs not in sys.path:
    sys.path.insert(0, path_to_pip_installs)

import torch
import torch.nn as nn
import torch.nn.functional as F
import monai
from monai.metrics import DiceMetric
from monai.losses import LocalNormalizedCrossCorrelationLoss, BendingEnergyLoss, DiceLoss

# NEW: your updated dataset
from dataset.dataset_dixon import CreateDatasetSynthetic

# NEW: updated net that outputs ONE flow
from notebooks.reg_u_net_single import RegistrationUNet, SpatialTransformer

SPATIAL_DIMS = 2
IMAGE_SIZE = (256, 256)  # must match your data (used by SpatialTransformer grid)

# --- Utility: Dice (scalar) for a batch of binary masks ---
@torch.no_grad()
def batch_dice(pred, target, eps: float = 1e-6):
    """
    pred, target: (B, 1, H, W) float tensors in {0,1} (pred will be thresholded at 0.5)
    returns: mean Dice over batch (python float)
    """
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + eps) / (union + eps)
    return dice.mean().item()

@torch.no_grad()
def validate_model(model, loader, device, img_st, mask_st, similarity_loss, smoothing_loss, lambda_regularization):
    """Validation epoch: loss and per-mask Dice (m1->mt, m2->mt, m3->mt)."""
    model.eval()
    val_loss = 0.0

    # Accumulators for Dice (sum over batches weighted by batch size)
    dice_m1_sum = 0.0
    dice_m2_sum = 0.0
    dice_m3_sum = 0.0
    n_samples = 0

    for x_synth, x_tgt, m1, m2, m3, m_tgt in loader:
        x_synth = x_synth.to(device)  # (B,1,H,W)
        x_tgt   = x_tgt.to(device)    # (B,1,H,W)
        m1      = m1.to(device)
        m2      = m2.to(device)
        m3      = m3.to(device)
        m_tgt   = m_tgt.to(device)

        # Input to U-Net: concat synthetic (moving) + target (fixed)
        model_input = torch.cat([x_synth, x_tgt], dim=1)  # (B,2,H,W)
        flow = model(model_input)                         # (B,2,H,W)

        # Warp images & masks
        x_synth_warped = img_st(x_synth, flow)           # bilinear
        m1_w = mask_st(m1, flow)                          # nearest
        m2_w = mask_st(m2, flow)
        m3_w = mask_st(m3, flow)

        # Loss
        loss_sim = similarity_loss(x_synth_warped, x_tgt)
        loss_reg = smoothing_loss(flow)
        total_loss = loss_sim + lambda_regularization * loss_reg
        val_loss += total_loss.item()

        # Dice per mask vs target
        # Ensure single-channel shape and same dtype
        bsz = x_synth.shape[0]
        dice_m1_sum += batch_dice(m1_w, m_tgt) * bsz
        dice_m2_sum += batch_dice(m2_w, m_tgt) * bsz
        dice_m3_sum += batch_dice(m3_w, m_tgt) * bsz
        n_samples += bsz

    avg_loss = val_loss / len(loader)
    dice_m1 = dice_m1_sum / max(1, n_samples)
    dice_m2 = dice_m2_sum / max(1, n_samples)
    dice_m3 = dice_m3_sum / max(1, n_samples)
    return avg_loss, (dice_m1, dice_m2, dice_m3)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Model & transformers ---
    model = RegistrationUNet(in_channels=2).to(device)  # 2 channels: synth + target
    img_st  = SpatialTransformer(size=IMAGE_SIZE, mode='bilinear').to(device)  # for images
    mask_st = SpatialTransformer(size=IMAGE_SIZE, mode='nearest').to(device)   # for masks

    # --- Losses & optimizer ---
    similarity_loss = LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size=75)
    smoothing_loss  = BendingEnergyLoss()
    dice_loss_fn    = DiceLoss(include_background=True, reduction="mean", sigmoid=True)
    lambda_regularization = 1.0
    lambda_dice_m1 = 1.0
    lambda_dice_m2 = 1.0
    lambda_dice_m3 = 1.0
    lambda_dice = 10.0

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)

    # --- Data ---
    input_path = "data/my_data2"
    input_path_synth = "data/synthetic/exp_brats2"  # (not used here, but could be different)
    batch_size = 16
    num_workers = 1

    train_ds = CreateDatasetSynthetic(phase="train", input_path_org=input_path, input_path_synthetic=input_path_synth)
    val_ds   = CreateDatasetSynthetic(phase="val",   input_path_org=input_path, input_path_synthetic=input_path_synth)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False, drop_last=True
    )

    # --- Training ---
    num_epochs = 15
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"--- Epoch {epoch+1}/{num_epochs} ---")
        model.train()
        train_loss = 0.0

        for it, (x_synth, x_tgt, m1, m2, m3, m_tgt) in enumerate(train_loader):
            x_synth = x_synth.to(device)
            x_tgt   = x_tgt.to(device)
            m1      = m1.to(device)
            m2      = m2.to(device)
            m3      = m3.to(device)
            m_tgt   = m_tgt.to(device)

            # Forward
            model_input = torch.cat([x_synth, x_tgt], dim=1)  # (B,2,H,W)
            flow = model(model_input)                         # (B,2,H,W)

            # Warp moving image
            x_synth_warped = img_st(x_synth, flow)

            # Loss
            loss_sim  = similarity_loss(x_synth_warped, x_tgt)           # scalar tensor
            loss_reg  = smoothing_loss(flow)                             # scalar tensor
            loss_d1   = dice_loss_fn(m1, m_tgt)                          # scalar tensor
            loss_d2   = dice_loss_fn(m2, m_tgt)                          # scalar tensor
            loss_d3   = dice_loss_fn(m3, m_tgt)                          # scalar tensor
            loss_dice = (lambda_dice_m1 * loss_d1 + lambda_dice_m2 * loss_d2 + lambda_dice_m3 * loss_d3) / 3

            total_loss = loss_sim + lambda_regularization * loss_reg + lambda_dice * loss_dice
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            print(f"Iteration {it+1} Loss: {total_loss.item():.4f}", end="\r")

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation (also computes per-mask Dice) ---
        avg_val_loss, (dice_m1, dice_m2, dice_m3) = validate_model(
            model, val_loader, device, img_st, mask_st, similarity_loss, smoothing_loss, lambda_regularization
        )

        print(f"\nEpoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"  Dice m1→mt: {dice_m1:.4f}")
        print(f"  Dice m2→mt: {dice_m2:.4f}")
        print(f"  Dice m3→mt: {dice_m3:.4f}")

        # Save best by val loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_registration_model_synth2target.pth")
            print("   -> Saved new best model.")

    print("\n--- Training finished ---")

if __name__ == "__main__":
    main()
