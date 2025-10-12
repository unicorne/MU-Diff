from pyexpat import model
import sys
path_to_pip_installs = "/tmp/test_env"
if path_to_pip_installs not in sys.path:
    sys.path.insert(0, path_to_pip_installs)
from dataset.dataset_dixon import CreateDatasetSynthesis_with_masks
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import monai
import math
from monai.metrics import DiceMetric
from monai.networks.blocks import Warp 
from monai.losses import LocalNormalizedCrossCorrelationLoss, BendingEnergyLoss

from notebooks.reg_u_net import RegistrationUNet, SpatialTransformer

SPATIAL_DIMS = 2
# Let's assume you have 3 moving channels (e.g., T1, BOLD, Diffusion)
# and 1 fixed channel.
MOVING_CHANNELS = 3
FIXED_CHANNELS = 1
IN_CHANNELS = MOVING_CHANNELS + FIXED_CHANNELS # Total input channels for U-Net
OUT_CHANNELS = SPATIAL_DIMS # U-Net outputs a 2-channel DDF for 2D images



def validate_model(model, loader, device, spatial_transformer, similarity_loss, smoothing_loss, lambda_regularization):
    """Run a validation epoch with the updated model that only predicts flows."""
    model.eval()  # Set model to evaluation mode
    val_loss = 0
    with torch.no_grad():  # Do not calculate gradients
        for x1, x2, x3, x4, m1, m2, m3, m4 in loader:
            # 1. Prepare moving and fixed images
            moving_images = torch.cat([x1, x2, x3], dim=1).to(device)
            fixed_image = x4.to(device)

            # 2. Create the single input tensor for the U-Net
            model_input = torch.cat([moving_images, fixed_image], dim=1)
            
            # 3. Get the tuple of three flow fields from the model
            flow1, flow2, flow3 = model(model_input)

            # 4. Explicitly warp each moving image with its corresponding flow
            moving1, moving2, moving3 = torch.chunk(moving_images, 3, dim=1)
            warped1 = spatial_transformer(moving1, flow1)
            warped2 = spatial_transformer(moving2, flow2)
            warped3 = spatial_transformer(moving3, flow3)
            warped_images = torch.cat([warped1, warped2, warped3], dim=1)

            # 5. Calculate similarity loss (this logic remains the same)
            loss_sim_1 = similarity_loss(warped_images[:, 0:1, ...], fixed_image)
            loss_sim_2 = similarity_loss(warped_images[:, 1:2, ...], fixed_image)
            loss_sim_3 = similarity_loss(warped_images[:, 2:3, ...], fixed_image)
            loss_sim = (loss_sim_1 + loss_sim_2 + loss_sim_3) / 3

            # 6. Calculate smoothing loss for each DDF and average them
            loss_reg = (smoothing_loss(flow1) + smoothing_loss(flow2) + smoothing_loss(flow3)) / 3
            
            total_loss = loss_sim + lambda_regularization * loss_reg
            val_loss += total_loss.item()
    
    avg_loss = val_loss / len(loader)
    return avg_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RegistrationUNet().to(device)
    spatial_transformer = SpatialTransformer(size=(256, 256)).to(device) 
    similarity_loss = monai.losses.LocalNormalizedCrossCorrelationLoss(spatial_dims=2)
    smoothing_loss = monai.losses.BendingEnergyLoss()
    lambda_weight = 0.5
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lambda_regularization = 0.5
    num_epochs = 30
    best_val_loss = float('inf')
    input_path = "data/my_data2"
    batch_size = 16
    dataset = CreateDatasetSynthesis_with_masks(phase="train", input_path=input_path)
    dataset_val = CreateDatasetSynthesis_with_masks(phase="val", input_path=input_path)

    data_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=1,
                                                pin_memory=False,
                                                drop_last=True)

    data_loader_val = torch.utils.data.DataLoader(dataset_val,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    num_workers=1,
                                                    pin_memory=False,
                                                    drop_last=True)

    
    for epoch in range(num_epochs):
        print(f"--- Epoch {epoch+1}/{num_epochs} ---")
        model.train() # Set model to training mode
        train_loss = 0

        for iteration, (x1, x2, x3, x4, m1, m2, m3, m4) in enumerate(data_loader):
            moving_images = torch.cat([x1, x2, x3], dim=1).to(device)
            fixed_image = x4.to(device)
            optimizer.zero_grad()
            # Create the single input tensor for the model
            model_input = torch.cat([moving_images, fixed_image], dim=1)
            
            # Get the tuple of flow fields from the model
            flow1, flow2, flow3 = model(model_input)
            
            # Apply the transformation explicitly using the transformer
            moving1, moving2, moving3 = torch.chunk(moving_images, 3, dim=1)
            warped1 = spatial_transformer(moving1, flow1)
            warped2 = spatial_transformer(moving2, flow2)
            warped3 = spatial_transformer(moving3, flow3)
            warped_images = torch.cat([warped1, warped2, warped3], dim=1)
            # --- Loss calculation ---
            # Similarity loss (this part is the same as your code)
            loss_sim_1 = similarity_loss(warped_images[:, 0:1, ...], fixed_image)
            loss_sim_2 = similarity_loss(warped_images[:, 1:2, ...], fixed_image)
            loss_sim_3 = similarity_loss(warped_images[:, 2:3, ...], fixed_image)
            loss_sim = (loss_sim_1 + loss_sim_2 + loss_sim_3) / 3
            loss_reg = (smoothing_loss(flow1) + smoothing_loss(flow2) + smoothing_loss(flow3)) / 3
            total_loss = loss_sim + lambda_regularization * loss_reg
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()
            print(f"Iteration {iteration+1} Loss: {total_loss.item():.4f}", end='\r')

        avg_train_loss = train_loss / len(data_loader)
        avg_val_loss = validate_model(
                        model, 
                        data_loader_val, 
                        device, 
                        spatial_transformer, 
                        similarity_loss, 
                        smoothing_loss, 
                        lambda_regularization
                        )
        
        print(f"Epoch {epoch+1} Avg Train Loss: {avg_train_loss:.4f}, Avg Validation Loss: {avg_val_loss:.4f}")

        # Save the model if validation loss has improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_registration_model2.pth")
            print("   -> Saved new best model.")

    print("\n--- Training finished ---")

if __name__ == "__main__":
    main()