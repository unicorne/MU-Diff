import sys
path_to_pip_installs = "/tmp/test_env"
if path_to_pip_installs not in sys.path:
    sys.path.insert(0, path_to_pip_installs)

import torch
import torch.nn.functional as F
import numpy as np
import math
import os

def to_homogeneous_batch(theta: torch.Tensor) -> torch.Tensor:
    """Converts a batch of 2x3 affine matrices to 3x3 homogeneous matrices."""
    B = theta.size(0)
    device = theta.device
    dtype = theta.dtype
    bottom_row = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
    bottom_row = bottom_row.unsqueeze(0).unsqueeze(0).expand(B, 1, 3)
    theta_hom = torch.cat([theta, bottom_row], dim=1)
    return theta_hom

def from_homogeneous_batch(theta_hom: torch.Tensor) -> torch.Tensor:
    """Converts a batch of 3x3 homogeneous matrices back to 2x3 affine matrices."""
    return theta_hom[:, :2, :3]

def combine_thetas_batch(theta1: torch.Tensor, theta2: torch.Tensor) -> torch.Tensor:
    """Combines two batches of affine transformations (theta2 @ theta1)."""
    theta1_hom = to_homogeneous_batch(theta1)
    theta2_hom = to_homogeneous_batch(theta2)
    combined_hom = torch.bmm(theta2_hom, theta1_hom)
    return from_homogeneous_batch(combined_hom)

def get_random_affine_batch(num_samples: int,
                            max_angle: float,
                            max_shift: float,
                            scale_range: tuple[float, float],
                            device: torch.device) -> torch.Tensor:
    """Generates a batch of random affine matrices (theta)."""
    dtype = torch.float32
    
    # 1. Random angles
    angles_deg = (torch.rand(num_samples, device=device, dtype=dtype) - 0.5) * 2 * max_angle
    angles_rad = torch.deg2rad(angles_deg)
    cos_a = torch.cos(angles_rad)
    sin_a = torch.sin(angles_rad)
    
    # 2. Random scales
    min_scale, max_scale = scale_range
    scales = torch.rand(num_samples, device=device, dtype=dtype) * (max_scale - min_scale) + min_scale
    
    # 3. Random shifts
    tx = (torch.rand(num_samples, device=device, dtype=dtype) - 0.5) * 2 * max_shift
    ty = (torch.rand(num_samples, device=device, dtype=dtype) - 0.5) * 2 * max_shift

    # --- Build transformation matrices ---
    zeros = torch.zeros_like(cos_a)
    ones = torch.ones_like(cos_a)

    # 1. Scaling matrix
    row1_s = torch.stack([scales, zeros, zeros], dim=1)
    row2_s = torch.stack([zeros, scales, zeros], dim=1)
    theta_scale = torch.stack([row1_s, row2_s], dim=1)
    
    # 2. Rotation matrix
    row1_r = torch.stack([cos_a, -sin_a, zeros], dim=1)
    row2_r = torch.stack([sin_a,  cos_a, zeros], dim=1)
    theta_rot = torch.stack([row1_r, row2_r], dim=1)

    # 3. Shift matrix
    row1_t = torch.stack([ones, zeros, tx], dim=1)
    row2_t = torch.stack([zeros, ones, ty], dim=1)
    theta_shift = torch.stack([row1_t, row2_t], dim=1)

    # Combine: 1. Scale, 2. Rotate, 3. Shift
    theta_final = combine_thetas_batch(theta_scale, theta_rot)
    theta_final = combine_thetas_batch(theta_final, theta_shift)
    
    return theta_final

def apply_random_affine_to_numpy_batch(
    image_array: np.ndarray,
    mask_array: np.ndarray,
    max_angle: float = 10.0,
    max_shift: float = 0.08,
    scale_range: tuple[float, float] = (0.95, 1.05)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies the *same* unique, random affine transformation to each
    image-mask pair in a NumPy batch.
    
    Uses 'bilinear' interpolation for images and 'nearest' for masks.
    
    Args:
        image_array: NumPy array of images, shape (num_samples, H, W).
        mask_array: NumPy array of masks, shape (num_samples, H, W).
        max_angle: Max rotation in degrees.
        max_shift: Max shift as a fraction of image size (e.g., 0.1 for 10%).
        scale_range: Tuple of min/max scale factor.
    
    Returns:
        A tuple of (transformed_image_array, transformed_mask_array).
    """
    
    # --- 0. Validation ---
    if image_array.ndim != 3:
        raise ValueError(f"Input image array must have 3 dimensions (num_samples, H, W), "
                         f"but got {image_array.ndim}")
    if image_array.shape != mask_array.shape:
        raise ValueError(f"Image array shape {image_array.shape} and mask array shape "
                         f"{mask_array.shape} must be identical.")
                         
    num_samples, H, W = image_array.shape
    
    # --- 1. Convert to PyTorch Tensor ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Add channel dim: (B, H, W) -> (B, 1, H, W)
    # Use float32 for images to support bilinear interpolation
    image_tensor = torch.from_numpy(image_array).unsqueeze(1).to(device, dtype=torch.float32)
    
    # Masks can be float or int, but float is fine for grid_sample
    mask_tensor = torch.from_numpy(mask_array).unsqueeze(1).to(device, dtype=torch.float32)

    # --- 2. Generate ONE set of Random Transformations for the batch ---
    theta_batch = get_random_affine_batch(
        num_samples=num_samples,
        max_angle=max_angle,
        max_shift=max_shift,
        scale_range=scale_range,
        device=device
    )

    # --- 3. Create ONE sampling grid ---
    # This grid is generated from theta_batch and will be applied to BOTH tensors
    grid = F.affine_grid(theta_batch, image_tensor.size(), align_corners=False)

    # --- 4. Apply Grid to BOTH Image and Mask ---
    
    # Apply to IMAGES using BILINEAR interpolation
    transformed_image_tensor = F.grid_sample(
        image_tensor,
        grid,
        mode='bilinear',
        padding_mode='zeros', # Fills image background with 0
        align_corners=False
    )
    
    # Apply to MASKS using NEAREST interpolation
    transformed_mask_tensor = F.grid_sample(
        mask_tensor,
        grid,
        mode='nearest',
        padding_mode='zeros', # Fills mask background with 0
        align_corners=False
    )

    # --- 5. Convert Back to NumPy ---
    # Squeeze channel dim: (B, 1, H, W) -> (B, H, W)
    transformed_images_squeezed = transformed_image_tensor.squeeze(1)
    transformed_masks_squeezed = transformed_mask_tensor.squeeze(1)
    
    # Move to CPU and convert to NumPy
    transformed_image_array = transformed_images_squeezed.cpu().numpy()
    transformed_mask_array = transformed_masks_squeezed.cpu().numpy()
    
    # Ensure mask data type matches original (e.g., if it was int)
    if np.issubdtype(mask_array.dtype, np.integer):
        transformed_mask_array = np.round(transformed_mask_array).astype(mask_array.dtype)
    
    return transformed_image_array, transformed_mask_array

def apply_specific_affine_to_numpy_batch(
    image_array: np.ndarray,
    mask_array: np.ndarray,
    angle_deg: float = 0.0,
    scale: float = 1.0,
    shift_x_frac: float = 0.0,
    shift_y_frac: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies the *same* specific affine transformation to each
    image-mask pair in a NumPy batch.
    
    Uses 'bilinear' interpolation for images and 'nearest' for masks.
    
    Args:
        image_array: NumPy array of images, shape (num_samples, H, W).
        mask_array: NumPy array of masks, shape (num_samples, H, W).
        angle_deg (float): Rotation angle in degrees. Default: 0.0.
        scale (float): Scale factor. 1.0 is no change. Default: 1.0.
        shift_x_frac (float): Horizontal shift as a fraction of image width.
                              Positive values shift image RIGHT. Default: 0.0.
        shift_y_frac (float): Vertical shift as a fraction of image height.
                              Positive values shift image DOWN. Default: 0.0.
    
    Returns:
        A tuple of (transformed_image_array, transformed_mask_array).
    """
    
    # --- 0. Validation ---
    if image_array.ndim != 3:
        raise ValueError(f"Input image array must have 3 dimensions (num_samples, H, W), "
                         f"but got {image_array.ndim}")
    if image_array.shape != mask_array.shape:
        raise ValueError(f"Image array shape {image_array.shape} and mask array shape "
                         f"{mask_array.shape} must be identical.")
                         
    num_samples, H, W = image_array.shape
    
    # --- 1. Convert to PyTorch Tensor ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    image_tensor = torch.from_numpy(image_array).unsqueeze(1).to(device, dtype=dtype)
    mask_tensor = torch.from_numpy(mask_array).unsqueeze(1).to(device, dtype=dtype)

    # --- 2. Generate ONE Specific Transformation Matrix ---
    
    # 1. Rotation
    angle_rad = torch.deg2rad(torch.tensor(angle_deg, dtype=dtype, device=device))
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)
    
    # 2. Scale
    s = torch.tensor(scale, dtype=dtype, device=device)
    
    # 3. Shift
    # Note: F.affine_grid uses an inverse mapping.
    # To make positive shift_x_frac shift the image RIGHT, we negate tx.
    # To make positive shift_y_frac shift the image DOWN, we negate ty.
    tx = torch.tensor(-shift_x_frac, dtype=dtype, device=device)
    ty = torch.tensor(-shift_y_frac, dtype=dtype, device=device)

    # --- Build component matrices (as a batch of 1) ---
    zeros = torch.zeros_like(cos_a)
    ones = torch.ones_like(cos_a)

    # 1. Scaling matrix (single 2x3, batched to [1, 2, 3])
    row1_s = torch.stack([s, zeros, zeros])
    row2_s = torch.stack([zeros, s, zeros])
    theta_scale = torch.stack([row1_s, row2_s]).unsqueeze(0) # Shape [1, 2, 3]

    # 2. Rotation matrix (single 2x3, batched to [1, 2, 3])
    row1_r = torch.stack([cos_a, -sin_a, zeros])
    row2_r = torch.stack([sin_a,  cos_a, zeros])
    theta_rot = torch.stack([row1_r, row2_r]).unsqueeze(0) # Shape [1, 2, 3]

    # 3. Shift matrix (single 2x3, batched to [1, 2, 3])
    row1_t = torch.stack([ones, zeros, tx])
    row2_t = torch.stack([zeros, ones, ty])
    theta_shift = torch.stack([row1_t, row2_t]).unsqueeze(0) # Shape [1, 2, 3]

    # Combine: 1. Scale, 2. Rotate, 3. Shift
    theta_final_single = combine_thetas_batch(theta_scale, theta_rot)
    theta_final_single = combine_thetas_batch(theta_final_single, theta_shift)
    # theta_final_single has shape [1, 2, 3]
    
    # --- 3. Create sampling grid ---
    # Expand the single transformation matrix to apply to the whole batch
    theta_batch = theta_final_single.expand(num_samples, 2, 3)
    
    grid = F.affine_grid(theta_batch, image_tensor.size(), align_corners=False)

    # --- 4. Apply Grid to BOTH Image and Mask ---
    
    # Apply to IMAGES using BILINEAR interpolation
    transformed_image_tensor = F.grid_sample(
        image_tensor,
        grid,
        mode='bilinear',
        padding_mode='zeros', # Fills image background with 0
        align_corners=False
    )
    
    # Apply to MASKS using NEAREST interpolation
    transformed_mask_tensor = F.grid_sample(
        mask_tensor,
        grid,
        mode='nearest',
        padding_mode='zeros', # Fills mask background with 0
        align_corners=False
    )

    # --- 5. Convert Back to NumPy ---
    transformed_images_squeezed = transformed_image_tensor.squeeze(1)
    transformed_masks_squeezed = transformed_mask_tensor.squeeze(1)
    
    transformed_image_array = transformed_images_squeezed.cpu().numpy()
    transformed_mask_array = transformed_masks_squeezed.cpu().numpy()
    
    if np.issubdtype(mask_array.dtype, np.integer):
        transformed_mask_array = np.round(transformed_mask_array).astype(mask_array.dtype)
    
    return transformed_image_array, transformed_mask_array

def main():
    input_folder = "/home/students/studweilc1/MU-Diff/data/my_data3"
    contrasts = ["DIXON", "T1_mapping_fl2d", "BOLD", "Diffusion"]
    splits = ["train", "val", "test"]
    output_folder = "/home/students/studweilc1/MU-Diff/data/transformed_data"
    os.makedirs(output_folder, exist_ok=True)
    for split in splits:
        os.makedirs(f"{output_folder}/{split}", exist_ok=True)

    for split in splits:
        for contrast in contrasts:
            file_path = f"{input_folder}/{split}/{contrast}.npy"
            file_path_mask = f"{input_folder}/{split}/{contrast}_masks.npy"
            data = np.load(file_path)
            mask = np.load(file_path_mask)
            transformed_data, transformed_mask = apply_random_affine_to_numpy_batch(
                data,
                mask,
                max_angle=15.0,
                max_shift=0.1,
                scale_range=(0.9, 1.1)
            )
            save_path = f"{output_folder}/{split}/{contrast}.npy"
            save_path_mask = f"{output_folder}/{split}/{contrast}_masks.npy"
            np.save(save_path, transformed_data)
            np.save(save_path_mask, transformed_mask)
            print(f"Saved transformed data to {save_path}")
            print(f"Saved transformed mask to {save_path_mask}")

if __name__ == "__main__":
    main()
