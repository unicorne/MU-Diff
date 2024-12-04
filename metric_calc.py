import os
import numpy as np
import torchvision
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import cv2
import torch
def load_image(image_path, grayscale=True):
    """Load an image as grayscale or color."""
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def load_binary_mask(mask_path):
    """Load a binary mask image."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 127).astype(np.uint8)  # Ensure the mask is binary (0 or 1)
    return mask

def normalize(image):
    """Basic min-max scaler."""
    min_ = np.min(image)
    max_ = np.max(image)
    scale = max_ - min_
    image = (image - min_) / scale
    return image

def irm_min_max_preprocess(image, low_perc=1, high_perc=99):
    """Pre-processing to remove outliers and perform min-max scaling."""
    non_zeros = image > 0
    low, high = np.percentile(image[non_zeros], [low_perc, high_perc])
    image = np.clip(image, low, high)
    image = normalize(image)
    return image

def apply_mask(image, mask):
    """Apply a binary mask to an image."""
    return image * mask

def save_image(image,save_dir,iteration):

    image=torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    torchvision.utils.save_image(image, '{}/{}_samples_{}.jpg'.format(save_dir, 'test', iteration),
                                 normalize=True)


def mean_absolute_error(real_image, synthesized_image):
    absolute_diff = np.abs(real_image - synthesized_image)
    mae = np.mean(absolute_diff)
    return mae

def ssim_val(fake, real):
    dr = np.max([real.max(), fake.max()]) - np.min([real.min(), fake.min()])
    ssim_score = ssim(np.squeeze(fake), np.squeeze(real), data_range=dr)
    return ssim_score

def psnr_calc(fake, real):
    dr = np.max([fake.max(), real.max()]) - np.min([fake.min(), real.min()])
    psnr_val = psnr(real, fake, data_range=dr)
    return psnr_val
def visualize_images(real_image, synth_image, mask_image, image_name):
        """Visualize real, synthetic, and mask images side by side."""
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(real_image, cmap='gray')
        axes[0].set_title(f'Real Image - {image_name}')
        axes[0].axis('off')

        axes[1].imshow(synth_image, cmap='gray')
        axes[1].set_title(f'Synthetic Image - {image_name}')
        axes[1].axis('off')

        axes[2].imshow(mask_image, cmap='gray')
        axes[2].set_title(f'Mask Image - {image_name}')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()
def sort_key(image_name):
    # Extract the numerical part from the image name
    return int(image_name.split('_')[2].split('.')[0])
def process_images(real_dir, synth_dir, mask_dir, masked_synth_dir):
    min_height = 7  # Minimum height for SSIM calculation
    min_width = 7
    iteration=-1
    ssim_scores = []
    psnr_values = []
    mae_values = []

    heights=[]
    widths=[]


    # Loop through all images in the real image directory
    for image_name in sorted(os.listdir(real_dir), key=sort_key):
        iteration=iteration+1

        real_image_path = os.path.join(real_dir, image_name)
        synth_image_path = os.path.join(synth_dir, image_name)
        mask_image_path = os.path.join(mask_dir, image_name)

        # Load the real image, synthesized image, and mask as grayscale
        real_image = load_image(real_image_path, grayscale=True)
        synth_image = load_image(synth_image_path, grayscale=True)
        mask = load_binary_mask(mask_image_path)

        #add dilation in isles lesion synthesis evalulation
        # dilation_kernel = np.ones((3, 3), np.uint8)  # Kernel size can be adjusted for more or less dilation
        # mask = cv2.dilate(mask.astype(np.uint8), dilation_kernel, iterations=3)


        # Preprocess and apply the mask to the real and synthesized images
        real_normalized = irm_min_max_preprocess(real_image)
        real_normalized_p=real_normalized/real_normalized.mean()
        synth_normalized = irm_min_max_preprocess(synth_image)
        synth_normalized_p=synth_normalized/synth_normalized.mean()

        #skip slices without any lesion mask (in lesion evaluation)
        if not np.any(mask):
            # Skip to the next iteration if the mask is empty
            ssim_scores.append(np.nan)
            psnr_values.append(np.nan)
            mae_values.append(np.nan)
            continue


        #apply mask
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        row_min, row_max = np.where(rows)[0][[0, -1]]
        col_min, col_max = np.where(cols)[0][[0, -1]]

        # Step 2: Calculate the cropped dimensions
        cropped_height = row_max - row_min + 1
        cropped_width = col_max - col_min + 1
        heights.append(cropped_height)
        widths.append(cropped_width)


        # Step 3: Ensure the cropped region meets the minimum size requirement
        if cropped_height < min_height:
            # Adjust row_min and row_max to meet the minimum height
            row_min = max(0, row_min - (min_height - cropped_height) // 2)
            row_max = min(real_normalized.shape[0] - 1, row_max + (min_height - cropped_height + 1) // 2)

        if cropped_width < min_width:
            # Adjust col_min and col_max to meet the minimum width
            col_min = max(0, col_min - (min_width - cropped_width) // 2)
            col_max = min(real_normalized.shape[1] - 1, col_max + (min_width - cropped_width + 1) // 2)

        # Ensure bounds are within image dimensions
        row_min = max(row_min, 0)
        row_max = min(row_max, real_normalized.shape[0] - 1)
        col_min = max(col_min, 0)
        col_max = min(col_max, real_normalized.shape[1] - 1)

        # Step 4: Crop the bounding box region from the real, synthesized images, and the mask
        real_normalized_p = real_normalized_p[row_min:row_max + 1, col_min:col_max + 1]
        synth_normalized_p = synth_normalized_p[row_min:row_max + 1, col_min:col_max + 1]

        real_normalized = real_normalized[row_min:row_max + 1, col_min:col_max + 1]
        synth_normalized = synth_normalized[row_min:row_max + 1, col_min:col_max + 1]

        mask = mask[row_min:row_max + 1, col_min:col_max + 1]
        #
        # # Step 5: Apply the binary mask to the cropped real and synthesized images
        masked_real_p = np.multiply(real_normalized_p, mask)
        masked_synth_p = np.multiply(synth_normalized_p, mask)

        masked_real = np.multiply(real_normalized, mask)
        masked_synth = np.multiply(synth_normalized, mask)

        # visualize_images(real_image, masked_real_p, masked_synth_p, image_name)

        # Save the masked images to the specified directories
        # save_image(masked_synth_p, masked_synth_dir,iteration)


        # Calculate SSIM, PSNR, and MAE for the normalized masked images
        ssim_score = ssim_val(masked_synth_p, masked_real_p)
        psnr_value = psnr_calc(masked_synth_p, masked_real_p)
        mae_value = mean_absolute_error(masked_real, masked_synth)

        ssim_scores.append(ssim_score)
        psnr_values.append(psnr_value)
        mae_values.append(mae_value)

    return ssim_scores, psnr_values, mae_values
def print_statistics(metric_name, values):
    mean_value = np.nanmean(values)
    std_value = np.nanstd(values)
    print(f"{metric_name} - Mean: {mean_value:.4f}, Std: {std_value:.4f}")

# Example usage
real_images_folder = r'results\t1\real'
synth_images_folder = r'results\t1\mu-diff'
masks_folder = r'results\brain_mask'
masked_synth_dir = r'results\t1\masked_img_save_dir'




ssim_scores, psnr_values, mae_values = process_images(
    real_images_folder,
    synth_images_folder,
    masks_folder, masked_synth_dir

)

# Print the mean and standard deviation for SSIM, MAE, and PSNR
print_statistics("SSIM", ssim_scores)
print_statistics("PSNR", psnr_values)
print_statistics("MAE", mae_values)


np.save('{}/psnr_values.npy'.format(synth_images_folder), psnr_values)
np.save('{}/ssim_values.npy'.format(synth_images_folder), ssim_scores)
np.save('{}/mae_values.npy'.format(synth_images_folder), mae_values)


# np.save('{}/psnr_values_lesion.npy'.format(synth_images_folder), psnr_values)
# np.save('{}/ssim_values_lesion.npy'.format(synth_images_folder), ssim_scores)
# np.save('{}/mae_values_lesion.npy'.format(synth_images_folder), mae_values)




