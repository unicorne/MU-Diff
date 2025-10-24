import sys
path_to_pip_installs = "/tmp/test_env"
if path_to_pip_installs not in sys.path:
    sys.path.insert(0, path_to_pip_installs)


import torch
import torch.nn as nn
import torch.nn.functional as F
import monai
import numpy as np
from monai.metrics import DiceMetric

dice_metric = DiceMetric(include_background=True, reduction="mean")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################### Basic Loss Wrapper ###################
class Loss:
    def __init__(self, loss_fn, name):
        self.loss_fn = loss_fn
        self.name = name

    def compute_loss(self, img1, img2):
        loss_value = self.loss_fn(img1, img2)
        return np.round(float(loss_value),4)
    
    def __call__(self, img1, img2):
        return self.compute_loss(img1, img2)
    
#################### Loss Functions ###################

################### Monai Losses ###################    
LNCC_loss = monai.losses.LocalNormalizedCrossCorrelationLoss(spatial_dims=2)
MI_loss = monai.losses.GlobalMutualInformationLoss()
SSIM_loss = monai.losses.ssim_loss.SSIMLoss(spatial_dims=2)

LNCC_loss_function = Loss(LNCC_loss, "LNCC_Monai")
MI_loss_function = Loss(MI_loss, "MI_Monai")
SSIM_loss_function = Loss(SSIM_loss, "SSIM_Monai")
    
################### DGIR Losses ###################
from losses_dgir import Loss_DGIR, NewLNCC_wrapper, NMI, MINDSSC, SSD, AdaptiveNCC, BlurredSSD, SquaredLNCC

NMI_loss_function = Loss_DGIR(NMI(), "NMI_Loss")
MINDSSC_loss_function = Loss_DGIR(MINDSSC(), "MIND_SSC_Loss")
SSD_loss_function = Loss_DGIR(SSD(), "SSD_Loss")
AdaptiveNCC_loss_function = Loss_DGIR(AdaptiveNCC(), "AdaptiveNCC_Loss")
BlurredSSD_loss_function = Loss_DGIR(BlurredSSD(), "Blurred_SSD_Loss")
SquaredLNCC_loss_function = Loss_DGIR(SquaredLNCC(sigma=4), "Squared_LNCC_Loss")


model_path = "/home/students/studweilc1/MU-Diff/losses/guided_diffusion/256x256_diffusion_uncond.pt"
NewLNCC_loss_function = NewLNCC_wrapper(model_path=model_path, sigma=4, device=device)

################ Other Losses ###################
from skimage.metrics import peak_signal_noise_ratio as psnr

def psnr_wrapper(img1, img2):
    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()
    psnr_value = psnr(img1_np, img2_np, data_range=img2_np.max() - img2_np.min())
    return np.round(float(psnr_value),4)

PSNR_loss_function = Loss(psnr_wrapper, "PSNR Loss")


################ Loss Dictionary ################
loss_dict = {
    "LNCC_Monai": LNCC_loss_function,
    "MI_Monai": MI_loss_function,
    "SSIM_Monai": SSIM_loss_function,
    "NMI_Loss": NMI_loss_function,
    "MIND_SSC_Loss": MINDSSC_loss_function,
    "SSD_Loss": SSD_loss_function,
    "Blurred_SSD_Loss": BlurredSSD_loss_function,
    "Squared_LNCC_Loss": SquaredLNCC_loss_function,
    "New_LNCC_Loss": NewLNCC_loss_function,
    "PSNR_Loss": PSNR_loss_function
}


######################## Results #########################
import pandas as pd

class Results:
    def __init__(self, loss_dict, filename=None):
        self.loss_dict = loss_dict
        for loss_name in loss_dict.keys():
            setattr(self, loss_name, [])
        self.case = []
        self.data_contrast = []
        self.target_contrast = []
        self.dice_loss = []
        self.split = []
        if filename is None:
            self.filename = "loss_results.csv"
            self.df = None
        else:
            self.filename = filename
            self.df = pd.read_csv(filename)
            


    def compute_losses(self, x_data, x_target, x_mask, target_mask, case, data_contrast, target_contrast, split):
        for loss_name, loss_fn in self.loss_dict.items():
            loss = loss_fn(x_data, x_target)
            getattr(self, loss_name).append(loss)

        dice_loss = float(dice_metric(x_mask, target_mask).item())
        self.dice_loss.append(dice_loss)
        self.case.append(case)
        self.data_contrast.append(data_contrast)
        self.target_contrast.append(target_contrast)
        self.split.append(split)

    def build_dataframe(self):
        data = {
            "Dice_Loss": self.dice_loss,
            "Case": self.case,
            "Data_Contrast": self.data_contrast,
            "Target_Contrast": self.target_contrast,
            "Split": self.split
        }
        for loss_name in self.loss_dict.keys():
            data[loss_name] = self.__dict__[loss_name]
        df_new = pd.DataFrame(data)

        if self.df is not None:
            df = pd.concat([self.df, df_new], ignore_index=True)
        else:
            df = df_new
        self.df = df
        return df
    
    def save_dataframe(self, filename=None):
        if filename is None:
            filename = self.filename
        df = self.build_dataframe()
        df.to_csv(filename, index=False)




