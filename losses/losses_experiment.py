import sys
path_to_pip_installs = "/tmp/test_env"
if path_to_pip_installs not in sys.path:
    sys.path.insert(0, path_to_pip_installs)

path = "/home/students/studweilc1/MU-Diff/"
# add this to path for imports
if path not in sys.path:
    sys.path.append(path)

import numpy as np
import matplotlib.pyplot as plt
import torch
from dataset.dataset_dixon import CreateDatasetSynthesis_single_with_masks, CreateDatasetSynthesis_single_with_masks_affine


from losses import loss_dict, Results
def create_data(intput_path, split, contrast1, contrast2):
    dataset_val= CreateDatasetSynthesis_single_with_masks(phase=split, input_path=intput_path, contrast=contrast1, target_contrast=contrast2)
    data_loader = torch.utils.data.DataLoader(dataset_val,
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=1,
                                                pin_memory=False,
                                                drop_last=True)
    data_loader.requires_grad = False
    return data_loader

def create_data_affine(input_path1, input_path2, split, contrast1, contrast2):
    dataset = CreateDatasetSynthesis_single_with_masks_affine(phase=split, input_path1=input_path1, input_path2=input_path2, contrast=contrast1, target_contrast=contrast2)
    data_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=1,
                                                pin_memory=False,
                                                drop_last=True)
    data_loader.requires_grad = False
    return data_loader

def main():

    losses_csv_path = "losses/loss_results_experiment.csv"
    splits = ["test", "val", "train"]
    res = Results(loss_dict)

    
    ######### Case 1 Orginal ##########
    input_path = "/home/students/studweilc1/MU-Diff/data/my_data3"
    target_contrast = "DIXON"
    contrasts = ["T1_mapping_fl2d", "BOLD", "Diffusion"]
    case = "Orginal"
    print("Case:", case)

    for split in splits:
        print("Split:", split)
        for contrast1 in contrasts:
            print("Contrast:", contrast1)
            data_loader = create_data(intput_path=input_path, split=split, contrast1=contrast1, contrast2=target_contrast)
            for i, data in enumerate(data_loader):
                x_data, x_target, mask_data, mask_target = data
                res.compute_losses(x_data, x_target, mask_data, mask_target, case, contrast1, target_contrast, split)

    df = res.build_dataframe()
    res.save_dataframe(losses_csv_path)


    ######### Case 2 Affine ##########
    input_path = "/home/students/studweilc1/MU-Diff/data/transformed_data"
    case = "Affine"
    print("Case:", case)


    for split in splits:
        print("Split:", split)
        for contrast1 in contrasts:
            print("Contrast:", contrast1)
            data_loader = create_data(intput_path=input_path, split=split, contrast1=contrast1, contrast2=target_contrast)
            for i, data in enumerate(data_loader):
                x_data, x_target, mask_data, mask_target = data
                res.compute_losses(x_data, x_target, mask_data, mask_target, case, contrast1, target_contrast, split)

    df = res.build_dataframe()
    res.save_dataframe(losses_csv_path)

    ######### Case 3 Affine to Orginal ##########
    input_path1 = "/home/students/studweilc1/MU-Diff/data/transformed_data"
    input_path2 = "/home/students/studweilc1/MU-Diff/data/my_data3"
    contrast1 = "DIXON"
    contrast2 = "DIXON"
    case = "Affine_DIXON"

    print("Case:", case)
    for split in splits:
        print("Split:", split)
        data_loader = create_data_affine(input_path1=input_path1, input_path2=input_path2, split=split, contrast1=contrast1, contrast2=contrast2)
        for i, data in enumerate(data_loader):
            x_data, x_target, mask_data, mask_target = data
            res.compute_losses(x_data, x_target, mask_data, mask_target, case, contrast1, contrast2, split)

    df = res.build_dataframe()
    res.save_dataframe(losses_csv_path)

if __name__ == "__main__":
    main()



    

