import torch.utils.data
import numpy as np


def CreateDatasetSynthetic(phase, input_path_org, input_path_synthetic):
    target_data = input_path_org +"/"+ phase + "/DIXON.npy"
    data_fs_target = LoadDataSet(target_data)

    cond_masks1 = input_path_org +"/"+ phase + "/BOLD_masks.npy"
    data_fs_s1_masks = LoadDataSet(cond_masks1)

    cond_masks2 = input_path_org +"/"+ phase + "/Diffusion_masks.npy"
    data_fs_s2_masks = LoadDataSet(cond_masks2)

    cond_masks3 = input_path_org +"/"+ phase + "/T1_mapping_fl2d_masks.npy"
    data_fs_s3_masks = LoadDataSet(cond_masks3)

    target_masks = input_path_org +"/"+ phase + "/DIXON_masks.npy"
    data_fs_s4_masks = LoadDataSet(target_masks)

    synthetic_data = input_path_synthetic +"/"+ phase + "/synthetic.npy"
    data_fs_synthetic = LoadDataSet(synthetic_data)

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(data_fs_synthetic), torch.from_numpy(data_fs_target), torch.from_numpy(data_fs_s1_masks), torch.from_numpy(data_fs_s2_masks), torch.from_numpy(data_fs_s3_masks), torch.from_numpy(data_fs_s4_masks))
    return dataset


def CreateDatasetSynthesis(phase, input_path):
    cond_data1 = input_path +"/"+ phase + "/BOLD.npy"
    data_fs_s1 = LoadDataSet(cond_data1)

    cond_data2 = input_path +"/"+ phase + "/Diffusion.npy"
    data_fs_s2 = LoadDataSet(cond_data2)

    cond_data3 = input_path +"/"+ phase + "/T1_mapping_fl2d.npy"
    data_fs_s3 = LoadDataSet(cond_data3)

    target_data = input_path +"/"+ phase + "/DIXON.npy"
    data_fs_s4 = LoadDataSet(target_data)

    dataset = torch.utils.data.TensorDataset( torch.from_numpy(data_fs_s1), torch.from_numpy(data_fs_s2), torch.from_numpy(data_fs_s3), torch.from_numpy(data_fs_s4))
    return dataset

def CreateDatasetSynthesis_with_masks(phase, input_path):
    cond_data1 = input_path +"/"+ phase + "/BOLD.npy"
    data_fs_s1 = LoadDataSet(cond_data1)

    cond_data2 = input_path +"/"+ phase + "/Diffusion.npy"
    data_fs_s2 = LoadDataSet(cond_data2)

    cond_data3 = input_path +"/"+ phase + "/T1_mapping_fl2d.npy"
    data_fs_s3 = LoadDataSet(cond_data3)

    target_data = input_path +"/"+ phase + "/DIXON.npy"
    data_fs_s4 = LoadDataSet(target_data)

    cond_masks1 = input_path +"/"+ phase + "/BOLD_masks.npy"
    data_fs_s1_masks = LoadDataSet(cond_masks1)

    cond_masks2 = input_path +"/"+ phase + "/Diffusion_masks.npy"
    data_fs_s2_masks = LoadDataSet(cond_masks2)

    cond_masks3 = input_path +"/"+ phase + "/T1_mapping_fl2d_masks.npy"
    data_fs_s3_masks = LoadDataSet(cond_masks3)

    target_masks = input_path +"/"+ phase + "/DIXON_masks.npy"
    data_fs_s4_masks = LoadDataSet(target_masks)

    dataset = torch.utils.data.TensorDataset( torch.from_numpy(data_fs_s1), torch.from_numpy(data_fs_s2), torch.from_numpy(data_fs_s3), torch.from_numpy(data_fs_s4), torch.from_numpy(data_fs_s1_masks), torch.from_numpy(data_fs_s2_masks), torch.from_numpy(data_fs_s3_masks), torch.from_numpy(data_fs_s4_masks))
    return dataset

def CreateDatasetSynthesis_single(phase, input_path, contrast, target_contrast):
    cond_data = input_path +"/"+ phase + "/"+ contrast +".npy"
    data_fs_s = LoadDataSet(cond_data)

    target_data = input_path +"/"+ phase + "/"+ target_contrast +".npy"
    data_fs_t = LoadDataSet(target_data)

    dataset = torch.utils.data.TensorDataset( torch.from_numpy(data_fs_s), torch.from_numpy(data_fs_t))
    return dataset

def CreateDatasetSynthesis_single_with_masks(phase, input_path, contrast, target_contrast):
    cond_data = input_path +"/"+ phase + "/"+ contrast +".npy"
    data_fs_s = LoadDataSet(cond_data)

    cond_masks = input_path +"/"+ phase + "/"+ contrast +"_masks.npy"
    data_fs_s_masks = LoadDataSet(cond_masks)

    target_data = input_path +"/"+ phase + "/"+ target_contrast +".npy"
    data_fs_t = LoadDataSet(target_data)

    target_masks = input_path +"/"+ phase + "/"+ target_contrast +"_masks.npy"
    data_fs_t_masks = LoadDataSet(target_masks)

    dataset = torch.utils.data.TensorDataset( torch.from_numpy(data_fs_s), torch.from_numpy(data_fs_t), torch.from_numpy(data_fs_s_masks), torch.from_numpy(data_fs_t_masks))
    return dataset

def CreateDatasetSynthesis_masks(phase, input_path):
    cond_data1 = input_path +"/"+ phase + "/BOLD_masks.npy"
    data_fs_s1 = LoadDataSet(cond_data1)

    cond_data2 = input_path +"/"+ phase + "/Diffusion_masks.npy"
    data_fs_s2 = LoadDataSet(cond_data2)

    cond_data3 = input_path +"/"+ phase + "/T1_mapping_fl2d_masks.npy"
    data_fs_s3 = LoadDataSet(cond_data3)

    target_data = input_path +"/"+ phase + "/DIXON_masks.npy"
    data_fs_s4 = LoadDataSet(target_data)

    dataset = torch.utils.data.TensorDataset( torch.from_numpy(data_fs_s1), torch.from_numpy(data_fs_s2), torch.from_numpy(data_fs_s3), torch.from_numpy(data_fs_s4))
    return dataset


def LoadDataSet(load_dir, padding=True, Norm=True):

    # Load the Numpy array
    data=np.load(load_dir)

    # Transpose and expand dimensions if necessary
    if data.ndim == 3:
        data = np.expand_dims(np.transpose(data, (0, 2, 1)), axis=1)
    else:
        data = np.transpose(data, (1, 0, 3, 2))

    data = data.astype(np.float32)

    if padding:
        pad_x = int((256 - data.shape[2]) / 2)
        pad_y = int((256 - data.shape[3]) / 2)
        print('padding in x-y with:' + str(pad_x) + '-' + str(pad_y))
        data = np.pad(data, ((0, 0), (0, 0), (pad_x, pad_x), (pad_y, pad_y)))

    if Norm:
        data = (data - 0.5) / 0.5

    return data