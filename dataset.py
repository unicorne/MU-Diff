import scipy
import torch.utils.data
import numpy as np
import random
import torchio as tio

# transform = tio.Compose([
#     tio.transforms.RandomMotion()  # Add custom transformation
#     # Add more augmentations as needed
# ])

def CreateDatasetSynthesis(phase, input_path, contrast1='T1', contrast2='T2'):
    input_path = '/data/shew0029/MedSyn/DATA/BRATS/brats19/'
    phase = 'train'
    target_file = input_path + phase + "/flair.npy"
    data_fs_s1 = LoadDataSet(target_file)

    target_file = input_path + phase + "/t2.npy"
    data_fs_s2 = LoadDataSet(target_file)

    target_file = input_path + phase + "/t1.npy"
    data_fs_s3 = LoadDataSet(target_file)
    # target_file = input_path + phase+"/seg.mat"
    # data_fs_s3=LoadDataSet(target_file, Norm = False)


    target_file = input_path + phase + "/t1ce.npy"
    data_fs_s4 = LoadDataSet(target_file)

    # dataset=torch.utils.data.TensorDataset(torch.from_numpy(data_fs_s1),torch.from_numpy(data_fs_s2))
    dataset = torch.utils.data.TensorDataset( torch.from_numpy(data_fs_s1), torch.from_numpy(data_fs_s2), torch.from_numpy(data_fs_s3), torch.from_numpy(data_fs_s4))
    return dataset


# Dataset loading from load_dir and converintg to 256x256
# def LoadDataSet(load_dir, variable='slices', padding=True, Norm=True):
#     f = scipy.io.loadmat(load_dir)
#     if np.array(f[variable]).ndim == 3:
#         data = np.expand_dims(np.transpose(np.array(f[variable]), (0, 2, 1)), axis=1)
#     else:
#         data = np.transpose(np.array(f[variable]), (1, 0, 3, 2))
#     if Norm:
#         data = data.astype(np.float32)
#     else:
#         data = data.astype(np.uint8)
#     if padding:
#         pad_x = int((256 - data.shape[2]) / 2)
#         pad_y = int((256 - data.shape[3]) / 2)
#         print('padding in x-y with:' + str(pad_x) + '-' + str(pad_y))
#         data = np.pad(data, ((0, 0), (0, 0), (pad_x, pad_x), (pad_y, pad_y)))
#     if Norm:
#         data = (data - 0.5) / 0.5
#     return data

def LoadDataSet(load_dir, variable='slices', padding=True, Norm=True):

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