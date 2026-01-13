from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import h5py



class FDA_case3(Dataset):
    def __init__(self, path, exp_num, view):
        #data1 = scipy.io.loadmat(path+'BDGP.mat')['X1'].astype(np.float32)
        #data2 = scipy.io.loadmat(path+'BDGP.mat')['X2'].astype(np.float32)
        #labels = scipy.io.loadmat(path+'BDGP.mat')['Y'].transpose()
        scaler = StandardScaler()
        self.view = view
        fake_label = np.ones(15) + 1
        labels = np.load(path + 'label.npy')
        labels = np.concatenate((fake_label,labels)).astype(np.float32)
        data = np.load(path + f'repeat{exp_num}.npy')
        for i in range(10):
            name = f'x{i+1}'
            matrix = np.array(data[i]).astype(np.float32).T
            setattr(self, f'{name}', scaler.fit_transform(matrix))
        
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        df = [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx]), torch.from_numpy(self.x3[idx]),
        torch.from_numpy(self.x4[idx]), torch.from_numpy(self.x5[idx]), torch.from_numpy(self.x6[idx]),
        torch.from_numpy(self.x7[idx]), torch.from_numpy(self.x8[idx]), torch.from_numpy(self.x9[idx]),
        torch.from_numpy(self.x10[idx])
        ]
        
        return df, torch.tensor(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class FDA_case4(Dataset):
    def __init__(self, path, exp_num, view):
        #data1 = scipy.io.loadmat(path+'BDGP.mat')['X1'].astype(np.float32)
        #data2 = scipy.io.loadmat(path+'BDGP.mat')['X2'].astype(np.float32)
        #labels = scipy.io.loadmat(path+'BDGP.mat')['Y'].transpose()
        scaler = StandardScaler()
        self.view = view
        labels = np.load(path + 'label.npy').astype(np.float32)
        data = np.load(path + f'repeat{exp_num}.npy')
        for i in range(view):
            name = f'x{i+1}'
            matrix = np.array(data[i]).astype(np.float32).T
            setattr(self, f'{name}', scaler.fit_transform(matrix))
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        df = [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx]), torch.from_numpy(self.x3[idx]),
              torch.from_numpy(self.x4[idx]), torch.from_numpy(self.x5[idx])
        ]
        
        return df, torch.tensor(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class FDA_case4_view9(Dataset):
    def __init__(self, path, exp_num, view):
        #data1 = scipy.io.loadmat(path+'BDGP.mat')['X1'].astype(np.float32)
        #data2 = scipy.io.loadmat(path+'BDGP.mat')['X2'].astype(np.float32)
        #labels = scipy.io.loadmat(path+'BDGP.mat')['Y'].transpose()
        scaler = StandardScaler()
        self.view = view
        labels = np.load(path + 'label.npy').astype(np.float32)
        data = np.load(path + f'repeat{exp_num}.npy')
        for i in range(9):
            name = f'x{i+1}'
            matrix = np.array(data[i]).astype(np.float32).T
            setattr(self, f'{name}', scaler.fit_transform(matrix))
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        df = [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx]), torch.from_numpy(self.x3[idx]),
              torch.from_numpy(self.x4[idx]), torch.from_numpy(self.x5[idx]), torch.from_numpy(self.x6[idx]),
              torch.from_numpy(self.x7[idx]), torch.from_numpy(self.x8[idx]), torch.from_numpy(self.x9[idx])
        ]
        
        return df, torch.tensor(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class FDA_2class(Dataset):
    def __init__(self, path, exp_num, view):
        #data1 = scipy.io.loadmat(path+'BDGP.mat')['X1'].astype(np.float32)
        #data2 = scipy.io.loadmat(path+'BDGP.mat')['X2'].astype(np.float32)
        #labels = scipy.io.loadmat(path+'BDGP.mat')['Y'].transpose()
        scaler = StandardScaler()
        self.view = view
        #scaler = MinMaxScaler()
        hdf_file_name = path + f'data{exp_num+1}_matrices.h5'
        with h5py.File(hdf_file_name, 'r') as file:
            self.y = (np.array(file['label']) - 1).astype(np.float32)
            #self.y = [value - 1 for value in self.y]
            matrices = {name: np.array(file[name]).astype(np.float32) for name in file.keys() if name != 'label'}
            for name, matrix in matrices.items():
                setattr(self, f'{name}', scaler.fit_transform(matrix))

    def __len__(self):
        return self.matrix1.shape[0]

    def __getitem__(self, idx):
        data = [torch.from_numpy(self.matrix1[idx]), torch.from_numpy(self.matrix2[idx]), torch.from_numpy(self.matrix3[idx]),
                torch.from_numpy(self.matrix4[idx]), torch.from_numpy(self.matrix5[idx]), torch.from_numpy(self.matrix6[idx]),
                torch.from_numpy(self.matrix7[idx]), torch.from_numpy(self.matrix8[idx]), torch.from_numpy(self.matrix9[idx]),
                torch.from_numpy(self.matrix10[idx]), torch.from_numpy(self.matrix11[idx]), torch.from_numpy(self.matrix12[idx])
                ]
        
        return data[0:self.view], torch.tensor(self.y[idx]), torch.from_numpy(np.array(idx)).long()
    
class FDA_3class(Dataset):
    def __init__(self, path, exp_num, view):
        #data1 = scipy.io.loadmat(path+'BDGP.mat')['X1'].astype(np.float32)
        #data2 = scipy.io.loadmat(path+'BDGP.mat')['X2'].astype(np.float32)
        #labels = scipy.io.loadmat(path+'BDGP.mat')['Y'].transpose()
        scaler = StandardScaler()
        self.view = view
        #scaler = MinMaxScaler()
        hdf_file_name = path + f'data{exp_num+4}_matrices.h5'
        with h5py.File(hdf_file_name, 'r') as file:
            self.y = (np.array(file['label']) - 1).astype(np.float32)
            #self.y = [value - 1 for value in self.y]
            matrices = {name: np.array(file[name]).astype(np.float32) for name in file.keys() if name != 'label'}
            for name, matrix in matrices.items():
                setattr(self, f'{name}', scaler.fit_transform(matrix))

    def __len__(self):
        return self.matrix1.shape[0]

    def __getitem__(self, idx):
        data = [torch.from_numpy(self.matrix1[idx]), torch.from_numpy(self.matrix2[idx]), torch.from_numpy(self.matrix3[idx]),
                torch.from_numpy(self.matrix4[idx]), torch.from_numpy(self.matrix5[idx]), torch.from_numpy(self.matrix6[idx]),
                torch.from_numpy(self.matrix7[idx]), torch.from_numpy(self.matrix8[idx]), torch.from_numpy(self.matrix9[idx]),
                torch.from_numpy(self.matrix10[idx]), torch.from_numpy(self.matrix11[idx]), torch.from_numpy(self.matrix12[idx])
                ]
        
        return data[0:self.view], torch.tensor(self.y[idx]), torch.from_numpy(np.array(idx)).long()






def load_data(dataset, exp_num, view):
    if dataset == "fd_sim_2class":
        dataset = FDA_2class('/home/gaop/FDA_project/FDA_data/', exp_num, view)
        dims = [[21, 50, 23, 73, 85, 13, 98, 32, 87, 76, 60, 23],
                [21,22,85,68,33,72,43,67,41,17,15,17],
                [21,14,89,10,42,28,95,94,15,61,89,98]
                ]
        view = view
        data_size = [2000,1000,500]
        class_num = 2
    elif dataset == "fd_sim_3class":
        dataset = FDA_3class('/home/gaop/FDA_project/FDA_data/', exp_num, view)
        dims = [[63,79,53,57,57,52,55,29,81,59,61,60],
                [63,55,61,94,66,75,28,99,20,72,59,68],
                [63,62,15,85,58,28,70,11,89,90,88,52]
                ]
        view = view
        data_size = [3000,1500,300]
        class_num = 3
    elif dataset == "fd_case3":
        dataset = FDA_case3('/home/gaop/FDA_project/MFLVC-main/data/simulation/case3/', exp_num, view)
        dims = [25,25,25,25,25,25,25,25,25,25]
        view = view
        data_size = 115
        class_num = 2
    elif dataset == "fd_case4":
        dataset = FDA_case4('/home/gaop/FDA_project/MFLVC-main/data/simulation/case4/', exp_num, view)
        dims = [21,21,21]
        view = view
        data_size = 200
        class_num = 2
    elif dataset == "fd_case4_1k":
        dataset = FDA_case4('/home/gaop/FDA_project/MFLVC-main/data/simulation2/case4_1k/', exp_num, view)
        dims = [21,21,21]
        view = view
        data_size = 1000
        class_num = 2
    elif dataset == "fd_case4_3k":
        dataset = FDA_case4('/home/gaop/FDA_project/MFLVC-main/data/simulation2/case4_3k/', exp_num, view)
        dims = [21,21,21]
        view = view
        data_size = 3000
        class_num = 2
    elif dataset == "fd_case4_5k":
        dataset = FDA_case4('/home/gaop/FDA_project/MFLVC-main/data/simulation2/case4_5k/', exp_num, view)
        dims = [21,21,21]
        view = view
        data_size = 5000
        class_num = 2
    elif dataset == "fd_case4_1k-9":
        dataset = FDA_case4_view9('/home/gaop/FDA_project/MFLVC-main/data/simulation3/case4_1k/', exp_num, view)
        dims = [21,21,21,21,21,21,21,21,21]
        view = view
        data_size = 1000
        class_num = 2
    elif dataset == "fd_case1_v5_c2":
        dataset = FDA_case4('/home/gaop/FDA_project/MFLVC-main/data/sim1/case1/', exp_num, view)
        dims = [97,97,97,97,97]
        view = view
        data_size = 500
        class_num = 2
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
