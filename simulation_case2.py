# Load packages
import numpy as np
import pickle
import FDApy
import torch.nn as nn
import torch

from FDApy.simulation import Brownian
from sklearn.preprocessing import StandardScaler



class sim_data_v2_c3(nn.Module):
    def __init__(self, sample_size=600, time_grid=101, view=2, num=0, path=None):
        super(sim_data_v2_c3, self).__init__()
        # Define parameters of the simulation
        self.sample_size = sample_size
        self.cluster_size = int(sample_size / 3)
        self.M = time_grid
        self.alpha = 0.4
        self.hurst1, self.hurst2 = 0.9, 0.8
        self.view = view

        np.random.seed(258 + num)
        torch.manual_seed(258 + num)
        self.generate_data(path)

    # Define mean functions
    def h(self, x, a):
        return 6 - np.abs(x - a)

    def generate_data(self, path):
        scaler = StandardScaler()

        x = np.linspace(1, 21, self.M)
        self.labels = np.repeat([0, 1, 2], repeats=(self.cluster_size, self.cluster_size, self.cluster_size)).astype(np.float32)

        h_1 = lambda x: self.h(x, 7) / 4 if self.h(x, 7) > 0 else 0
        h_2 = lambda x: self.h(x, 15) / 4 if self.h(x, 15) > 0 else 0
        h_3 = lambda x: self.h(x, 11) / 4 if self.h(x, 11) > 0 else 0

        # Generate data
        A = np.zeros((self.sample_size, self.M))
        B = np.zeros((self.sample_size, self.M))
        for idx in range(self.sample_size):
            h1 = np.array([h_1(i) for i in x])
            h2 = np.array([h_2(i) for i in x])
            h3 = np.array([h_3(i) for i in x])
            
            brownian = Brownian(name='fractional')
            brownian.new(1, argvals=np.linspace(0, 2, 2 * self.M), hurst=self.hurst1)
            rand_part1 = brownian.data.values[0, self.M:] / (1 + np.linspace(0, 1, self.M)) ** self.hurst1
                
            brownian = Brownian(name='fractional')
            brownian.new(1, argvals=np.linspace(0, 2, 2 * self.M), hurst=self.hurst2)
            rand_part2 = brownian.data.values[0, self.M:] / (1 + np.linspace(0, 1, self.M)) ** self.hurst2
            
            eps = np.random.normal(0, np.sqrt(0.2), size=self.M)
            if idx < self.cluster_size:
                A[idx, :] = h1 + rand_part1 + eps
                B[idx, :] = h3 + 1.2 * rand_part2 + eps
            elif self.cluster_size <= idx < self.cluster_size + self.cluster_size:
                A[idx, :] = h2 + rand_part1 + eps
                B[idx, :] = h3 + 0.2 * rand_part2 + eps
            else:
                A[idx, :] = h2 + 0.1 * rand_part1 + eps
                B[idx, :] = h1 + 0.1 * rand_part2 + eps
        V1 = A + self.alpha * B
        V2 = B
        # self.x1 = scaler.fit_transform(A)
        # self.x2 = scaler.fit_transform(B)
        self.x1 = V1.astype(np.float32)
        self.x2 = V2.astype(np.float32)
        data = np. stack((self.x1,self.x2), axis=0)
        np.save(path + '/' + 'x.npy', data)
        np.save(path + '/' + 'y.npy', self.labels)
    
    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        df = [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx])]
        
        return df, torch.tensor(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


def load_data(dataset, sample_size, time_grid, view, num_class, num, path):
    if dataset == "fd_v2_c3":
        dataset = sim_data_v2_c3(sample_size, time_grid, view, num, path)
        dims = [time_grid,time_grid]
        view = view
        data_size = sample_size
        class_num = num_class
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
