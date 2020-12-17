import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch import FloatTensor, LongTensor
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import sys

SIDE_LENGTH = 25
NUM_WARM_UP_STEPS = 5
GRID_SHAPE = (SIDE_LENGTH, SIDE_LENGTH)
device="cuda"

TOTAL_NUM_CELLS = torch.tensor(SIDE_LENGTH*SIDE_LENGTH).type(torch.LongTensor).to(device)
MASK = torch.tensor([[1,1,1],
                     [1,0,1],
                     [1,1,1]]).view(1,1,3,3).type(torch.FloatTensor).to(device)

def line2grid_tensor(data, device='cuda'):
    grid = data.to_numpy().reshape((data.shape[0], 1, 25, 25))
    return torch.tensor(grid).type(torch.int).to(device)


class TaskDataset(Dataset):
    def __init__(self, data, data_initialize = None, device='cuda'):
        self.id = LongTensor(data.iloc[:,0].to_numpy()).to(device)
        self.delta = LongTensor(data.iloc[:,1].to_numpy()).to(device)
        self.stop = line2grid_tensor(data.iloc[:,2:], device)
        if data_initialize is not None:
            self.start_guess = line2grid_tensor(data_initialize.iloc[:,1:], device)
        else:
            self.start_guess = None
        
    def __len__(self):
        return len(self.delta)

    def __getitem__(self, idx):
        if self.start_guess is not None:
            return {'start_guess': self.start_guess[idx], 'stop': self.stop[idx], 'delta': self.delta[idx], 'id': self.id[idx]}
        else:
            return {'stop': self.stop[idx], 'delta': self.delta[idx], 'id': self.id[idx]}


def G(s, grid):
    sumIs2 = s == 2
    sumIs3 = s == 3
    isAlive = grid == 1
    grid = torch.logical_or(torch.logical_and(torch.logical_or(sumIs2, sumIs3), isAlive),
                         torch.logical_and(sumIs3, torch.logical_not(isAlive)))
    grid = grid.type(torch.int)
    return grid

def sum_neighbors(grid):
    grid_padded = F.pad(grid,(1,1,1,1), mode="circular").type(torch.cuda.FloatTensor)
    s_layer = F.conv2d(grid_padded, MASK).type(torch.cuda.IntTensor)
    return s_layer

def step_single(grid):
    s_layer = sum_neighbors(grid)
    grid = G(s_layer, grid)
    return grid, s_layer

def step(grid, delta=1):
    for _ in range(0,delta):
        grid,s_layer = step_single(grid)
    return grid, s_layer

def accuracy_error(grid_pred, grid_true):
    C = torch.sum(grid_pred != grid_true, (2,3)).type(torch.cuda.FloatTensor)/TOTAL_NUM_CELLS
    return C

def flip_n_bits(grid,n,batch_size):
    flat_index_per_frame = torch.randint(0,SIDE_LENGTH*SIDE_LENGTH, (batch_size,n))
    total_cell_upto_last_frame = torch.arange(0, SIDE_LENGTH*SIDE_LENGTH*batch_size, SIDE_LENGTH*SIDE_LENGTH).view(batch_size,1)
    flat_indices = total_cell_upto_last_frame + flat_index_per_frame
    flat_indices=flat_indices.squeeze()
    grid.flatten()[flat_indices] = 1 - grid.flatten()[flat_indices] 
    return grid

def plotg(grid):
    plt.imshow(grid, cmap='Greys',  interpolation='nearest')