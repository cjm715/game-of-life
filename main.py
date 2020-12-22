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
from model import *
import math


dataset_test_pd = pd.read_csv('./data/test.csv')
for delta in range(1, 6):
    dataset_test_pd[dataset_test_pd['delta'] == delta].sort_values(
        'id').to_csv(f"./data/test_delta_{delta}.csv", index=False)

sample_sub_df = pd.read_csv(f'./data/sample_submission.csv')

batch_size_desired = 5000
first_batch = True

for delta in range(5, 2, -1):
    print(delta)
    dataset_test_pd = pd.read_csv(f'./data/test_delta_{delta}.csv')
    dataset_test = TaskDataset(dataset_test_pd)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size_desired, shuffle=False)

    for i_batch, sample_batched in enumerate(dataloader_test):
        stop_grid = sample_batched['stop']
        batch_size = stop_grid.size()[0]

        start_grid_pred = torch.zeros((batch_size, 1, 25, 25), device=device)

        cost = accuracy_error(step(start_grid_pred, delta=delta)[0], stop_grid)
        cost_best = cost.clone()
        print(cost_best.mean())
        start_grid_pred_best = start_grid_pred.clone()
        pbar = tqdm(torch.arange(360000, device=device))
        temperature = 0.1
        gamma = 0.999
        for i in pbar:
            start_grid_pred_new = start_grid_pred.clone()

            if i % 500 == 0:
                start_grid_pred_new = start_grid_pred_best.clone()

            start_grid_pred_new = flip_n_bits(start_grid_pred_new, 1, batch_size)
            stop_grid_pred_new, _ = step(start_grid_pred_new, delta=delta)
            cost_new = accuracy_error(stop_grid_pred_new, stop_grid)

            cost_change = cost_new - cost  # + reg

            temperature = max(temperature*gamma, 1e-4)

            prob, _ = torch.min(torch.exp(-cost_change/temperature), 1)
            choice = (torch.rand(batch_size, device=device) < prob)
            start_grid_pred = torch.where(choice.view(batch_size, 1, 1, 1),
                                          start_grid_pred_new,
                                          start_grid_pred)
            cost = torch.where(choice.view(batch_size, 1), cost_new, cost)

            start_grid_pred_best = torch.where(
                (cost < cost_best).view(batch_size, 1, 1, 1),
                start_grid_pred,
                start_grid_pred_best)
            cost_best = torch.where(cost < cost_best, cost, cost_best)

            pbar.set_description(f' Cost (best): {cost_best.mean()} Temp : {temperature}')

        start_grid_pred_best_np = start_grid_pred_best.cpu().numpy().reshape(batch_size, SIDE_LENGTH*SIDE_LENGTH)
        sub_df = pd.DataFrame(start_grid_pred_best_np)
        sub_df.insert(loc=0, column='id', value=sample_batched['id'].cpu().numpy())
        sub_df.columns = sample_sub_df.columns

        if first_batch:
            full_sub_df = sub_df.copy()
            cost_best_all = cost_best.clone()
            first_batch = False
        else:
            full_sub_df = pd.concat([full_sub_df, sub_df])
            cost_best_all = torch.cat([cost_best_all, cost_best])

        print(cost_best_all.mean())
        full_sub_df.sort_values('id').to_csv('submission.csv', index=False)
