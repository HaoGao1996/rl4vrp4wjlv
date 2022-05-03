"""Defines the main task for the TSP

The TSP is defined by the following traits:
    1. Each city in the list must be visited once and only once
    2. The salesman must return to the original node at the end of the tour

Since the TSP doesn't have dynamic elements, we return an empty list on
__getitem__, which gets processed in trainer.py to be None

"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

def get_features(data, idx, grid2idx, region2idx, grid2region):
    """
    param: data
    param: idx
    param: grid2idx
    param: region2idx
    param: grid2region
    return: DataFrame (num_grids, num_features)
    ----------------------------
    index: grid_id
    features: [latitude, longitude, morning_amount, afternoon_amount, 'arrival_time',
              region_id, final_time, final_location]
    """
    # to pandas
    df_grid_features = pd.DataFrame(index=grid2idx.keys())
    df_grid_location_gps = pd.DataFrame(eval(data.loc[idx, 'GPS_location']),
                                        index=['latitude', 'longitude']).T
    df_morning_amount = pd.DataFrame(eval(data.loc[idx, 'morning_amount']),
                                     index=['morning_amount']).T
    df_afternoon_amount = pd.DataFrame(eval(data.loc[idx, 'afternoon_amount']),
                                       index=['afternoon_amount']).T
    df_arrival_time = pd.DataFrame(eval(data.loc[idx, 'arrival_time']),
                                   index=['arrival_time']).T
    df_region_id = pd.DataFrame(grid2region, index=['region_id']).T

    df_grid_features = df_grid_features.join([df_grid_location_gps, df_morning_amount,
                                              df_afternoon_amount, df_arrival_time,
                                              df_region_id], how='left').rename_axis('grid_id').reset_index()

    df_region_features = pd.DataFrame(index=region2idx.keys())
    df_final_time = pd.DataFrame(eval(data.loc[idx, 'final_time']),
                                 index=['final_time']).T
    df_final_location = pd.DataFrame(eval(data.loc[idx, 'final_location']),
                                     index=['final_location']).T
    df_region_features = df_region_features.join([df_final_time, df_final_location],
                                                 how='left').rename_axis('region_id').reset_index()

    df = pd.merge(df_grid_features, df_region_features, on='region_id', how='left')
    df[['arrival_time', 'final_time']] = df[['arrival_time', 'final_time']].fillna('1999-01-01 00:00:00.000000')
    df['arrival_time'] = (
                pd.to_datetime(df['arrival_time']) - pd.to_datetime('1999-01-01 00:00:00.000000')).dt.total_seconds()
    df['final_time'] = (
                pd.to_datetime(df['final_time']) - pd.to_datetime('1999-01-01 00:00:00.000000')).dt.total_seconds()

    # to tensor and return
    return torch.tensor(df.values).T


def get_agg_matrix(grid2idx, region2idx, grid2region):
    agg_matrix = pd.DataFrame(data=0, index=grid2idx.keys(), columns=region2idx.keys())
    for idx, col in grid2region.items():
        agg_matrix.loc[idx, col] = 1
    agg_matrix /= agg_matrix.sum()
    return torch.tensor(agg_matrix.values)

class TSPDataset(Dataset):
    def __init__(self, file_name):
        super(TSPDataset, self).__init__()
        self.file_name = file_name
        self.data = pd.read_csv(self.file_name)
        self.size = len(self.data)

        self.grid2idx = {key: i for i, key in enumerate(eval(self.data.loc[0, 'GPS_location']).keys())}
        self.num_grids = len(self.grid2idx)

        self.region2idx = {key: i for i, key in enumerate(eval(self.data.loc[0, 'dic_final']).keys())}
        self.num_regions = len(self.region2idx)

        self.grid2region = {grid: region for region in self.region2grid.keys() for grid in self.region2grid[region]}

        # (size, static_size, num_regions)
        self.dataset = torch.stack([get_features(self.data, idx,
                                                 self.grid2idx, self.region2idx,
                                                 self.grid2region) for idx in range(10)], dim=0)

        self.num_nodes = self.num_regions
        self.dynamic = torch.zeros(self.size, 1, self.num_nodes)
        self.static_size = self.dataset.shape[1]
        self.dynamic_size = 1
        self.agg_matrix = get_agg_matrix(self.grid2idx,
                                         self.region2idx,
                                         self.grid2region)  # (num_grids, nums_regions)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # (static, dynamic, start_loc)
        return (self.dataset[idx], self.dynamic[idx], [])


def update_mask(mask, dynamic, chosen_idx):
    """Marks the visited city, so it can't be selected a second time."""
    mask.scatter_(1, chosen_idx.unsqueeze(1), 0)
    return mask


def reward(static, tour_indices):
    """
    Parameters
    ----------
    static: torch.FloatTensor containing static (e.g. x, y) data
    tour_indices: torch.IntTensor of size (batch_size, num_cities)

    Returns
    -------
    Euclidean distance between consecutive nodes on the route. of size
    (batch_size, num_cities)
    """

    # Convert the indices back into a tour
    idx = tour_indices.unsqueeze(1).expand_as(static)
    tour = torch.gather(static.data, 2, idx).permute(0, 2, 1)

    # Make a full tour by returning to the start
    y = torch.cat((tour, tour[:, :1]), dim=1)

    # Euclidean distance between each consecutive point
    tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))

    return tour_len.sum(1).detach()


def render(static, tour_indices, save_path):
    """Plots the found tours."""

    plt.close('all')

    num_plots = 3 if int(np.sqrt(len(tour_indices))) >= 3 else 1

    _, axes = plt.subplots(nrows=num_plots, ncols=num_plots,
                           sharex='col', sharey='row')

    if num_plots == 1:
        axes = [[axes]]
    axes = [a for ax in axes for a in ax]

    for i, ax in enumerate(axes):

        # Convert the indices back into a tour
        idx = tour_indices[i]
        if len(idx.size()) == 1:
            idx = idx.unsqueeze(0)

        # End tour at the starting index
        idx = idx.expand(static.size(1), -1)
        idx = torch.cat((idx, idx[:, 0:1]), dim=1)

        data = torch.gather(static[i].data, 1, idx).cpu().numpy()

        #plt.subplot(num_plots, num_plots, i + 1)
        ax.plot(data[0], data[1], zorder=1)
        ax.scatter(data[0], data[1], s=4, c='r', zorder=2)
        ax.scatter(data[0, 0], data[1, 0], s=20, c='k', marker='*', zorder=3)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=400)
