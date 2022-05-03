import pandas as pd
import torch
import datetime

data = pd.read_csv('../data/data_sample_3000.csv', index_col=0)
num_samples = len(data)
grid_location_gps = eval(data.loc[0, 'GPS_location'])
grid2idx = {key: i for i, key in enumerate(grid_location_gps.keys())}
num_grids = len(grid2idx)
region2grid = eval(data.loc[0, 'dic_final'])
region2idx = {key: i for i, key in enumerate(region2grid.keys())}
grid2region = {grid: region for region in region2grid.keys() for grid in region2grid[region]}
num_regions = len(region2idx)

def get_features(idx):
    """
    param: idx
    return: DataFrame (num_grids, num_features)
    index: grid_id
    features: [latitude, longitude,
    morning_amount, afternoon_amount, region_id,
    final_time, final_location]
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

    # to tensor
    data_sample = torch.tensor(df.values)

    return data_sample.T

dataset = torch.stack([get_features(idx) for idx in range(10)], dim=0)

print(dataset.shape)


def get_agg_matrix(grid2idx, region2idx, grid2region):
    agg_matrix = pd.DataFrame(data=0, index=grid2idx.keys(), columns=region2idx.keys())
    for idx, col in grid2region.items():
        agg_matrix.loc[idx, col] = 1
    agg_matrix /= agg_matrix.sum()
    return torch.tensor(agg_matrix.values)

agg_matrix = get_agg_matrix(grid2idx, region2idx, grid2region)

print(agg_matrix.shape)
