import h5py
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader


def mask_data(data, bandwidth, rng):
    nan_mask = np.full(data.shape, np.nan)
    for i, sample in enumerate(data):
        for j, timestep in enumerate(sample):
            neuron_ixs = np.arange(len(timestep))
            sampled_ixs = rng.choice(neuron_ixs, size=bandwidth, replace=False)
            nan_mask[i, j, sampled_ixs] = 1.0
    return data * nan_mask

def mask_data_time(data, bandwidth, rng):
    nan_mask = np.full(data.shape, np.nan)
    for i, sample in enumerate(data):
        for j in range(np.shape(sample)[1]):
            time_ixs = np.arange(np.shape(sample)[0])
            sampled_ixs = rng.choice(time_ixs, size=bandwidth, replace=False)
            nan_mask[i, sampled_ixs, j] = 1.0
    return data * nan_mask

def uniform_mask_data(data, bandwidth):
    nan_mask = np.full(data.shape, np.nan)
    for i, sample in enumerate(data):
        for j, timestep in enumerate(sample):
            sampled_ixs = np.int32(np.linspace(0, len(timestep)-1, bandwidth))
            nan_mask[i, j, sampled_ixs] = 1.0
    return data * nan_mask

def uniform_mask_data_time(data, bandwidth):
    nan_mask = np.full(data.shape, np.nan)
    for i, sample in enumerate(data):
        for j in range(np.shape(sample)[1]):
            sampled_ixs = np.int32(np.linspace(0, np.shape(sample)[0]-1, bandwidth))
            nan_mask[i, sampled_ixs, j] = 1.0
    return data * nan_mask


class LorenzDataModule(pl.LightningDataModule):
    def __init__(self, data_path, bandwidth=None, batch_size=64, num_workers=4, mask_type='random', mask_axis='neuron', dont_mask=False, seed=0):
        super().__init__()
        self.save_hyperparameters()
        self.rng = np.random.RandomState(seed=seed)
    
    def setup(self, stage=None):
        hps = self.hparams
        # Load data arrays from file
        with h5py.File(hps.data_path, 'r') as h5file:
            data_dict = {k: v[()] for k, v in h5file.items()}
        train_spikes = data_dict['train_data']
        valid_spikes = data_dict['valid_data']
        train_rates = data_dict['train_truth']
        valid_rates = data_dict['valid_truth']
        # Simulate bandwidth-limited sampling
        if not(hps.dont_mask):
            if hps.bandwidth is not None:
                if hps.mask_type == 'random':
                    if hps.mask_axis == 'neuron':
                        train_spikes = mask_data(train_spikes, hps.bandwidth, self.rng)
                        valid_spikes = mask_data(valid_spikes, hps.bandwidth, self.rng)
                    elif hps.mask_axis == 'time':
                        train_spikes = mask_data_time(train_spikes, hps.bandwidth, self.rng)
                        valid_spikes = mask_data_time(valid_spikes, hps.bandwidth, self.rng)
                elif hps.mask_type == 'uniform':
                    if hps.mask_axis == 'neuron':
                        train_spikes = uniform_mask_data(train_spikes, hps.bandwidth)
                        valid_spikes = uniform_mask_data(valid_spikes, hps.bandwidth)
                    elif hps.mask_axis == 'time':
                        train_spikes = uniform_mask_data_time(train_spikes, hps.bandwidth)
                        valid_spikes = uniform_mask_data_time(valid_spikes, hps.bandwidth)

        # Convert data to Tensors
        train_spikes = torch.tensor(train_spikes, dtype=torch.float)
        valid_spikes = torch.tensor(valid_spikes, dtype=torch.float)
        train_rates = torch.tensor(train_rates, dtype=torch.float)
        valid_rates = torch.tensor(valid_rates, dtype=torch.float)
        # Store datasets
        self.train_ds = TensorDataset(train_spikes, train_rates)
        self.valid_ds = TensorDataset(valid_spikes, valid_rates)

    def train_dataloader(self):
        train_dl = DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )
        return train_dl
    
    def val_dataloader(self):
        valid_dl = DataLoader(
            self.valid_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )
        return valid_dl
