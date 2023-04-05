import numpy as np
import h5py
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-n', '--neurons_count', type=int, default=50, help='Number of neurons in one sample')
parser.add_argument('-t', '--timesteps_count', type=int, default=29, help='timesteps count')
parser.add_argument('-s', '--samples_count', type=int, default=2000, help='Number of samples')
parser.add_argument('-a', '--alpha', type=float, default=0.85, help='First order AR model parameter')
parser.add_argument('-k', '--latent_dimension', type=int, default=10, help='Dimension of latent states')

args = parser.parse_args()

neurons_count = args.neurons_count
timesteps_count = args.timesteps_count
samples_count = args.samples_count
alpha = args.alpha
latent_dimension = args.latent_dimension

# Generate W and u: they map latent states to a space with dimension = neuron count
W = np.random.randn(neurons_count, latent_dimension)
muBias = 1
sigmaBias = 1
b = np.random.normal(muBias, sigmaBias, size=(neurons_count, 1))

Shi = np.zeros((samples_count, neurons_count, timesteps_count))
Yhi = np.zeros((samples_count, neurons_count, timesteps_count))
rates = np.zeros((samples_count, neurons_count, timesteps_count))

for sample_idx in np.arange(samples_count):
    if sample_idx % 200 == 0:
        # Latent state: not varying with time
        u = np.random.randn(latent_dimension, 1)
        # Bernoulli probability for spikes
        p = np.exp(np.matmul(W, u) + b)/(1 + np.exp(np.matmul(W, u) + b))

    for t in np.arange(timesteps_count):
        Shi[sample_idx, :, t] = np.squeeze(np.random.binomial(n=1, p=p, size=(neurons_count, 1)))
        if t == 0:
            Yhi[sample_idx, :, t] = Shi[sample_idx, :, t]
        else:
            Yhi[sample_idx, :, t] = alpha*Yhi[sample_idx, :, t-1] + Shi[sample_idx, :, t]
        rates[sample_idx, :, t] = np.squeeze(p)

# split train and validation data
 
train_size = int(0.8*samples_count)
random_idx = np.random.choice(samples_count, train_size, replace=False)
data_dict = {}
data_dict['train_data'] = Yhi[random_idx, :, :]
data_dict['train_truth'] = rates[random_idx, :, :]
data_dict['train_spikes'] = Shi[random_idx, :, :]
valid_sample_idx = [i for i in range(samples_count) if i not in random_idx]
data_dict['valid_data'] = Yhi[valid_sample_idx, :, :]
data_dict['valid_truth'] = rates[valid_sample_idx, :, :]
data_dict['valid_spikes'] = Shi[valid_sample_idx, :, :]

# Store it as a h5 file

with h5py.File('calcium2Pdata.h5', 'w') as h5File:
    for k, v in data_dict.items():
        h5File.create_dataset(k, data=np.array(v))






