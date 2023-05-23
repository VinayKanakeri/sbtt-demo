import numpy as np
import h5py
import argparse
import sys
sys.path.insert(1, '/home/vinay_kanakeri/Codes/OASIS/')
from oasis.functions import gen_data, gen_sinusoidal_data, deconvolve, estimate_parameters
from oasis.plotting import simpleaxis
from oasis.oasis_methods import oasisAR1, oasisAR2
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('-n', '--neurons_count', type=int, default=29, help='Number of neurons in one sample')
parser.add_argument('-t', '--timesteps_count', type=int, default=50, help='timesteps count')
parser.add_argument('-s', '--samples_count', type=int, default=2000, help='Number of samples')
parser.add_argument('-a', '--alpha', type=float, default=0.85, help='First order AR model parameter')
parser.add_argument('-k', '--latent_dimension', type=int, default=5, help='Dimension of latent states')
parser.add_argument('-sn', '--noise_std', type=float, default=0.5, help='Standard deviation of additive noise for OASIS')
parser.add_argument('-mt', '--mask_type', type=str, default='uniform', help='Mask type: uniform, random or staggered')
args = parser.parse_args()

neurons_count = args.neurons_count
timesteps_count = args.timesteps_count
samples_count = args.samples_count
alpha = args.alpha
latent_dimension = args.latent_dimension
mask_type = args.mask_type

# Generate W and u: they map latent states to a space with dimension = neuron count
W = np.random.randn(neurons_count, latent_dimension)
muBias = 1
sigmaBias = 1
b = np.random.normal(muBias, sigmaBias, size=(neurons_count, 1))

Shi = np.zeros((samples_count, timesteps_count, neurons_count))
Yhi = np.zeros((samples_count, timesteps_count, neurons_count))
rates = np.zeros((samples_count, timesteps_count, neurons_count))

for sample_idx in np.arange(samples_count):
    if sample_idx % 200 == 0:
        # Latent state: not varying with time
        u = np.random.randn(latent_dimension, 1)
        # Bernoulli probability for spikes
        p = np.exp(np.matmul(W, u) + b)/(1 + np.exp(np.matmul(W, u) + b))

    for t in np.arange(timesteps_count):
        Shi[sample_idx, t, :] = np.squeeze(np.random.binomial(n=1, p=p, size=(neurons_count, 1)))
        if t == 0:
            Yhi[sample_idx, t, :] = Shi[sample_idx, t, :]
        else:
            Yhi[sample_idx, t, :] = alpha*Yhi[sample_idx, t-1, :] + Shi[sample_idx, t, :]
        rates[sample_idx, t, :] = np.squeeze(p)

# split train and validation data
 
train_size = int(0.8*samples_count)
random_idx = np.random.choice(samples_count, train_size, replace=False)
valid_sample_idx = [i for i in range(samples_count) if i not in random_idx]
data_dict = {}
data_dict['train_data'] = Yhi[random_idx, :, :]
data_dict['train_truth'] = rates[random_idx, :, :]
data_dict['train_spikes'] = Shi[random_idx, :, :]
data_dict['valid_data'] = Yhi[valid_sample_idx, :, :]
data_dict['valid_truth'] = rates[valid_sample_idx, :, :]
data_dict['valid_spikes'] = Shi[valid_sample_idx, :, :]

# Store it as a h5 file

with h5py.File('calcium2Pdata.h5', 'w') as h5File:
    for k, v in data_dict.items():
        h5File.create_dataset(k, data=np.array(v))


# Generate deconvolved data usng OASIS
# Generate data for all bandwidths
bandwidths = [5, 15, 25, 35, 45, 50]
noise_std = args.noise_std
Yhi_oasis = Yhi + noise_std*np.random.randn(samples_count, timesteps_count, neurons_count)
oasis_data_dict = {}
for bw in bandwidths:   
    # Deconvolve
    deconvolved_spikes = np.full(Yhi_oasis.shape, np.nan)
    denoised_Yhi_oasis = np.full(Yhi_oasis.shape, np.nan)

    for sample_idx in np.arange(samples_count):
        stg_start = 0
        for neuron_idx in np.arange(neurons_count):
            if mask_type == 'uniform':
                sampled_ixs = np.sort(np.int32(np.linspace(0, timesteps_count-1, bw)))
            elif mask_type == 'random':
                sampled_ixs = np.sort(np.random.choice(np.arange(timesteps_count), size=bw, replace=False))
            elif mask_type == 'staggered':
                sampled_ixs = np.sort(np.int32(np.linspace(stg_start, stg_start + timesteps_count-1, bw))%timesteps_count)
                stg_start = stg_start + 1
            delta_t = max([sampled_ixs[i+1] - sampled_ixs[i] for i in range(len(sampled_ixs)-1)])
            c, s, b, g, lam = deconvolve(y = Yhi_oasis[sample_idx, sampled_ixs, neuron_idx], g = (alpha**delta_t,), sn = noise_std, b = 0, penalty=1, optimize_g=0)
            deconvolved_spikes[sample_idx, sampled_ixs, neuron_idx] = s
            denoised_Yhi_oasis[sample_idx, sampled_ixs, neuron_idx] = c
            # c, s, b, g, lam = deconvolve(y = Yhi_oasis[sample_idx, neuron_idx, sampled_ixs_random], g = (0.95,), sn = noise_std, b = 0, penalty=1, optimize_g=1)
            # deconvolved_spikes_random[sample_idx, neuron_idx, sampled_ixs_random] = s
            # denoised_Yhi_oasis_random[sample_idx, neuron_idx, sampled_ixs_random] = c
    smin = np.nanmin(deconvolved_spikes)
    print(f'b = {b}, g = {g}, lam = {lam}')

    plt.figure(figsize=(20,4))
    plt.suptitle(f'OASIS deconvolution, mask: {mask_type}, bw: {bw}')
    plt.subplot(311)
    plt.plot(Yhi_oasis[100, :, 20], label='cal data noisy')
    plt.plot(denoised_Yhi_oasis[100, :, 20], label='cal data denoised')
    plt.legend()
    plt.subplot(312)
    plt.stem(np.arange(timesteps_count), Shi[100, :, 20], 'g', markerfmt='go', label='True spikes')
    plt.legend()
    plt.subplot(313)
    plt.stem(np.arange(timesteps_count), deconvolved_spikes[100, :, 20], 'b', markerfmt='bo', label='deconvolved spikes')
    plt.legend()
    plt.savefig(f'oasis_bw_{bw}_mask_{mask_type}.png')
    plt.show()
    

    oasis_data_dict[str(bw)] = {}
    oasis_data_dict[str(bw)]['smin'] = smin
    oasis_data_dict[str(bw)]['train_data'] = deconvolved_spikes[random_idx, :, :]
    # oasis_data_dict[str(bw)]['train_data_random'] = deconvolved_spikes_random[random_idx, :, :]
    oasis_data_dict[str(bw)]['train_truth'] = rates[random_idx, :, :]
    oasis_data_dict[str(bw)]['train_spikes'] = Shi[random_idx, :, :]
    oasis_data_dict[str(bw)]['train_calcium_data'] = Yhi_oasis[random_idx, :, :]
    oasis_data_dict[str(bw)]['train_denoised_calcium_data'] = denoised_Yhi_oasis[random_idx, :, :]
    # oasis_data_dict[str(bw)]['train_denoised_calcium_data_random'] = denoised_Yhi_oasis_random[random_idx, :, :]

    oasis_data_dict[str(bw)]['valid_data'] = deconvolved_spikes[valid_sample_idx, :, :]
    # oasis_data_dict[str(bw)]['valid_data_random'] = deconvolved_spikes_random[valid_sample_idx, :, :]
    oasis_data_dict[str(bw)]['valid_truth'] = rates[valid_sample_idx, :, :]
    oasis_data_dict[str(bw)]['valid_spikes'] = Shi[valid_sample_idx, :, :]
    oasis_data_dict[str(bw)]['valid_calcium_data'] = Yhi_oasis[valid_sample_idx, :, :]
    oasis_data_dict[str(bw)]['valid_denoised_calcium_data'] = denoised_Yhi_oasis[valid_sample_idx, :, :]
    # oasis_data_dict[str(bw)]['valid_denoised_calcium_data_random'] = denoised_Yhi_oasis_random[valid_sample_idx, :, :]

    # Store it as a h5 file

    with h5py.File('calcium2PdataOasis'+mask_type+str(bw)+'.h5', 'w') as h5File:
        for k, v in oasis_data_dict[str(bw)].items():
            h5File.create_dataset(k, data=np.array(v))

