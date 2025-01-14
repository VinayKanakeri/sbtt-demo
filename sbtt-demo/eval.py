import os
import torch
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from glob import glob

from data import LorenzDataModule
from model import SequentialAutoencoder
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--eval_dir', help="Directory for stored checkpoints", type=str, default='lightning_logs')
parser.add_argument('-m', '--mask_type', help="Mask type: random or uniform", type=str, default='random')
parser.add_argument('-ma', '--mask_axis', help="Mask axis: time or neuron", type=str, default='neuron')
parser.add_argument('-dp', '--data_path', help="Data path", type=str, default='lorenz_dataset.h5')
args = parser.parse_args()

eval_dir = args.eval_dir
mask_type = args.mask_type
data_path = args.data_path
mask_axis = args.mask_axis

train_info = eval_dir.split('lightning_logs_')[1]
train_mask = train_info.split('_')[0]
dataset = train_info.split('_')[1]
sampling_axis = train_info.split('_')[2]

# model_data = {
#     29: eval_dir + '/version_0',
#     25: eval_dir + '/version_1',
#     20: eval_dir + '/version_2',
#     15: eval_dir + '/version_3',
#     10: eval_dir + '/version_4',
#     5: eval_dir + '/version_5',
#     3: eval_dir + '/version_6',
#     2: eval_dir + '/version_7',
# }

if mask_axis == 'neuron':
    TOTAL_OBS = 29
    model_data = {
        29: eval_dir + '/version_0',
        25: eval_dir + '/version_1',
        15: eval_dir + '/version_2',
        5: eval_dir + '/version_3',
        2: eval_dir + '/version_4',
    }
# elif mask_axis == 'time':
#     TOTAL_OBS = 50
#     model_data = {
#         50: eval_dir + '/version_0',
#         45: eval_dir + '/version_1',
#         35: eval_dir + '/version_2',
#         25: eval_dir + '/version_3',
#         15: eval_dir + '/version_4',
#         5: eval_dir + '/version_5',
#     } 
elif mask_axis == 'time':
    TOTAL_OBS = 50
    model_data = {
        50: eval_dir + '/version_0',
        25: eval_dir + '/version_1',
        15: eval_dir + '/version_2',
        5: eval_dir + '/version_3',
    } 
results = []
for bandwidth, model_dir in model_data.items():
    # Load the model
    ckpt_pattern = os.path.join(model_dir, 'checkpoints/*.ckpt')
    print(f'CKPT pattern: {ckpt_pattern}')
    ckpt_path = sorted(glob(ckpt_pattern))[0]
    model = SequentialAutoencoder.load_from_checkpoint(ckpt_path)
    if 'Oasis' in data_path:
        data_path_oasis = data_path + str(bandwidth) + '.h5'
        datamodule = LorenzDataModule(data_path_oasis, bandwidth=bandwidth, mask_type=mask_type, mask_axis=mask_axis, dont_mask=True)
    else:
        datamodule = LorenzDataModule(data_path, bandwidth=bandwidth, mask_type=mask_type, mask_axis=mask_axis)
    # Create a trainer
    trainer = pl.Trainer(logger=False, devices=int(torch.cuda.is_available()), accelerator='gpu')
    # trainer = pl.Trainer(logger=False)
    result = trainer.validate(model, datamodule)[0]
    print(f'RESULT: {result}')
    result['drop_ratio'] = 1 - bandwidth / TOTAL_OBS
    results.append(result)
    # Plot examples
    fig, axes = plt.subplots(figsize=(20, 15), nrows=5, ncols=3, sharex=True, sharey=True)
    dataloader = datamodule.val_dataloader()
    for i, (ax_row, batch) in enumerate(zip(axes, dataloader)):
        valid_spikes, valid_truth = batch
        if 'Oasis' in data_path:
            alpha_beta_nl, q_nl = model(valid_spikes)
            rates = q_nl*(alpha_beta_nl[..., ::2]*alpha_beta_nl[..., 1::2])
            valid_rates = rates.detach().cpu().numpy() 
        else:
            valid_logrates = model(valid_spikes)
            valid_rates = torch.exp(valid_logrates).detach().numpy() * 0.05
        # Plot just the first sample from each batch
        if i == 0:
            ax_row[0].imshow(valid_spikes[0].T)            
            ax_row[1].imshow(valid_truth[0].T)
            ax_row[2].imshow(valid_rates[0].T)
            ax_row[0].set_title('Input spikes')
            ax_row[1].set_title('True rates')
            ax_row[2].set_title('Recovered rates')
        else:
            ax_row[0].imshow(valid_spikes[0].T)            
            ax_row[1].imshow(valid_truth[0].T)
            ax_row[2].imshow(valid_rates[0].T)
    fig.suptitle(f'Rate recovery - data: {dataset}, sampling axis: {sampling_axis}, train mask: {train_mask}, eval mask: {mask_type}, bandwidth: {bandwidth}')
    plt.tight_layout()
    plt.savefig(train_info + '_eval_' + mask_type + '_samples_' + str(bandwidth) + '.png')
results = pd.DataFrame(results)
plt.figure(figsize=(20, 15))
plt.plot(results.drop_ratio, results.valid_mse, marker='o', color='slateblue')
# plt.xlim(-0.1, 1.1)
# plt.ylim(0, 1)
plt.xlabel('Fraction dropped samples')
plt.ylabel('MSE')
plt.title(f'MSE - data: {dataset}, sampling axis: {sampling_axis}, train mask: {train_mask}, eval mask: {mask_type}')
plt.grid()
plt.tight_layout()
plt.savefig(train_info + '_eval_' + mask_type + '_result.png')
np.save(train_info + '_eval_' + mask_type + '_result_mse.npy', np.array(results.valid_mse))
np.save(train_info + '_eval_' + mask_type + '_result_drop_ratio.npy', np.array(results.drop_ratio))
plt.show()
