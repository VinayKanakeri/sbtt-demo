import os
import torch
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from glob import glob

from data import LorenzDataModule
from model import SequentialAutoencoder
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--eval_dir', help="Directory for stored checkpoints", type=str, default='lightning_logs')
parser.add_argument('-m', '--mask_type', help="Mask type: random or uniform", type=str, default='random')
parser.add_argument('-dp', '--data_path', help="Data path", type=str, default='loenz_dataset.h5')
args = parser.parse_args()

eval_dir = args.eval_dir
mask_type = args.mask_type
data_path = args.data_path

TOTAL_OBS = 29
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

model_data = {
    29: eval_dir + '/version_0',
    25: eval_dir + '/version_1',
    15: eval_dir + '/version_2',
    5: eval_dir + '/version_3',
    2: eval_dir + '/version_4',
}

results = []
for bandwidth, model_dir in model_data.items():
    # Load the model
    ckpt_pattern = os.path.join(model_dir, 'checkpoints/*.ckpt')
    ckpt_path = sorted(glob(ckpt_pattern))[0]
    model = SequentialAutoencoder.load_from_checkpoint(ckpt_path)
    datamodule = LorenzDataModule(data_path, bandwidth=bandwidth, mask_type=mask_type)
    # Create a trainer
    trainer = pl.Trainer(logger=False, devices=int(torch.cuda.is_available()), accelerator='gpu')
    # trainer = pl.Trainer(logger=False)
    result = trainer.validate(model, datamodule)[0]
    print(f'RESULT: {result}')
    result['drop_ratio'] = 1 - bandwidth / TOTAL_OBS
    results.append(result)
    # Plot examples
    fig, axes = plt.subplots(nrows=5, ncols=3, sharex=True, sharey=True)
    dataloader = datamodule.val_dataloader()
    for i, (ax_row, batch) in enumerate(zip(axes, dataloader)):
        valid_spikes, valid_truth = batch
        valid_logrates = model(valid_spikes)
        valid_rates = torch.exp(valid_logrates).detach().numpy() * 0.05
        # Plot just the first sample from each batch
        ax_row[0].imshow(valid_spikes[0].T)
        ax_row[1].imshow(valid_truth[0].T)
        ax_row[2].imshow(valid_rates[0].T)
    fig.suptitle(f'rate recovery, bandwidth: {bandwidth}')
    plt.tight_layout()
results = pd.DataFrame(results)
plt.figure(figsize=(3, 2.5))
plt.plot(results.drop_ratio, results.valid_mse, marker='o', color='slateblue')
# plt.xlim(-0.1, 1.1)
# plt.ylim(0, 1)
plt.xlabel('Fraction dropped samples')
plt.ylabel('MSE')
plt.grid()
plt.tight_layout()
plt.savefig('result.png')
plt.show()
