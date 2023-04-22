import torch
import pytorch_lightning as pl
from data import LorenzDataModule
from model import SequentialAutoencoder
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-m', '--mask_type', help="Mask type: random or uniform", type=str, default='random')
parser.add_argument('-ma', '--mask_axis', help="Mask axis: time or neuron", type=str, default='neuron')
parser.add_argument('-dp', '--data_path', help="Data path", type=str, default='loenz_dataset.h5')
parser.add_argument('-lt', '--loss_type', help="loss computation with input or ground truth", type=str, default='input')
args = parser.parse_args()

mask_type = args.mask_type
loss_type = args.loss_type
data_path = args.data_path
mask_axis = args.mask_axis

# for bandwidth in [None, 25, 20, 15, 10, 5, 3, 2]:
if mask_axis == 'neuron':
    for bandwidth in [29, 25, 15, 5, 2]:
        datamodule = LorenzDataModule(data_path, bandwidth=bandwidth, mask_type=mask_type, mask_axis=mask_axis)
        model = SequentialAutoencoder(loss_type=loss_type)
        trainer = pl.Trainer(
            callbacks=[
                pl.callbacks.ModelCheckpoint(monitor='valid_loss'),
            ],
            devices=int(torch.cuda.is_available()), accelerator='gpu'
        )
        trainer.fit(model, datamodule)
elif mask_axis == 'time':
    for bandwidth in [50, 45, 35, 25, 15, 5]:
        if 'Oasis' in data_path:
            data_path = data_path + str(bandwidth) + '.h5'
            datamodule = LorenzDataModule(data_path, bandwidth=bandwidth, mask_type=mask_type, mask_axis=mask_axis, dont_mask=True)
        else:
            datamodule = LorenzDataModule(data_path, bandwidth=bandwidth, mask_type=mask_type, mask_axis=mask_axis)
        model = SequentialAutoencoder(loss_type=loss_type)
        trainer = pl.Trainer(
            callbacks=[
                pl.callbacks.ModelCheckpoint(monitor='valid_loss'),
            ],
            devices=int(torch.cuda.is_available()), accelerator='gpu'
        )
        trainer.fit(model, datamodule)