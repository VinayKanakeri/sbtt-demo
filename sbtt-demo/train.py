import torch
import pytorch_lightning as pl
from data import LorenzDataModule
from model import SequentialAutoencoder
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-m', '--mask_type', help="Mask type: random or uniform", type=str, default='random')
parser.add_argument('-dp', '--data_path', help="Data path", type=str, default='loenz_dataset.h5')
parser.add_argument('-lt', '--loss_type', help="loss computation with input or ground truth", type=str, default='input')
args = parser.parse_args()

mask_type = args.mask_type
loss_type = args.loss_type
data_path = args.data_path

# for bandwidth in [None, 25, 20, 15, 10, 5, 3, 2]:
for bandwidth in [None, 25, 15, 5, 2]:
    datamodule = LorenzDataModule(data_path, bandwidth=bandwidth, mask_type=mask_type)
    model = SequentialAutoencoder(loss_type=loss_type)
    trainer = pl.Trainer(
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor='valid_loss'),
        ],
        devices=int(torch.cuda.is_available()), accelerator='gpu'
    )
    trainer.fit(model, datamodule)
