from nequip.model import model_from_config
from nequip.utils import (
    Output,
    Config,
    instantiate_from_cls_name,
    instantiate,
    save_file,
    load_file,
    load_callable,
    atomic_write,
    finish_all_writes,
    atomic_write_group,
    dtype_from_name,
)

import sys
import inspect
import logging
from copy import deepcopy
from os.path import isfile
from time import perf_counter
from typing import Callable, Optional, Union, Tuple, List
from pathlib import Path

if sys.version_info[1] >= 7:
    import contextlib
else:
    # has backport of nullcontext
    import contextlib2 as contextlib

import numpy as np
import torch
from torch_ema import ExponentialMovingAverage

from nequip.data import DataLoader, AtomicData, AtomicDataDict, AtomicDataset
from nequip.utils.versions import check_code_version

from nequip.train.loss import Loss, LossStat
from nequip.train.metrics import Metrics
from nequip.train._key import ABBREV, LOSS_KEY, TRAIN, VALIDATION
from nequip.train.early_stopping import EarlyStopping
from nequip.data import dataset_from_config

from nequip.model import model_from_config

train_idcs = None
val_idcs = None
n_train = 950
n_val = 50
config = Config.from_file('configs/CSH/l3/tob14_l3.yaml')
dataset = dataset_from_config(config, prefix="dataset")
validation_dataset = None
train_val_split = "random"
dataset_rng = torch.Generator()



if train_idcs is None or val_idcs is None:
    if validation_dataset is None:
        # Sample both from `dataset`:
        total_n = len(dataset)
        if (n_train + n_val) > total_n:
            raise ValueError(
                "too little data for training and validation. please reduce n_train and n_val"
            )
        if train_val_split == "random":
            idcs = torch.randperm(total_n, generator=dataset_rng)
        elif train_val_split == "sequential":
            idcs = torch.arange(total_n)
        else:
            raise NotImplementedError(
                f"splitting mode {train_val_split} not implemented"
            )
        train_idcs = idcs[: n_train]
        val_idcs = idcs[n_train : n_train + n_val]
    else:
        if n_train > len(dataset):
            raise ValueError("Not enough data in dataset for requested n_train")
        if n_val > len(validation_dataset):
            raise ValueError(
                "Not enough data in validation dataset for requested n_val"
            )
        if train_val_split == "random":
            train_idcs = torch.randperm(
                len(dataset), generator=dataset_rng
            )[: n_train]
            val_idcs = torch.randperm(
                len(validation_dataset), generator=self.dataset_rng
            )[: self.n_val]
        elif train_val_split == "sequential":
            train_idcs = torch.arange(self.n_train)
            val_idcs = torch.arange(self.n_val)
        else:
            raise NotImplementedError(
                f"splitting mode {train_val_split} not implemented"
            )
if validation_dataset is None:
    validation_dataset = dataset


dataset_train = dataset.index_select(train_idcs)
dataset_val = validation_dataset.index_select(val_idcs)
shuffle = True
batch_size = 4
validation_batch_size = 4
max_epochs = 1000
exclude_keys = []
dataloader_num_workers= 0
torch_device = torch.device("cuda")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model_from_config(
        config=config, initialize=False, dataset=dataset_train
    )
model.to(torch_device)


dl_kwargs = dict(
            exclude_keys=exclude_keys,
            num_workers=dataloader_num_workers,
            # keep stuff around in memory
            persistent_workers=(
                dataloader_num_workers > 0 and max_epochs > 1
            ),
            # PyTorch recommends this for GPU since it makes copies much faster
            pin_memory=(torch_device != torch.device("cpu")),
            # avoid getting stuck
            timeout=(10 if dataloader_num_workers > 0 else 0),
            # use the right randomness
            generator=dataset_rng,
        )

dl_train = DataLoader(
            dataset=dataset_train,
            shuffle=shuffle,  # training should shuffle
            batch_size=batch_size,
            **dl_kwargs,
        )
dl_val = DataLoader(
            dataset=dataset_val,
            batch_size=batch_size,
            **dl_kwargs,
)

iepoch = 1

dataloaders = {TRAIN: dl_train, VALIDATION: dl_val}
categories = [TRAIN, VALIDATION] if iepoch >= 0 else [VALIDATION]
dataloaders = [dataloaders[c] for c in categories]  # get the right dataloaders for the catagories we actually run


for category, dataset in zip(categories, dataloaders):
    if category == VALIDATION and self.use_ema:
        cm = self.ema.average_parameters()
    else:
        cm = contextlib.nullcontext()
    with cm:
        #reset_metrics()
        n_batches = len(dataset)
        for ibatch, batch in enumerate(dataset):
            print(batch)
            break
    break

rescale_layers = []
outer_layer = model
while hasattr(outer_layer, "unscale"):
    rescale_layers.append(outer_layer)
    outer_layer = getattr(outer_layer, "model", None)

data = batch
#optim.zero_grad(set_to_none=True)
model.train()
data = data.to(torch_device)
data = AtomicData.to_AtomicDataDict(data)
data_unscaled = data
for layer in rescale_layers:
    data_unscaled = layer.unscale(data_unscaled)
# Run model
# We make a shallow copy of the input dict in case the model modifies it

_remove_from_model_input = set()

input_data = {k: v for k, v in data_unscaled.items() if k not in _remove_from_model_input}
out = model(input_data)
print(out)

from thop import profile

flops, params = profile(model, inputs=(input_data,))
print(flops,params)
