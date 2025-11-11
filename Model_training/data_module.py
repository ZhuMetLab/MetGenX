"""
# File       : data_module.py
# Time       : 2025/10/23 10:28
# Author     : Hongmiao Wang
# version    : python 3.10
# Description: 
"""

"""
The data module is modified from the following source for general evaluation in MassSpecGym:
https://github.com/pluskal-lab/MassSpecGym
@article{bushuiev2024massspecgym,
      title={MassSpecGym: A benchmark for the discovery and identification of molecules}, 
      author={Roman Bushuiev and Anton Bushuiev and Niek F. de Jonge and Adamo Young and Fleming Kretschmer and Raman Samusevich and Janne Heirman and Fei Wang and Luke Zhang and Kai Dührkop and Marcus Ludwig and Nils A. Haupt and Apurva Kalia and Corinna Brungs and Robin Schmid and Russell Greiner and Bo Wang and David S. Wishart and Li-Ping Liu and Juho Rousu and Wout Bittremieux and Hannes Rost and Tytus D. Mak and Soha Hassoun and Florian Huber and Justin J. J. van der Hooft and Michael A. Stravs and Sebastian Böcker and Josef Sivic and Tomáš Pluskal},
      year={2024},
      eprint={2410.23326},
      url={https://arxiv.org/abs/2410.23326},
      doi={10.48550/arXiv.2410.23326}
}
"""

import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pathlib import Path
from typing import Optional
from torch.utils.data.dataset import Subset
from torch.utils.data.dataloader import DataLoader
class MassSpecDataModule(pl.LightningDataModule):
    """
    Data module containing a mass spectrometry dataset. This class is responsible for loading, splitting, and wrapping
    the dataset into data loaders according to pre-defined train, validation, test folds.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        num_workers: int = 0,
        persistent_workers: bool = True,
        split_pth: Optional[Path] = None,
        **kwargs
    ):
        """
        Args:
            split_pth (Optional[Path], optional): Path to a .tsv file with columns "identifier" and "fold",
                corresponding to dataset item IDs, and "fold", containg "train", "val", "test"
                values. Default is None, in which case the split from the `dataset` is used.
        """
        super().__init__(**kwargs)
        self.dataset = dataset
        self.split_pth = split_pth
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers if num_workers > 0 else False

    def prepare_data(self):
        if self.split_pth is None:
            self.split = self.dataset.metadata[["identifier", "fold"]]
        else:
            # NOTE: custom split is not tested
            self.split = pd.read_csv(self.split_pth, sep="\t")
            if set(self.split.columns) != {"identifier", "fold"}:
                raise ValueError('Split file must contain "id" and "fold" columns.')
            self.split["identifier"] = self.split["identifier"].astype(str)
            if set(self.dataset.metadata["identifier"]) != set(self.split["identifier"]):
                raise ValueError(
                    "Dataset item IDs must match the IDs in the split file."
                )

        self.split = self.split.set_index("identifier")["fold"]
        if not set(self.split) <= {"train", "val", "test"}:
            raise ValueError(
                '"Folds" column must contain only "train", "val", or "test" values.'
            )

    def setup(self, stage=None):
        split_mask = self.split.loc[self.dataset.metadata["identifier"]].values
        if stage == "fit" or stage is None:
            self.train_dataset = Subset(
                self.dataset, np.where(split_mask == "train")[0]
            )
            self.val_dataset = Subset(self.dataset, np.where(split_mask == "val")[0])
        if stage == "test":
            self.test_dataset = Subset(self.dataset, np.where(split_mask == "test")[0])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            collate_fn=self.dataset.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            collate_fn=self.dataset.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            collate_fn=self.dataset.collate_fn,
        )
