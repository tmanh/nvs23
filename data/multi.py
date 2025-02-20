import os
import torch
import torch.nn.functional as F

import random
import numpy as np

from torch.utils.data import DataLoader
from .arkit import ArkitDataset
from .dtu import DTU_Dataset
from .wildrgb import WildRGBDataset


class MultiDataLoader:
    def __init__(self, file_list, batch_size, shuffle, num_workers):
        """
        Args:
            dataloader_list (list): List of PyTorch DataLoaders, one for each dataset.
            weights (list): List of weights corresponding to each DataLoader. 
                            Typically these are the lengths (number of samples) of each dataset.
        """
        self.dtu          = DTU_Dataset(os.path.join(file_list, 'dtu_down_4'))
        self.arkitscenes  = ArkitDataset(os.path.join(file_list, 'arkitscenes'))
        self.parkitscenes = ArkitDataset(os.path.join(file_list, 'p_arkitscenes'))
        self.wildrgb      = WildRGBDataset(os.path.join(file_list, 'wildrgb'))

        self.len_dtu = len(self.dtu)
        self.len_arkit = len(self.arkitscenes)
        self.len_parkit = len(self.parkitscenes)
        self.len_wildrgb = len(self.wildrgb)

        self.count = 0

        self.datasets = [self.dtu, self.arkitscenes, self.parkitscenes, self.wildrgb]
        self.weights = [self.len_dtu, self.len_arkit, self.len_parkit, self.len_wildrgb]

        self.dtu_loader      = DataLoader(self.dtu, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.arkit_loader    = DataLoader(self.arkitscenes, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.parkit_loader   = DataLoader(self.parkitscenes, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.wildrgb_loader  = DataLoader(self.wildrgb, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        self.dataloader_list = [self.dtu_loader, self.arkit_loader, self.parkit_loader, self.wildrgb_loader]

        # Create an iterator for each dataloader.
        self._reset_iterators()

    def _reset_iterators(self):
        self.iterators = [iter(dl) for dl in self.dataloader_list]

    def __iter__(self):
        # Reset iterators at the start of a new epoch.
        self._reset_iterators()
        self.count = 0
        return self

    def __next__(self):
        if self.count >= len(self):
            raise StopIteration

        # Randomly choose one of the DataLoaders based on the provided weights.
        chosen_idx = random.choices(range(len(self.dataloader_list)), weights=self.weights, k=1)[0]
        try:
            batch = next(self.iterators[chosen_idx])
        except StopIteration:
            # If one iterator is exhausted, reset it.
            self.iterators[chosen_idx] = iter(self.dataloader_list[chosen_idx])
            batch = next(self.iterators[chosen_idx])
        
        self.count += 1
        
        # Return both the batch and the dataset index (if needed for further processing)
        return batch
    
    def __len__(self):
        return self.len_dtu + self.len_arkit + self.len_parkit + self.len_wildrgb


class MultiDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, val=None):
        self.file_list = file_list
        self.which_data = 0

        self.dtu = DTU_Dataset(os.path.join(file_list, 'dtu_down_4'))
        self.arkitscenes = ArkitDataset(os.path.join(file_list, 'arkitscenes'))
        self.parkitscenes = ArkitDataset(os.path.join(file_list, 'p_arkitscenes'))
        self.wildrgb = WildRGBDataset(os.path.join(file_list, 'wildrgb'))

        self.len_dtu = len(self.dtu)
        self.len_arkit = len(self.arkitscenes)
        self.len_parkit = len(self.parkitscenes)
        self.len_wildrgb = len(self.wildrgb)

        self.change_data()

    def __len__(self):
        return self.len_dtu + self.len_arkit + self.len_parkit + self.len_wildrgb

    def __getitem__(self, index):
        if self.which_data == 0:
            data = self.dtu
        elif self.which_data == 1:
            data = self.arkitscenes
        elif self.which_data == 2:
            data = self.parkitscenes
        elif self.which_data == 3:
            data = self.wildrgb

        return data[index % len(data)]
    
    def change_data(self):
        weights = [self.len_dtu, self.len_arkit, self.len_parkit, self.len_wildrgb]
        self.which_data = random.choices([0, 1, 2, 3], weights=weights, k=1)[0]
