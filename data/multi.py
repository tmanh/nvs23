import os
import torch
import torch.nn.functional as F

import glob
import imageio
import numpy as np

from .arkit import ArkitDataset
from .dtu import DTU_Dataset


class MultiDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, val=None):
        self.file_list = file_list
        self.which_data = 0
        self.dtu = DTU_Dataset(os.path.join(file_list, 'dtu_down_4'))
        self.arkitscenes = ArkitDataset(os.path.join(file_list, 'arkitscenes'))

    def __len__(self):
        self.len_dtu = len(self.dtu)
        self.len_arkit = len(self.arkitscenes)
        return self.len_dtu + self.len_arkit

    def __getitem__(self, index):
        if self.which_data == 0:
            data = self.dtu
        elif self.which_data == 1:
            data = self.arkitscenes

        return data[index % len(data)]