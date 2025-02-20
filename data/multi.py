import os
import torch
import torch.nn.functional as F

import glob
import imageio
import numpy as np

from .arkit import ArkitDataset
from .dtu import DTU_Dataset
from .wildrgb import WildRGBDataset


class MultiDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, val=None):
        self.file_list = file_list
        self.which_data = 0
        self.dtu = DTU_Dataset(os.path.join(file_list, 'dtu_down_4'))
        self.arkitscenes = ArkitDataset(os.path.join(file_list, 'arkitscenes'))
        self.parkitscenes = ArkitDataset(os.path.join(file_list, 'p_arkitscenes'))
        self.wildrgb = WildRGBDataset(os.path.join(file_list, 'wildrgb'))

    def __len__(self):
        self.len_dtu = len(self.dtu)
        self.len_arkit = len(self.arkitscenes)
        self.len_parkit = len(self.parkitscenes)
        self.len_wildrgb = len(self.wildrgb)
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