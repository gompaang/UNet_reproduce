import os
import numpy as np
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, data_dir, transform=None, seed=None):
        self.data_dir = data_dir
        self.transform = transform
        self.seed = seed

        self.data_dir_input = self.data_dir + '/input'
        self.data_dir_label = self.data_dir + '/label'

        self.list_data_input = os.listdir(self.data_dir_input)
        self.list_data_label = os.listdir(self.data_dir_label)

    def len(self):
        return len(self.data_label)

    def __getitem__(self, index):
        input = np.load(os.path.join(self.data_dir_input, self.list_data_input[index]))
        label = np.load(os.path.join(self.data_dir_label, self.list_data_label[index]))

        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label}

        if self.transform:
            torch.manual_seed(self.seed)
            data['input'] = self.transform(data['input'])
            data['label'] = self.transform(data['label'])

        return data