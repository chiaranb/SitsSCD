import numpy as np
import torch
import json
from os.path import join
import random
import torchvision
from datetime import date
import pandas as pd
from torch.utils.data import Dataset
from .transforms import random_resize_crop, random_rotate, random_crop, random_flipv, random_fliph
import time

'''SitsDataset is a base class for datasets that handle time series of satellite images.'''
class SitsDataset(Dataset):
    def __init__(self,
                 path,
                 split,
                 domain_shift_type,
                 num_channels,
                 num_classes,
                 img_size,
                 true_size,
                 train_length,
                 ): 
        super(SitsDataset, self).__init__()
        self.path = path
        self.domain_shift_type = domain_shift_type
        self.split = split 
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.img_size = img_size
        self.true_size = true_size
        self.train_length = train_length
        self.monthly_dates = get_monthly_dates_dict()
        self.gt, self.sits_ids = self.load_ground_truth(split)
        self.collate_fn = collate_fn
        self.mean, self.std, self.month_list = None, None, None  # Needs to be defined in subclass
        
        # costruisci indice globale di tutte le patch
        self.indices = []
        n_patches = self.true_size // self.img_size
        for sits_idx, sits_id in enumerate(self.sits_ids):
            for month in range(24):
                for i in range(n_patches):
                    for j in range(n_patches):
                        self.indices.append((sits_idx, month, i, j))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sits_idx, month, i, j = self.indices[idx]
        print(f"Processing index {idx}: sits_idx={sits_idx}, month={month}, i={i}, j={j}")
        sits_id = self.sits_ids[sits_idx]
        if self.split == 'train':
            curr_sits_path = join(self.path, 'train', sits_id)
        else:
            curr_sits_path = join(self.path, 'test', sits_id)

        # Carica immagine di quel mese
        data_patch = self.load_data(sits_idx, sits_id, month, curr_sits_path)
        # Estrai patch
        data_patch = data_patch[:, i*self.img_size:(i+1)*self.img_size,
                                    j*self.img_size:(j+1)*self.img_size]
        gt_patch = self.gt[sits_idx, month,
                           i*self.img_size:(i+1)*self.img_size,
                           j*self.img_size:(j+1)*self.img_size]

        # Normalizza
        data_patch = self.normalize(data_patch)
                
        output = {
            "data": data_patch,     # (C, H, W)
            "gt": gt_patch.long(),  # (H, W)
            "idx": sits_idx,
            "positions": month
        }
        return output

    def load_ground_truth(self, split):
        sits_ids = json.load(open(join(self.path, 'split.json')))[split]
        sits_ids.sort()
        num_sits = len(sits_ids)
        gt = torch.zeros((num_sits, 24, 1024, 1024), dtype=torch.int8)
        for s in range(num_sits):
            gt[s] = torch.tensor(np.load(join(self.path,'labels',f'{sits_ids[s]}.npy')), dtype=torch.int8)
        return gt, sits_ids

    def normalize(self, data):
        return (data - self.mean) / self.std

    def load_data(self, sits_number, sits_id, months, curr_sits_path):
        data, days = None, None
        return data, days


class Muds(SitsDataset):
    def __init__(
            self,
            path,
            domain_shift_type="none",
            split="train",
            num_channels=3,
            num_classes=3,
            img_size=128,
            true_size=1024,
            train_length=6,
    ):
        super(Muds, self).__init__(path=path,
                                   split=split,
                                   domain_shift_type=domain_shift_type,
                                   num_channels=num_channels,
                                   num_classes=num_classes,
                                   img_size=img_size,
                                   true_size=true_size,
                                   train_length=train_length                                 
                                   )  
        """Initializes the dataset.
        Args:
            path (str): path to the dataset
            split (str): split to use (train, val, test)
            domain_shift (bool): if val/test, whether we are in a domain shift setting or not
        """
        # Load the ground truth labels: 0 if there is the timestep, 1 if there is no timestep
        month_list = [(((self.gt[k].float() - 2) ** 2).mean((1, 2)) == 0).int().numpy().tolist() for k in range(self.gt.shape[0])]
        # Add only timestep indices where there is 0 in the month_list
        self.month_list = [[j for j in range(24) if month_list[i][j] == 0] for i in range(len(month_list))]
        self.mean = torch.tensor([119.9347, 105.3608, 77.5125], dtype=torch.float16).reshape(3, 1, 1)
        self.std = torch.tensor([59.5921, 48.2708, 44.7296], dtype=torch.float16).reshape(3, 1, 1)
        self.month_string = [f'{year}_{month}' for year, month in zip(
            [2018 for _ in range(12)] + [2019 for _ in range(12)],
            [f'0{m}' for m in range(1, 10)] + ['10', '11', '12'] + [f'0{m}' for m in range(1, 10)] + ['10', '11',
                                                                                                      '12'])]

    def load_data(self, sits_number, sits_id, month, curr_sits_path):
        data = torch.zeros((self.num_channels, self.true_size, self.true_size), dtype=torch.float16)
        name_rgb = f'{sits_id}_{self.month_string[month]}.jpg'
        if month in self.month_list[sits_number]:
            data = torchvision.io.read_image(join(curr_sits_path, name_rgb))
        return data


class DynamicEarthNet(SitsDataset):
    def __init__(self, path, split="train", domain_shift_type="none", num_channels=4, num_classes=7,
                 img_size=128, true_size=1024, train_length=6, date_aug_range=2):
        super().__init__(path, split, domain_shift_type, num_channels, num_classes, img_size, true_size, train_length)
        self.date_aug_range = date_aug_range
        self.monthly_dates = self.get_monthly_dates_dict()
        self.gt, self.sits_ids = self.load_ground_truth(split)
        self.month_list = [list(range(24)) for _ in range(self.gt.shape[0])]
        self.mean = torch.tensor([83.1029, 80.7615, 69.3328, 133.8648], dtype=torch.float16).reshape(4, 1, 1)
        self.std = torch.tensor([33.2714, 25.5288, 23.9868, 30.5591], dtype=torch.float16).reshape(4, 1, 1)

    def load_data(self, sits_number, sits_id, months, curr_sits_path):
        data = torch.zeros((len(months), self.num_channels, self.true_size, self.true_size), dtype=torch.float16)
        days = [self.random_date_augmentation(month) for month in months]
        name_rgb = [f'{sits_id}_{day}_rgb.jpeg' for day in days]
        name_infra = [f'{sits_id}_{day}_infra.jpeg' for day in days]
        for d, (n_rgb, n_infra) in enumerate(zip(name_rgb, name_infra)):
            data[d, :3] = torchvision.io.read_image(join(curr_sits_path, n_rgb))
            data[d, 3] = torchvision.io.read_image(join(curr_sits_path, n_infra))
        return data, days

    def random_date_augmentation(self, month):
        if self.split == 'train':
            return max(0, random.randint(0, self.date_aug_range*2) - self.date_aug_range + self.monthly_dates[month])
        else:
            return self.monthly_dates[month]
        
        
def collate_fn(batch):
    """Collate function for the dataloader.
    Args:
        batch (list): list of dictionaries with keys "data", "gt", "positions" and "idx"
    Returns:
        dict: dictionary with keys "data", "gt", "positions" and "idx"
    """
    keys = list(batch[0].keys())
    idx = [x["idx"] for x in batch]
    output = {"idx": idx}
    keys.remove("idx")
    for key in keys:
        output[key] = torch.stack([x[key] for x in batch])
    return output

"""Returns a list of monthly dates as indices for the dataset."""
def get_monthly_dates_dict():
    s_date = date(2018, 1, 1)
    e_date = date(2019, 12, 31)
    dates_monthly = [f'{year}-{month}-01' for year, month in zip(
        [2018 for _ in range(12)] + [2019 for _ in range(12)],
        [f'0{m}' for m in range(1, 10)] + ['10', '11', '12'] + [f'0{m}' for m in range(1, 10)] + ['10', '11', '12']
    )] # ['2018-01-01', '2018-02-01', ..., '2019-12-01']
    dates_daily = pd.date_range(s_date, e_date, freq='d').strftime('%Y-%m-%d').tolist() # ['2018-01-01', '2018-01-02', ..., '2019-12-31']
    monthly_dates = [] # [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365, 396, 424, 455, 485, 516, 546, 577, 608, 638, 669, 699]
    i, j = 0, 0
    while i < 730 and j < 24: # 730 days and 24 months (2 years)
        if dates_monthly[j] == dates_daily[i]:
            monthly_dates.append(i)
            j += 1
        i += 1
    return monthly_dates