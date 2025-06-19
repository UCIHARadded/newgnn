import os
import numpy as np
import torch
import torch.nn.functional as F
from datautil.actdata.util import *
from datautil.util import mydataset, Nmax

class ActList(mydataset):
    def __init__(self, args, dataset, root_dir, people_group, group_num, transform=None, target_transform=None, pclabels=None, pdlabels=None, shuffle_grid=True):
        super(ActList, self).__init__(args)
        self.domain_num = 0
        self.dataset = dataset
        self.task = 'cross_people'
        self.transform = transform
        self.target_transform = target_transform
        data_root = os.path.abspath(root_dir)
        x, cy, py, sy = loaddata_from_numpy(self.dataset, self.task, data_root)
        self.people_group = people_group
        self.position = np.sort(np.unique(sy))
        self.comb_position(x, cy, py, sy)
        self.x = self.x[:, :, np.newaxis, :]
        
        # Add normalization transform
        self.transform = self.get_default_transform()
        
        self.x = torch.tensor(self.x).float()
        if pclabels is not None:
            self.pclabels = pclabels
        else:
            self.pclabels = np.ones(self.labels.shape)*(-1)
        if pdlabels is not None:
            self.pdlabels = pdlabels
        else:
            self.pdlabels = np.ones(self.labels.shape)*(0)
        self.tdlabels = np.ones(self.labels.shape)*group_num
        self.dlabels = np.ones(self.labels.shape) * \
            (group_num-Nmax(args, group_num))

    def get_default_transform(self):
        return transforms.Compose([
            transforms.Lambda(self.normalize_emg)
        ])
    
    def normalize_emg(self, x):
        """Normalize EMG signals to [-1, 1] range"""
        min_val = x.min()
        max_val = x.max()
        return 2 * (x - min_val) / (max_val - min_val + 1e-8) - 1

    def comb_position(self, x, cy, py, sy):
        for i, peo in enumerate(self.people_group):
            index = np.where(py == peo)[0]
            tx, tcy, tsy = x[index], cy[index], sy[index]
            for j, sen in enumerate(self.position):
                index = np.where(tsy == sen)[0]
                if j == 0:
                    ttx, ttcy = tx[index], tcy[index]
                else:
                    ttx = np.hstack((ttx, tx[index]))
            if i == 0:
                self.x, self.labels = ttx, ttcy
            else:
                self.x, self.labels = np.vstack(
                    (self.x, ttx)), np.hstack((self.labels, ttcy))
                
        # Add data verification
        print(f"Data range before normalization: min={self.x.min()}, max={self.x.max()}")
        print(f"Class distribution: {np.bincount(self.labels.astype(int))}")

    def set_x(self, x):
        self.x = x
