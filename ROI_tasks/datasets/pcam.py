import numpy as np
import h5py
from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import random


class DataSet(Dataset):
    num_classes=5
    def __init__(self, root='/home/jmabq/data/PCAM', phase='train', transformer=None) -> None:
        super().__init__()
        random.seed(0)
        assert phase in ['train', 'val', 'test']
        postfix = {'train': 'train', 'val': 'valid', 'test': 'test'}
        
        x_path = os.path.join(root, 'camelyonpatch_level_2_split_{}_x.h5'.format(postfix[phase]))
        y_path = os.path.join(root, 'camelyonpatch_level_2_split_{}_y.h5'.format(postfix[phase]))
        
        self.images = h5py.File(x_path).get('x')
        self.labels = h5py.File(y_path).get('y')
        self.transformer = transformer
        
            
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        img = self.images[index]
        img = Image.fromarray(img)
        label = self.labels[index].item()
        # force to resize
        img = img.resize((224, 224))
        if self.transformer:
            img = self.transformer(img)
        return img, (index, torch.tensor(label))
        