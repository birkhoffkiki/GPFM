from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import random
import h5py
import numpy as np

class DataSet(Dataset):
    num_classes=7
    def __init__(self, root='/home/jmabq/data/LYSTO', phase='train', transformer=None) -> None:
        super().__init__()
        random.seed(0)
        assert phase in ['train', 'val']
        self.root = root
        self.phase = phase
        self.data_items = self.parser()
        random.shuffle(self.data_items)
        self.transformer = transformer
        
    def parser(self):
        # train      
        ds = h5py.File(os.path.join(self.root, 'training.h5'), 'r')
        x = ds['x'][:]
        y = ds['y'][:]
        print('img shape:', x.shape)
        print('label shape', y.shape)
        data_index = {i:[] for i in range(7)}
        # reset labels
        for i in range(len(y)):
            count_num = y[i]
            if count_num == 0:
                label = 0
            elif 1 <= count_num <=5:
                label = 1
            elif 6 <= count_num <= 10:
                label = 2
            elif 11 <= count_num <= 20: 
                label = 3
            elif 21 <= count_num <= 50:
                label = 4
            elif 51 <= count_num <= 200:
                label = 5
            elif count_num > 200:
                label = 6
            else:
                raise RuntimeError
            y[i] = label
            data_index[label].append(i)
        # 0.8 for train, 0.2 for test
        iter_index = []
        for label, d_index in data_index.items():
            num = int(len(d_index)*0.8)
            print('class {}, num={}'.format(label, num))
            if self.phase == 'train':
                iter_index.extend(d_index[:num])
            else:
                iter_index.extend(d_index[num:])
        
        self.images = x
        self.labels = y
        
        return iter_index
            
    def __len__(self):
        return len(self.data_items)
    
    def __getitem__(self, index):
        index = self.data_items[index]
        
        img = self.images[index]
        img = Image.fromarray(img)
        label = self.labels[index].item()
        # force to resize
        img = img.resize((224, 224))
        if self.transformer:
            img = self.transformer(img)
        return img, (index, torch.tensor(label))