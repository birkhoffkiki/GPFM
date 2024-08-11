from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import random


class DataSet(Dataset):
    num_classes = 4 
    def __init__(self, root='/home/jmabq/data/PanCancer-TIL', phase='train', transformer=None) -> None:
        super().__init__()
        random.seed(0)
        assert phase in ['train', 'val', 'test']
        self.root = root
        self.phase = phase
        self.cls_names = {name: value for value, name in enumerate(['til-negative', 'til-positive'])}

        self.data_items = self.parser()
        random.shuffle(self.data_items)
        self.transformer = transformer

    def parser(self):
        # train
        data_items = []
        
        csv_name = 'images-tcga-tils-metadata.csv'
        with open(os.path.join(self.root, csv_name)) as f:
            _ = f.readline()
            for line in f:
                partition, _, _, label, path, _ = line.split(',')
                if partition == self.phase:
                    full_path = os.path.join(self.root, path)
                    data_items.append([full_path, self.cls_names[label]])
            
        return data_items
            
    def __len__(self):
        return len(self.data_items)
    
    def __getitem__(self, index):
        path, label = self.data_items[index]
        img = Image.open(path).convert('RGB')
        # force to resize
        img = img.resize((224, 224))
        
        if self.transformer:
            img = self.transformer(img)
        return img, (index, torch.tensor(label))