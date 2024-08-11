from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import random


class DataSet(Dataset):
    num_classes=4
    def __init__(self, root='/home/jmabq/data/CCRCC-TCGA_HEL', phase='train', transformer=None) -> None:
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
        labels = ['blood', 'cancer', 'normal', 'stroma']
        self.cls_names = {name: value for value, name in enumerate(labels)}
        temp_dict = {k: [] for k, _ in self.cls_names.items()}
        
        for label in labels:
            dir = os.path.join(self.root, label)
            image_names = os.listdir(dir)
            for name in image_names:
                full_path = os.path.join(dir, name)
                temp_dict[label].append([full_path, self.cls_names[label]])
        temp_dict = {k: sorted(v) for k, v in temp_dict.items()} 
        for k, v in temp_dict.items():
            random.shuffle(v)
        
        final = []
        for k, v in temp_dict.items():
            length = int(len(v)*0.8)
            if self.phase == 'train':
                final.extend(v[:length])
            else:
                final.extend(v[length:])
                
        return final
            
    def __len__(self):
        return len(self.data_items)
    
    def __getitem__(self, index):
        path, label = self.data_items[index]
        img = Image.open(path)
        # force to resize
        img = img.resize((224, 224))
        if self.transformer:
            img = self.transformer(img)
        return img, (index, torch.tensor(label))