from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import random
import json


class DataSet(Dataset):
    num_classes=9
    def __init__(self, root='/home/jmabq/data/WSSS4LUAD', phase='train', transformer=None) -> None:
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
        dir = os.path.join(self.root, 'data')
        image_names = [i for i in os.listdir(dir) if '.png' in i]

        temp_dict = {k: [] for k in [0, 1, 2]}

        for name in image_names:
            prefix = name.split('.')[0]
            label_list = prefix.split('-')[-1]
            label_list = [int(i) for i in json.loads(label_list)]
            # print(label_list)
            full_path = os.path.join(dir, name)
            label = label_list.index(1)
            temp_dict[label].append([full_path, label])
                
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
        # force to resize to avoid cat problem
        img = img.resize((224, 224))
        if self.transformer:
            img = self.transformer(img)
        return img, (index, torch.tensor(label))