from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import random
from glob import glob

class DataSet(Dataset):
    num_classes=4
    def __init__(self, root='/home/jmabq/data/PanCancer-TCGA/data', phase='train', transformer=None) -> None:
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
        labels = os.listdir(self.root)
        
        self.cls_names = {name: value for value, name in enumerate(labels)}
        temp_dict = {k: [] for k, _ in self.cls_names.items()}
        
        for label in labels:
            p = os.path.join(self.root, label, '0', '*', '*.jpg') # consider the 0.5um /pixel resolution
            image_names = glob(p, recursive=True)
            for full_path in image_names:
                temp_dict[label].append([full_path, self.cls_names[label]])
                
        temp_dict = {k: sorted(v) for k, v in temp_dict.items()} 
        for k, v in temp_dict.items():
            random.shuffle(v)
        
        final = []
        _total = 0
        for k, v in temp_dict.items():
            length = int(len(v)*0.8)
            _total += len(v)
            print('{} ({})'.format(k, len(v)))
            if self.phase == 'train':
                final.extend(v[:length])
            else:
                final.extend(v[length:])
        print('Total images:', _total)
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

if __name__ == '__main__':
    dataset = DataSet(phase='train')
    print('Total:', len(dataset))