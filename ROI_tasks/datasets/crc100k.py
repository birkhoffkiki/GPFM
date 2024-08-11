from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import random


class DataSet(Dataset):
    num_classes=2
    def __init__(self, root='/home/jmabq/data/DRYAD', phase='train', transformer=None) -> None:
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
        postfix = 'NCT-CRC-HE-100K-NONORM' if self.phase == 'train' else 'CRC-VAL-HE-7K'
        p = os.path.join(self.root, postfix)
        labels = sorted(os.listdir(p))
        self.cls_names = {name: value for value, name in enumerate(labels)}
        data_items = []
        for label in labels:
            dir = os.path.join(p, label)
            image_names = os.listdir(dir)
            for name in image_names:
                full_path = os.path.join(dir, name)
                data_items.append([full_path, self.cls_names[label]])
        return data_items
            
    def __len__(self):
        return len(self.data_items)
    
    def __getitem__(self, index):
        path, label = self.data_items[index]
        try:
            img = Image.open(path).copy()
        except:
            print('failed to read:', path)
        if self.transformer:
            img = self.transformer(img)
        return img, (index, torch.tensor(label))


if __name__ == "__main__":
    import json
    dataset = DataSet(phase='val')
    with open('/home/jmabq/data/CRC-100K/val_order.json', 'w') as f:
        json.dump(dataset.data_items, f)

    dataset = DataSet(phase='train')
    with open('/home/jmabq/data/CRC-100K/train_order.json', 'w') as f:
        json.dump(dataset.data_items, f)