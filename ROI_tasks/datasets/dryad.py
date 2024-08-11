import cv2
import numpy as np
import os
from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import random
from glob import glob


class DataSet(Dataset):
    num_classes=2
    def __init__(self, root='/home/jmabq/data/DRYAD', phase='train', transformer=None) -> None:
        super().__init__()
        random.seed(0)
        assert phase in ['train', 'val']
        
        self.cohorts = {'train': ['TCGA', 'CWRU', 'CINJ'],
                        'val': ['HUP']}
        
        self.root = root
        self.phase = phase
        self.data_items = self.parser()
        random.shuffle(self.data_items)
        self.transformer = transformer
        
    def parser(self):
        # train      
        labels = ['invasive', 'non_invasive']
        self.cls_names = {name: value for value, name in enumerate(labels)}
        
        temp_dict = {k: [] for k, _ in self.cls_names.items()}
        
        for label in labels:
            for cohort in self.cohorts[self.phase]:
                dir = os.path.join(self.root, cohort, label)
                image_names = glob(os.path.join(dir, '**.png'), recursive=True)
                
                for full_path in image_names:
                    temp_dict[label].append([full_path, self.cls_names[label]])
                
        temp_dict = {k: sorted(v) for k, v in temp_dict.items()} 
        for k, v in temp_dict.items():
            random.shuffle(v)
        
        final = []
        for k, v in temp_dict.items():
            final.extend(v)
            
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
    
    def __repr__(self) -> str:
        print(self.cls_names)
        result_dict = {v: 0 for k, v in self.cls_names.items()}
        for _, label in self.data_items:
            result_dict[label] += 1
        return '{}'.format(result_dict)


def extract_positive_examples(save_root, img_root, mask_root, patch=448):
    file_names = os.listdir(mask_root)
    for name in file_names:
        ori_mask = cv2.imread(os.path.join(mask_root, name))
        if 'CINJ' in mask_root:
            _name = name.split('.')[0]+'_idx5.png'
        elif 'HUP' in mask_root:
            _name = name.split('_')[0]+'_idx5.png'
        else:
            _name = name
        ori_img = cv2.imread(os.path.join(img_root, _name))
        w, h, _ = ori_mask.shape
        for x in range(0, w, patch):
            for y in range(0, h, patch):
                if x+patch > w or y + patch > h:
                    print('edge, skip..')
                    continue
                mask = ori_mask[x:x+patch, y:y+patch]
                print(mask.shape)
                if mask.mean() > 0.05*255:
                    # invasive, postive example
                    p = os.path.join(save_root, 'invasive', name+'{}_{}.png'.format(x, y))
                else:
                    p = os.path.join(save_root, 'non_invasive', name+'{}_{}.png'.format(x, y))
                _temp_root = os.path.split(p)[0]
                if not os.path.exists((_temp_root)):
                    os.makedirs(_temp_root)
                
                img = ori_img[x:x+patch, y:y+patch]
                if img.mean() > 235:
                    print('Background, skip...')
                    # skip, background
                    continue
                try:
                    cv2.imwrite(p, img)
                except:
                    print('failed to save img, skip')
                

if __name__ == '__main__':
    save_root = '/home/jmabq/data/DRYAD/CWRU'
    img_root = '/home/jmabq/data/DRYAD/CWRU_imgs_idx8'
    mask_root = '/home/jmabq/data/DRYAD/CWRU_masks'
    
    extract_positive_examples(save_root=save_root, img_root=img_root, mask_root=mask_root,patch=448)
    