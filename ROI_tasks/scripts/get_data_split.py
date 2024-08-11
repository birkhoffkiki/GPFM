import torch
import os


dataset = 'CCRCC-TCGA_HEL'
root = '/home/jmabq/data/'

train = os.path.join(root, dataset, 'features/resnet50/train.pt')
train = torch.load(train)
print('Train:', len(train['label']))

train = os.path.join(root, dataset, 'features/resnet50/val.pt')
train = torch.load(train)
print('Val:', len(train['label']))

try:
    train = os.path.join(root, dataset, 'features/resnet50/test.pt')
    train = torch.load(train)
    print('Test:', len(train['label']))
except:
    pass