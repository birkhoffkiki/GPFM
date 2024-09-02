import os
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
from datasets.dataset_survival import Generic_WSI_Survival_Dataset, Generic_MIL_Survival_Dataset
from datasets import get_survival_dataset, get_subtying_dataset

import argparse
import numpy as np


parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default= 1.0,
                    help='fraction of labels (default: 1)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=10,
                    help='number of splits (default: 10)')
parser.add_argument('--task', type=str)
parser.add_argument('--prefix', type=str, default='splits')
parser.add_argument('--val_frac', type=float, default= 0.1,
                    help='fraction of labels for validation (default: 0.1)')
parser.add_argument('--test_frac', type=float, default= 0.1,
                    help='fraction of labels for test (default: 0.1)')
parser.add_argument('--backbone', type=str, default='resnet50')
parser.add_argument('--mode', type = str, choices=['path', 'omic', 'pathomic', 'cluster'], default='path', help='which modalities to use')
parser.add_argument('--apply_sig', action='store_true', default=False, help='Use genomic features as signature embeddings')
parser.add_argument('--apply_sigfeats',  action='store_true', default=False, help='Use genomic features as tabular features.')

args = parser.parse_args()

if 'survival' in args.task:
    args.n_classes=4
    dataset = get_survival_dataset(args.task)
else:
    dataset = get_subtying_dataset(args.task)
    args.n_classes = len(dataset.label_dict)

    
print('patient cls ids:', dataset.patient_cls_ids)
num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
val_num = np.round(num_slides_cls * args.val_frac).astype(int)
test_num = np.round(num_slides_cls * args.test_frac).astype(int)
print('Val num:', val_num, 'test_num:', test_num)

if __name__ == '__main__':
    if args.label_frac > 0:
        label_fracs = [args.label_frac]
    else:
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    for lf in label_fracs:
        split_dir = f'{args.prefix}/'+ str(args.task) + '_{}'.format(int(lf * 100))
        os.makedirs(split_dir, exist_ok=True)
        dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf)

        for i in range(args.k):
            dataset.set_splits()
            splits = dataset.return_splits(None, from_id=True)
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
            
            if not 'survival' in args.task:
                descriptor_df = dataset.test_split_gen(return_descriptor=True)
                descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))



