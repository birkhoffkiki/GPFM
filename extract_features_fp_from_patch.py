import torch
import os
import time
from torch.utils.data import DataLoader, Dataset
from models import get_custom_transformer, get_model

import argparse
from utils.file_utils import save_hdf5
from multiprocessing import Process
import h5py
from PIL import Image
import pandas as pd


class Dataset_All_Bags(Dataset):

	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path, dtype={'case_id': str, 'slide_id': str})
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['dir'][idx], self.df['slide_id'][idx]


class PatchDataset(Dataset):
    def __init__(self, img_root, patch_h5_path, transform=None) -> None:
        super().__init__()
        self.img_root = img_root
        self.coords = h5py.File(patch_h5_path)['coords']
        actual_files = os.listdir(img_root)
        self.transform = transform
        assert len(actual_files) + 1 >= len(self.coords), 'real patch {} not match h5 patch number {}'.format(len(actual_files), len(self.coords))
    
    def __len__(self):
        return len(self.coords)
    
    def __getitem__(self, index):
        x, y = self.coords[index]
        img_name = '{}_{}_512_512.jpg'.format(x, y)
        p = os.path.join(self.img_root, img_name)
        img = Image.open(p)
        if self.transform is not None:
            img = self.transform(img)

        return img


def save_feature(path, feature):
    s = time.time()
    torch.save(feature, path)
    e = time.time()
    print('Feature is sucessfully saved at: {}, cost: {:.1f} s'.format(path, e-s))


def save_feature_subprocess(path, feature):
    kwargs = {'feature': feature, 'path': path}
    process = Process(target=save_feature, kwargs=kwargs)
    process.start()


def light_compute_w_loader(loader, model, print_every=20):

    features_list = []
    _start_time = time.time()
    for count, batch in enumerate(loader):
        with torch.no_grad():	
            if count % print_every == 0:
                batch_time = time.time()
                print('batch {}/{}, {} files processed, used_time: {} s'.format(
                    count, len(loader), count * len(batch), batch_time - _start_time))

            batch = batch.to(device, non_blocking=True)
            features = model(batch)
            features = features.cpu()
            features_list.append(features)
    features = torch.cat(features_list, dim=0)
    return features


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--patch_img_dir', type=str, default='')
parser.add_argument('--data_h5_dir', type=str, default='')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--skip_partial', type=str, default= 'no')
parser.add_argument('--model', type=str)
parser.add_argument('--datatype', type=str)

args = parser.parse_args()



if __name__ == '__main__':
    process_start_time = time.time()
    
    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError

    bags_dataset = Dataset_All_Bags(csv_path)
    
    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'pt_files', args.model), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'h5_files', args.model), exist_ok=True)
    dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files', args.model))

    print('loading model checkpoint:', args.model)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Device:{}, GPU Count:{}'.format(device.type, torch.cuda.device_count()))

    model = get_model(args.model, device, torch.cuda.device_count())
    custom_transformer = get_custom_transformer(args.model)

    total = len(bags_dataset)
        
    # check the exists wsi
    exist_idxs = []
    for bag_candidate_idx in range(total):
        dataset_dir, slide_id = bags_dataset[bag_candidate_idx]
        bag_name = slide_id+'.h5'
        h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
        if not os.path.exists(h5_file_path):
            print(h5_file_path, 'does not exist ...')
            continue
        elif not args.no_auto_skip and slide_id+'.pt' in dest_files:
            print('skipped {}'.format(slide_id))
            continue 
        else:
            exist_idxs.append(bag_candidate_idx)

    for index, bag_candidate_idx in enumerate(exist_idxs):
        dataset_dir, slide_id = bags_dataset[bag_candidate_idx]
        
        print('\nprogress: {}/{}, slide_id: {}'.format(bag_candidate_idx, len(exist_idxs), slide_id))
        patch_img_dir = args.patch_img_dir if args.patch_img_dir else os.path.join(dataset_dir, 'images')
        
        output_feature_path = os.path.join(args.feat_dir, 'pt_files', args.model, slide_id+'.pt')
        images_path = os.path.join(patch_img_dir, slide_id)
        h5_file_path = os.path.join(args.data_h5_dir, 'patches', slide_id+'.h5')
        
        # skip if '.partial' file exists
        if args.skip_partial == 'yes' and os.path.exists(output_feature_path+'.partial'):
            print("Another process is extrating {}".format(output_feature_path))
            continue
        
        one_slide_start = time.time()

        # init dataset
        # skip if h5 not exists,
        if not os.path.exists(h5_file_path):
            print('{} not exists, skip'.format(h5_file_path))
            continue
        patch_dataset = PatchDataset(images_path, h5_file_path, transform=custom_transformer)
        loader = DataLoader(patch_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

        # create an temp file, help other processes
        with open(output_feature_path+'.partial', 'w') as f:
            f.write("")
            
        features = light_compute_w_loader(loader=loader, model=model)

        #save results
        save_feature_subprocess(output_feature_path, features)
        print('coords shape:', features.shape)

        # clear temp file
        os.remove(output_feature_path+'.partial')
        
        print('time per slide: {:.1f}'.format(time.time() - one_slide_start))
    print('Extracting end!')
    print('Time used for this dataset:{:.1f}'.format(time.time() - process_start_time))
