import torch
import os
import time
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
from models import get_custom_transformer, get_model


import argparse
from utils.clam_utils import collate_features
from utils.file_utils import save_hdf5
from utils.hit_cache import RamDiskCache
import openslide
import numpy as np
from multiprocessing import Process
import glob
from wsi_core.WholeSlideImage import ImgReader


def get_wsi_handle(wsi_path):
    if not os.path.exists(wsi_path):
        raise FileNotFoundError(f'{wsi_path} is not found')
    postfix = wsi_path.split('.')[-1]
    if postfix.lower() in ['svs', 'tif', 'ndpi', 'tiff']:
        handle = openslide.OpenSlide(wsi_path)
    elif postfix.lower() in ['jpg', 'jpeg', 'tiff', 'png']:
        handle = ImgReader(wsi_path)
    else:
        raise NotImplementedError(f'{postfix} is not implemented...')
    return handle



def save_feature(path, feature):
    s = time.time()
    torch.save(feature, path)
    e = time.time()
    print('Feature is sucessfully saved at: {}, cost: {:.1f} s'.format(path, e-s))


def save_hdf5_subprocess(output_path, asset_dict):
    kwargs = {'output_path': output_path, 'asset_dict': asset_dict, 
                   'attr_dict': None, 'mode': 'w'}
    process = Process(target=save_hdf5, kwargs=kwargs)
    process.start()


def save_feature_subprocess(path, feature):
    kwargs = {'feature': feature, 'path': path}
    process = Process(target=save_feature, kwargs=kwargs)
    process.start()


def light_compute_w_loader(file_path, wsi, model,
     batch_size = 8, verbose = 0, print_every=20, pretrained=True, 
    custom_downsample=1, target_patch_size=-1, custom_transformer=None):
    """
    Do not save features to h5 file to save storage
    args:
        file_path: directory of bag (.h5 file)
        output_path: directory to save computed features (.h5 file)
        model: pytorch model
        batch_size: batch_size for computing features in batches
        verbose: level of feedback
        pretrained: use weights pretrained on imagenet
        custom_downsample: custom defined downscale factor of image patches
        target_patch_size: custom defined, rescaled image size before embedding
    """
    dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, custom_transforms=custom_transformer,
        custom_downsample=custom_downsample, target_patch_size=target_patch_size, fast_read=True)
    kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}
    print('Data Loader args:', kwargs)
    loader = DataLoader(dataset=dataset, batch_size=batch_size,  **kwargs, collate_fn=collate_features, prefetch_factor=16)

    if verbose > 0:
        print('processing {}: total of {} batches'.format(file_path,len(loader)))

    features_list = []
    coords_list = []
    _start_time = time.time()
    cal_time = time.time()
    for count, (batch, coords) in enumerate(loader):
        read_time_flag = time.time()
        img_read_time = abs(read_time_flag - cal_time)
        # print('Reading images time:', img_read_time)
        with torch.no_grad():	
            if count % print_every == 0:
                batch_time = time.time()
                print('batch {}/{}, {} files processed, used_time: {} s'.format(
                    count, len(loader), count * batch_size, batch_time - _start_time))
            batch = batch.to(device, non_blocking=True)
            features = model(batch)
            features = features.cpu()
            features_list.append(features)
            coords_list.append(coords)
            cal_time = time.time()
        # print('Calculation time: {} s'.format(cal_time-read_time_flag))
        
    features = torch.cat(features_list, dim=0)
    coords = np.concatenate(coords_list, axis=0)
    return features, coords


def find_all_wsi_paths(wsi_root, ext):
    """
    find the full wsi path under data_root, return a dict {slide_id: full_path}
    """
    ext = ext[1:]
    result = {}
    all_paths = glob.glob(os.path.join(wsi_root, '**'), recursive=True)
    all_paths = [i for i in all_paths if i.split('.')[-1].lower() == ext.lower()]
    for h in all_paths:
        slide_name = os.path.split(h)[1]
        slide_id = '.'.join(slide_name.split('.')[0:-1])
        result[slide_id] = h
    print("found {} wsi".format(len(result)))
    return result


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
parser.add_argument('--model', type=str)
parser.add_argument('--datatype', type=str)
parser.add_argument('--save_storage', type=str, default='no')

# speed up
parser.add_argument('--ramdisk_cache', default='', type=str)
parser.add_argument('--use_cache', default='yes', type=str)
parser.add_argument('--ramdisk_capacity', default=100, type=int)
parser.add_argument('--ramdisk_warning_size', default=5, type=int)

# Histlogy-pretrained MAE setting
# parser.add_argument('--mae_checkpoint', type=str, default=None, help='path to pretrained mae checkpoint')

args = parser.parse_args()



if __name__ == '__main__':
    process_start_time = time.time()
    print('initializing dataset')
    if args.ramdisk_cache:
        os.makedirs(args.ramdisk_cache)
        ramdisk_engine = RamDiskCache(args.ramdisk_cache, capacity=args.ramdisk_capacity, warning_mem=args.ramdisk_warning_size)

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
    
    # obtain slide_id
    get_slide_id = lambda idx: str(bags_dataset[idx]).split(args.slide_ext)[0]
    # check the exists wsi
    exist_idxs = []
    all_wsi_paths = find_all_wsi_paths(args.data_slide_dir, args.slide_ext)
    for bag_candidate_idx in range(total):
        slide_id = get_slide_id(bag_candidate_idx)
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

    
    def find_next_file_paths(index, next_num=4):
        # cache next num files
        indexs = [index + i for i in range(next_num)]
        next_ids = [exist_idxs[index % len(exist_idxs)] for index in indexs]
        next_slide_ids = [get_slide_id(i) for i in next_ids]
        file_paths = [all_wsi_paths[nd] for nd in next_slide_ids]
        return file_paths
    

    for index, bag_candidate_idx in enumerate(exist_idxs):
        slide_id = get_slide_id(bag_candidate_idx)
        bag_name = slide_id+'.h5'
        h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
        if not os.path.exists(h5_file_path):
            print(h5_file_path, 'does not exist ...')
            continue

        # TCGA
        slide_file_path = all_wsi_paths[slide_id]
        print('\nprogress: {}/{}, slide_id: {}'.format(bag_candidate_idx, len(exist_idxs), slide_id))

        if not args.no_auto_skip and slide_id+'.pt' in dest_files:
            print('skipped {}'.format(slide_id))
            continue 

        output_h5_path = os.path.join(args.feat_dir, 'h5_files', args.model, bag_name)
        bag_base, _ = os.path.splitext(bag_name)
        output_feature_path = os.path.join(args.feat_dir, 'pt_files', args.model, bag_base+'.pt')
        
        # skip if '.partial' file exists
        if os.path.exists(output_feature_path+'.partial'):
            print("Another process is extrating {}".format(output_feature_path))
            continue
        
        one_slide_start = time.time()
        if args.use_cache == 'yes':
            # ----------- ramdisk cache, speed up next iteration-------------
            next_file_paths = find_next_file_paths(index=index, next_num=2)
            ramdisk_engine.add_cache(next_file_paths)
            # ----------------------------------------------------------
            time_start = time.time()
            # wait until cache success
            _slide_file_path = ramdisk_engine.new_path(slide_file_path)
            wait_start_time = time.time()
            total_wait_time = 3600
            while _slide_file_path == slide_file_path:
                _slide_file_path = ramdisk_engine.new_path(slide_file_path)
                time.sleep(10)
                print('Please wait, waiting for caching: {}, time:{:.1f} s'.format(_slide_file_path, time.time() - wait_start_time), flush=True)
                if ramdisk_engine.re_copy(slide_file_path):
                    # force to re copy
                    print('Caching process may die, trying to recaching ...')
                    ramdisk_engine.__copy_file(slide_file_path)
                if time.time() - wait_start_time > total_wait_time:
                    # raise RuntimeError('Failed to load data within {} s'.format(total_wait_time))
                    _slide_file_path = slide_file_path
                    print('Failed to use cache, reading data from NAS...')
                    break
        else:
            print('Reading data directly from original source ...')
            _slide_file_path = slide_file_path
   
        print('Using:', _slide_file_path)
  
        # wsi = openslide.open_slide(_slide_file_path)
        wsi = get_wsi_handle(_slide_file_path)

        # create an temp file, help other processes
        with open(output_feature_path+'.partial', 'w') as f:
            f.write("")
            
        if args.save_storage == 'yes':
            features, coords = light_compute_w_loader(h5_file_path, wsi, 
                        model = model, batch_size = args.batch_size, verbose = 1, print_every = 20, 
                        custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size,
                        custom_transformer=custom_transformer)
        else:
            raise ValueError('We only support save storage mode...')
        #save results
        save_feature_subprocess(output_feature_path, features)
        print('coords shape:', coords.shape)
        asset_dict = {'coords': coords}
        save_hdf5_subprocess(output_h5_path, asset_dict=asset_dict)		
        if args.use_cache=='yes':
            ramdisk_engine.remove_cache(slide_file_path)
        # clear temp file
        os.remove(output_feature_path+'.partial')
        
        print('time per slide: {:.1f}'.format(time.time() - one_slide_start))
    print('Extracting end!')
    print('Time used for this dataset:{:.1f}'.format(time.time() - process_start_time))
