import os
import json
import cv2
from PIL import Image
from multiprocessing.pool import Pool
import random


def process(arg):
    root, name = arg
    p = os.path.join(root, name)
    return name


def get_all_files(roots, target=None, excludes=None):
    random.seed(0)
    
    all_root = []
    for root in roots:
        subtypes = os.listdir(root)
        for sub in subtypes:
            if target is not None:
                if target in sub:
                    print('processs:', sub)
                else:
                    continue
            if excludes is not None:
                if excludes in sub:
                    continue
                    
            r = os.path.join(root, sub, 'pt_files/dinov2_vitl')
            if not os.path.exists(r):
                print(sub, ": is blank ...")
                continue
            slides = os.listdir(r)
            if len(slides) == 0:
                print(subtypes, ": is blank ...")
                continue
            # having sub slides
            slides = [os.path.join(r, s) for s in slides if '.pt' in s]
            all_root.extend(slides)
    print('Total slides:', len(all_root))
    
    return all_root


if __name__ == '__main__':
    # roots = ['/home/jmabq/DATA/Pathology']
    target = None # None for all
    excludes = None # None for nothing
    roots = ['/storage/Pathology/Patches']
    # paths = get_all_files_v1(roots)
    paths = get_all_files(roots, target=target, excludes=excludes)
    
    save_path = '/storage/Pathology/pathology_wsi_feature_slides_{}.json'.format(len(paths))

    with open(save_path, 'w') as f:
        json.dump(paths, f)