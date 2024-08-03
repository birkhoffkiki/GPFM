import os
import h5py
import argparse


def read_images(h5_path):
    h5 = h5py.File(h5_path)
    coors = h5['coords']
    num = coors.shape[0]
    return num

def count_one_subtype(root, subtype):
    total = 0
    h5_root = os.path.join(root, subtype, 'patches')
    if not os.path.exists(h5_root):
        d = os.path.join(root, sub, 'images')
        if not os.path.exists(d):
            return 0
        return len(os.listdir(d))
    
    h5_files = os.listdir(h5_root)
    h5_paths = [os.path.join(h5_root, p) for p in h5_files]
    for p in h5_paths:
        try:
            num = read_images(p)
        except:
            print('Failed to count:', p)
            num = 0
        total += num
    return total


if __name__ == '__main__':
    # root = '/home/jmabq/DATA/Pathology'
    root = '/storage/Pathology/Patches'
    subtypes = os.listdir(root)
    total_number = 0
    for sub in subtypes:
        num = count_one_subtype(root, sub)
        print('{}: {}'.format(sub, num))
        total_number += num
    print('Total number:', total_number)




