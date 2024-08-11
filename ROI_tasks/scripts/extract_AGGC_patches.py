# from tiffslide import TiffSlide as OpenSlide
import PIL.Image
from openslide import OpenSlide
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
import numpy as np
import time
import PIL
from multiprocessing import Pool
PIL.Image.MAX_IMAGE_PIXELS = 9331200009999


STEP = 512

def extract_patches(save_root, wsi_path, label_paths):
    try:
        wsi_handle = OpenSlide(wsi_path)
    except:
        print('Unsuppoted wsi:', wsi_path)
        return
    
    wsi_name = os.path.split(wsi_path)[-1].split('.')[0]
    
    print('wsi_path:', wsi_path)
    _start = time.time()

    for lp in label_paths:


        label_name = os.path.split(lp)[-1].split('.')[0]
        
        dir = os.path.join(save_root, label_name)
        exists_files = list(set([i.split('__')[0] for i in os.listdir(dir)]))
        if wsi_name in exists_files:
            print('Exists:', lp)
            continue

        label_mask = PIL.Image.open(lp)
        label_mask = np.array(label_mask)
        height, width = label_mask.shape
        
        if not os.path.exists(dir):
            os.makedirs(dir)

        for y in range(0, height, STEP):
            for x in range(0, width, STEP):
                img = label_mask[y: y+STEP, x:x+STEP]
                ratio = np.array(img)[:, :, 0]/255.
                if ratio.mean() > 0.5:
                    patch = wsi_handle.read_region((x, y), 0, (STEP, STEP)).convert('RGB')
                    p = os.path.join(dir, '{}__{}__{}__{}.jpg'.format(wsi_name, label_name, x, y))
                    patch.save(p)
    print('Finished: {:.1f} s'.format(time.time() - _start))


def mp_wrapper(args):
    save_root, wsi_path, labels = args
    extract_patches(save_root=save_root, wsi_path=wsi_path, label_paths=labels)
    
    

if __name__ == '__main__':
    phase = 'train'
    root = '/jhcnas3/Pathology/original_data/AGGC2022/{}'.format(phase)
    save_root = '/home/jmabq/data/AGGC/{}'.format(phase)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
    slide_names = os.listdir(os.path.join(root, 'images'))
    args = []
    for slide_name in slide_names:
        label_root = os.path.join(root, 'annotation', slide_name.split('.')[0])
        labels = [os.path.join(label_root, n) for n in os.listdir(label_root)]
        wsi_path = os.path.join(root, 'images', slide_name)
        args.append([save_root, wsi_path, labels])
        extract_patches(save_root=save_root, wsi_path=wsi_path, label_paths=labels)
    
    # pool = Pool(24)
    # pool.map(mp_wrapper, args)