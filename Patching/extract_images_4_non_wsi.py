import os
import h5py
import numpy as np
from multiprocessing.pool import Pool
import argparse
import cv2
from wsi_core.WholeSlideImage import ImgReader


def is_forground(img):
    t = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = t < 220
    ratio = mask.sum()/mask.shape[0]/mask.shape[1]
    if ratio > 0.5:
        return True
    return False


def process_images(arg):
    save_root, img_root, img_name, prefix, SIZE = arg

    path = os.path.join(img_root, img_name)
    fix, postfix = img_name.split('.')
    save_name = fix + '.jpg'
    print('processing:', path)
    img = cv2.imread(path)
    if img is None:
        return
    # TODO, fix this ugly code later. 
    if parser.resize_flag == 'yes':
        print('GLOBAL RESIZING...')
        img = cv2.resize(img, (SIZE, SIZE))
        p = os.path.join(save_root, '{}{}'.format(prefix, save_name))
        cv2.imwrite(p, img)
        return 

    h, w = img.shape[:2]
    if h < SIZE and w < SIZE:
        img = cv2.resize(img, (SIZE, SIZE))
        p = os.path.join(save_root, '{}{}'.format(prefix, save_name))
        cv2.imwrite(p, img)
    elif h >= SIZE and w >= SIZE:
        xs = list(range(0, w, SIZE))
        xs = [i for i in xs if i+SIZE <= w]
        ys = list(range(0, h, SIZE))
        ys = [i for i in ys if i+SIZE <= h]
        for x in xs:
            for y in ys:
                temp = img[y:y+SIZE, x:x+SIZE].copy()
                p = os.path.join(save_root, '{}{}_{}_{}'.format(prefix, x, y, save_name))
            if is_forground(temp.copy()):
                cv2.imwrite(p, temp)
            else:
                print(p, 'is Backgournd ....')

    elif h < SIZE and w >= SIZE:
        xs = range(0, w, h)
        y = 0
        for x in xs:
            temp = img[:, x:x+h].copy()
            p = os.path.join(save_root, '{}{}_{}_{}'.format(prefix, x, y, save_name))
            if is_forground(temp.copy()):
                cv2.imwrite(p, temp)
            else:
                print(p, 'is Backgournd ....')
    elif h >= SIZE and w < SIZE:
        ys = range(0, h, w)
        x = 0
        for y in ys:
            temp = img[y:y + w, :].copy()
            p = os.path.join(save_root, '{}{}_{}_{}'.format(prefix, x, y, save_name))
            if is_forground(temp.copy()):
                cv2.imwrite(p, temp)
            else:
                print(p, 'is Backgournd ....')

    else:
        raise NotImplementedError


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--cpu_cores', type=int, default=48)
    parser.add_argument('--save_root')
    parser.add_argument('--wsi_root')
    parser.add_argument('--prefix', type=str)
    parser.add_argument('--format', type=str)
    parser.add_argument('--resize_flag', type=str, default='no')
    return parser


if __name__ == '__main__':
    import glob
    parser = argparser().parse_args()

    if not os.path.exists(parser.save_root):
        os.makedirs(parser.save_root)

    files = glob.glob(os.path.join(parser.wsi_root, '**'), recursive=True)
    files = [i for i in files if parser.format in i]
    wsi_roots = [ os.path.split(i)[0] for i in files]
    files = [os.path.split(i)[-1] for i in files]
    
    save_roots = [parser.save_root for _ in files]
    prefix = [parser.prefix for _ in files]
    size = [parser.size for _ in files]

    args = [(sr, ir, i, pre, s) for sr, ir, i, pre, s in zip(save_roots, wsi_roots, files, prefix, size)]

    mp = Pool(parser.cpu_cores)
    mp.map(process_images, args)


