import os
import json
import cv2
from PIL import Image
from multiprocessing.pool import Pool
import random


def get_files(arg):

    _r, subt = arg
    paths = {}
    if not os.path.exists(_r):
        print('Skip:', _r)
        return paths
    paths[subt] = {}
    _r = os.path.join(_r, subt, 'images')
    print(_r)
    slides = os.listdir(_r)

    s0 =  os.path.join(_r, slides[0])
    if os.path.isdir(s0):
        paths[subt]['over']=False
        paths[subt][_r] = {}
        for s in slides:
            _s = os.path.join(_r, s)
            names = os.listdir(_s)
            paths[subt][_r][_s] = []

            for n in names:
                p = os.path.join(_s, n)
                # read test
                img = cv2.imread(p)
                if img is not None:
                    paths[subt][_r][_s].append(n)
                else:
                    print(p)
    else:
        paths[subt]['over'] = True
        paths[subt][_r] =[]
        for n in slides:
            p = os.path.join(_r, n)
            # read test
            img = cv2.imread(p)
            if img is not None:
                paths[subt][_r].append(n)
            else:
                print(p)

    
def get_all_files_v1(roots):
    args = []
    for r in roots:
        subtypes = os.listdir(r)

        args.extend([(r, sub) for sub in subtypes])

    paths = {}
    pool = Pool(48)
    results = pool.map(get_files, args)
    for r in results:
        for k, v in r.items():
            paths[k]  = v
    return paths


def process(arg):
    root, name = arg
    p = os.path.join(root, name)
    return name
    # try:
    #     img = Image.open(p)
    #     return name
    # except:
    #     return None


def get_all_files(roots, slide_num=None, target=None, excludes=None):
    random.seed(0)
    
    bad_samples = []
    paths = {}
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
                    
            r = os.path.join(root, sub, 'images')
            if not os.path.exists(r):
                print(sub, ": is blank ...")
                continue
            slides = os.listdir(r)
            if len(slides) == 0:
                print(subtypes, ": is blank ...")
                continue
            # having sub slides
            test_p = os.path.join(r, slides[0])
            if os.path.isdir(test_p):
                all_root.extend([os.path.join(r, s) for s in slides])
            else:
                all_root.append(r)
    print('Total slides:', len(all_root))
    
    pool = Pool(48)
    if slide_num is None:
        selected_roots = all_root
    else:
        random.shuffle(all_root)
        selected_roots = all_root[:slide_num]
    
    for index, prefix in enumerate(selected_roots):
        print('procsssing: {}/{}, {}'.format(index, len(selected_roots), prefix))
        names = os.listdir(prefix)
        args = [(prefix, n) for n in names]
        results = pool.map(process, args)

        results = [r for r in results if r is not None]
        bad_samples.extend([r for r in results if r is None])
        paths[prefix] = results
    
    total_num = 0
    for k, v in paths.items():
        total_num += len(v)
    print('total num:', total_num)
    
    return paths, bad_samples


if __name__ == '__main__':
    # roots = ['/home/jmabq/DATA/Pathology']
    slide_num = None
    num_per_slide = 2000
    target = None # None for all
    excludes = None # None for nothing
    roots = ['/storage/Pathology/Patches']
    
    # paths = get_all_files_v1(roots)
    paths, bad_samples = get_all_files(roots, slide_num=slide_num, target=target, excludes=excludes)
    new_paths = {}
    
    total = 0
    for k, v in paths.items():
        random.shuffle(v)
        new_paths[k] = v[:num_per_slide]
        total += len(v[:num_per_slide])
        
    paths = new_paths
    
    save_path = '/storage/Pathology/pathology_patches_{}_slides_{}.json'.format(total, len(paths))

    with open(save_path, 'w') as f:
        json.dump(paths, f)
    print(bad_samples)
