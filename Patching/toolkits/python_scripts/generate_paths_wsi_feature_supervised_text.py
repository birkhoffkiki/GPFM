import os
import json
from PIL import Image
from multiprocessing.pool import Pool
import random


def process(arg):
    root, name = arg
    p = os.path.join(root, name)
    return name


with open('/storage/Pathology/codes/Patching/toolkits/python_scripts/frozen_dict.json') as f:
    frozen_dict = json.load(f)

    
def get_slide_label(subtype, slide_id):
    # for TCGA and CPTAC
    if 'TCGA-frozen' in subtype:
        # if slide_id in frozen_dict.keys():
        #     return frozen_dict[slide_id]
        # print('Not found:', slide_id)
        return None
    
    elif 'TCGA' in subtype or 'CPTAC' in subtype:
        _, label = subtype.split('__')
        return label
    
    # for AGGC2022
    elif subtype == 'AGGC2022':
        root = '/jhcnas3/Pathology/original_data/AGGC2022'
        candicates = ['G5_Mask.tif', 'G4_Mask.tif', 'G3_Mask.tif', 'Normal_Mask.tif']
        labels = ['AGGC2022_G5', 'AGGC2022_G4', 'AGGC2022_G3', 'AGGC2022_Normal']
        prefix = slide_id.split('.')[0]
        if 'Train' in slide_id:
            dir = os.path.join(root, 'train', 'annotation', prefix)
        else:
            dir = os.path.join(root, 'test', 'annotation', prefix)

        target_masks = os.listdir(dir)
        for index, c in enumerate(candicates):
            if c in target_masks:
                return labels[index]
    
    elif subtype == 'PAIP2019':
        return 'liver'
    elif subtype == 'PAIP2020':
        return 'CRC'
    elif subtype == 'PAIP2021':
        if 'Col' in slide_id:
            return 'colon'
        if 'Pros' in slide_id:
            return 'prostate'
        if 'Pan' in slide_id:
            return 'pancreatobiliary tract'
    return None    


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
                if sub in excludes:
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
            slide_ids = [os.path.split(i)[-1][:-3] for i in slides]

            slides_result = []
            for index, slide_id in enumerate(slide_ids):
                label = get_slide_label(subtype=sub, slide_id=slide_id)
                if label is not None:
                    slides_result.append((slides[index], label))
            all_root.extend(slides_result)
    
    keys = list(set([i[1] for i in all_root]))
    keys = {k:i for i, k in enumerate(keys)}
    final = {
        'keys': keys,
        'data_items': all_root,
    }
    print('Total slides:', len(all_root))
    
    return final


if __name__ == '__main__':
    # roots = ['/home/jmabq/DATA/Pathology']
    target = None # None for all
    excludes = ['CAMELYON16', 'CAMELYON17', 'TCGA__LUAD', 'TCGA__LUSC', 'PANDA', 'BRACS'] # None for nothing
    
    roots = ['/storage/Pathology/Patches']
    # paths = get_all_files_v1(roots)
    paths = get_all_files(roots, target=target, excludes=excludes)
    
    save_path = '/storage/Pathology/pathology_wsi_feature_slides_supervised_{}.json'.format(len(paths['data_items']))

    with open(save_path, 'w') as f:
        json.dump(paths, f)