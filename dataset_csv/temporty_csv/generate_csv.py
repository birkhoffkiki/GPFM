dataset_name = 'TPM'


if __name__ == '__main__':
    save_path = '{}.csv'.format(dataset_name)
    # h5_root = '/storage/Pathology/Patches/{}/patches'.format(dataset_name)
    # v = '/storage/Pathology/Patches/{}'.format(dataset_name)
    # h5_root = '/jhcnas3/Pathology/Patches/{}/patches'.format(dataset_name)
    # v = '/jhcnas3/Pathology/Patches/{}'.format(dataset_name)
    h5_root = '/jhcnas5/gzr/data/{}/patches'.format(dataset_name)
    v = '/jhcnas5/gzr/data/{}'.format(dataset_name)
    
    import os
    import random
    random.seed(0)
    
    data_items = []
    slide_ids = os.listdir(h5_root)
    for slide_id in slide_ids:
        slide_id = '.'.join(slide_id.split('.')[:-1])
        data_items.append([slide_id, 'TPM'])
    
    random.shuffle(data_items)
    
    handle = open(save_path, 'w')
    handle.write('dir,case_id,slide_id,label\n')
    for sid, isup in data_items:
        line = '{},{},{},{}\n'.format(v,sid, sid, isup)
        handle.write(line)
    handle.close()