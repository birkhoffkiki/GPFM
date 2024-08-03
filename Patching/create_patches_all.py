# internal imports
from wsi_core.batch_process_utils import initialize_df
# other imports
import os
import numpy as np
import argparse
import pandas as pd
from PIL import Image
import h5py


def seg_and_patch(source, save_dir, patch_save_dir,  
                  patch_size = 256, 
                  seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
                  'keep_ids': 'none', 'exclude_ids': 'none'},
                  filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}, 
                  vis_params = {'vis_level': -1, 'line_thickness': 500},
                  patch_params = {'use_padding': True, 'contour_fn': 'four_pt'},
                  auto_skip=True, process_list = None,
                  redudant=0,
                  wsi_format='svs'
                  ):

    slides = []
    for root, dirs, filenames in os.walk(source):
        for filename in filenames:
            postfix = filename.split('.')[-1].lower()
            if postfix == wsi_format:
                slides.append(os.path.join(root, filename))

    if process_list is None:
        df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)
    else:
        df = pd.read_csv(process_list)
        df = initialize_df(df, seg_params, filter_params, vis_params, patch_params)

    mask = df['process'] == 1
    process_stack = df[mask]

    total = len(process_stack)

    print('Total:', total)
    for i in range(total):
        df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
        idx = process_stack.index[i]
        slide = process_stack.loc[idx, 'slide_id']
        print("\n\nprogress: {:.2f}, {}/{}".format(i/total, i, total), flush=True)
        print('processing {}'.format(slide))
        
        df.loc[idx, 'process'] = 0
        slide_id, _ = os.path.splitext(slide)
        slide_id = slide_id.split('/')[-1]

        if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
            print('{} already exist in destination location, skipped'.format(slide_id))
            df.loc[idx, 'status'] = 'already_exist'
            continue

        # Inialize WSI
        full_path = os.path.join(source, slide)
        # use PIL to read imgaes
        wsi_obj = Image.open(full_path)
        width, height = wsi_obj.width, wsi_obj.height
        patch_coords = []
        for x in range(0, width, patch_size-redudant):
            for y in range(0, height, patch_size-redudant):
                patch_coords.append((x, y))
        patch_coords = np.array(patch_coords)
        file_path = os.path.join(patch_save_dir, slide_id+'.h5')
        h5 = h5py.File(file_path, 'w')
        h5.create_dataset('coords', data=patch_coords)
        h5.close()


parser = argparse.ArgumentParser(description='seg and patch')
parser.add_argument('--source', type = str,
                    help='path to folder containing raw wsi image files')
parser.add_argument('--patch_size', type = int, default=256,
                    help='patch_size')
parser.add_argument('--redudant', default=0, type=int)
parser.add_argument('--no_auto_skip', default=True, action='store_false')
parser.add_argument('--save_dir', type = str,
                    help='directory to save processed data')
parser.add_argument('--process_list',  type = str, default=None,
                    help='name of list of images to process with parameters (.csv)')
parser.add_argument('--wsi_format', type = str, default='svs')


if __name__ == '__main__':
    args = parser.parse_args()

    wsi_format = args.wsi_format
    patch_save_dir = os.path.join(args.save_dir, 'patches')

    if args.process_list:
        process_list = os.path.join(args.save_dir, args.process_list)

    else:
        process_list = None

    print('source: ', args.source)
    print('patch_save_dir: ', patch_save_dir)
    
    directories = {'source': args.source, 
                   'save_dir': args.save_dir,
                   'patch_save_dir': patch_save_dir, }

    for key, val in directories.items():
        print("{} : {}".format(key, val))
        if key not in ['source']:
            os.makedirs(val, exist_ok=True)

    seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
                  'keep_ids': 'none', 'exclude_ids': 'none'}
    filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':32}
    vis_params = {'vis_level': -1, 'line_thickness': 120}
    patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}
    
    parameters = {'seg_params': seg_params,
                  'filter_params': filter_params,
                   'patch_params': patch_params,
                  'vis_params': vis_params}

    print(parameters)

    seg_times, patch_times = seg_and_patch(**directories, **parameters,
                                            patch_size = args.patch_size,
                                            redudant=args.redudant, 
                                            auto_skip=args.no_auto_skip,
                                            wsi_format=wsi_format
                                            )
