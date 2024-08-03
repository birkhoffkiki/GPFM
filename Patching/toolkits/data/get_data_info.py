import pandas as pd
import os


def count_sub_data(path):
    slide_dirs = os.listdir(path)
    if len(slide_dirs) !=0 and os.path.isfile(os.path.join(path, slide_dirs[0])):
        return len(slide_dirs)
    num = 0
    for name in slide_dirs:
        p = os.path.join(path, name)
        num += len(os.listdir(p))
    return num


if __name__ == '__main__':
    excel_path = './data.xlsx'

    roots = ['/storage/Pathology/Patches']
    frame = {'dataset': [], 'number': [], 'slide#': [], 'Path': []}
    total_number = 0
    # if os.path.exists(excel_path):
    #     df = pd.read_excel(excel_path, sheet_name='Pathology Data')
    #     frame['dataset']=df['dataset'].to_list()
    #     frame['number']=df['number'].to_list()
    #     frame['slide#']=df['slide#'].to_list()
    #     frame['Path']=df['Path'].to_list()

    for root in roots:
        subtypes = os.listdir(root)
        for index, sub in enumerate(subtypes):
            print('processing:', sub)
            # if sub in frame['dataset']:
            #     continue
            path = os.path.join(root, sub, 'images')
            if not os.path.isdir(path) or len(os.listdir(path)) == 0:
                continue

            slide_num = len(os.listdir(path)) if os.path.isdir(os.path.join(path, os.listdir(path)[0])) else 0
            
            frame['slide#'].append(slide_num)
            print('{}: [{}/{}]'.format(root, index+1, len(subtypes)))
            if not os.path.exists(path):
                continue
            num = count_sub_data(path)
            frame['dataset'].append(sub)
            frame['number'].append(num)
            frame['Path'].append(path)
            total_number += num
    print('Total number:', total_number)
    writer = pd.ExcelWriter(excel_path)
    df = pd.DataFrame(frame)
    df.to_excel(writer, sheet_name='Pathology Data')
    writer.close()

