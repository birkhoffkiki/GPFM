from glob import glob
import os

manifest_txt = '/jhcnas3/Pathology/original_data/TCGA/COAD/files.txt'
tcga_root = '/jhcnas3/Pathology/original_data/TCGA/COAD/*/*.svs'

if __name__ == '__main__':
    with open(manifest_txt) as f:
        lines = [l.strip() for l in f]
    all_files = glob(tcga_root)
    all_files = [os.path.split(p)[-1] for p in all_files]
    dd = []
    for p in all_files:
        exist = False
        for l in lines:
            if p in l:
                dd.append(l)
    
    for l in lines:
        if l not in dd:
            print(l)
 