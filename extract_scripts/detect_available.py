import os
import argparse

prefix='/jhcnas3'

meta = {
    'NSCLC': [f'{prefix}/Pathology/Patches/TCGA__LUAD/pt_files', f'{prefix}/Pathology/Patches/TCGA__LUSC/pt_files'],
    'CAMELYON': [f'{prefix}/Pathology/Patches/CAMELYON16/pt_files', f'{prefix}/Pathology/Patches/CAMELYON17/pt_files'],
    'RCC': [f'{prefix}/Pathology/Patches/TCGA__KIRP/pt_files', f'{prefix}/Pathology/Patches/TCGA__KIRC/pt_files', f'{prefix}/Pathology/Patches/TCGA__KICH/pt_files'],
    'PANDA': [f'{prefix}/Pathology/Patches/PANDA/pt_files'],
    'BRCA': [f'{prefix}/Pathology/Patches/TCGA__BRCA/pt_files'],
    'BRACS': [f'{prefix}/Pathology/Patches/BRACS/pt_files'],
    'TUPAC': [f'{prefix}/Pathology/Patches/TUPAC16/pt_files'],
    'UBC-OCEAN': [f'{prefix}/Pathology/Patches/UBC-OCEAN/pt_files'],
    'STAD': [f'{prefix}/Pathology/Patches/TCGA__STAD/pt_files'],
    'COADREAD': [f'{prefix}/Pathology/Patches/TCGA__COADREAD/pt_files'],
    'UCEC': [f'{prefix}/Pathology/Patches/TCGA__UCEC/pt_files'],
    'LGG': [f'{prefix}/Pathology/Patches/TCGA__LGG/pt_files'],
    'GBM': [f'{prefix}/Pathology/Patches/TCGA__GBM/pt_files'],
    'BLCA': [f'{prefix}/Pathology/Patches/TCGA__BLCA/pt_files'],
    'RCC-DHMC': [f'{prefix}/Pathology/Patches/RCC-DHMC/pt_files'],
    'CPTAC-GBM': [f'{prefix}/Pathology/Patches/CPTAC__GBM/pt_files'],
    'CPTAC-COAD': [f'{prefix}/Pathology/Patches/CPTAC__COAD/pt_files'],
    'CPTAC-UCEC': [f'{prefix}/Pathology/Patches/CPTAC__UCEC/pt_files'],
    'CPTAC-BRCA': [f'{prefix}/Pathology/Patches/CPTAC__BRCA/pt_files'],
    
}

def print_feature_info(data_lists):
    for root in data_lists:
        print('*', root)
        models = os.listdir(root)
        for m in models:
            p = os.path.join(root, m)
            if not os.path.isdir(p):
                continue
            slides = [i for i in os.listdir(p) if '.partial' not in i]
            print('\t{}:{}'.format(m, len(slides)))
            
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, default='')
    
    args = parser.parse_args()
    
    print_feature_info(meta[args.name])
    
    