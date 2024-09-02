import json
import os


with open('/storage/Pathology/codes/EasyMIL/dataset_csv/temporty_csv/clinical_BRCA.json') as f:
    clinical = json.load(f)
    
with open('/storage/Pathology/codes/EasyMIL/dataset_csv/temporty_csv/metadata_BRCA.json') as f:
    meta = json.load(f)
    
items = []

root = '/storage/Pathology/Patches/TCGA__BRCA'

def find_slide_id(case_id):
    for m in meta:
        mid = m['associated_entities'][0]['case_id']
        if mid == case_id:
            return m['file_name'][:-4]
    raise FileNotFoundError


for c in clinical:
    case_id = c['case_id']

    diag = c['diagnoses']
    
    for d in diag: 
        primary_diagnosis = d['primary_diagnosis'].lower()
        s_id = find_slide_id(case_id)
        
        if 'duct' in primary_diagnosis and 'lobular' in primary_diagnosis:
            x = None
            print('ID: {}, IDC and ILC'.format(primary_diagnosis))
        elif 'duct' in primary_diagnosis:
            x = '{},{},{},{}\n'.format(root, case_id, s_id, 'IDC')
        elif 'lobu' in primary_diagnosis:
            x = '{},{},{},{}\n'.format(root, case_id, s_id, 'ILC')
        else:
            x = None
            print('Failed:', case_id, primary_diagnosis)
        if x:
            if os.path.exists(os.path.join('/storage/Pathology/Patches/TCGA__BRCA/pt_files/dinov2_vitl', s_id+'.pt')):
                items.append(x)
            else:
                print('Pt file not exist')

with open('/storage/Pathology/codes/EasyMIL/dataset_csv/BRCA_subtyping.csv', 'w') as f:
    f.write('dir,case_id,slide_id,label\n')
    for i in items:
        f.write(i)