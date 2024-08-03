import json


with open('/storage/Pathology/codes/Patching/toolkits/python_scripts/frozen_meta.json') as f:
    data = json.load(f)

result = {}
for case in data:
    file_name = case['file_name'][:-4]
    label = case['cases'][0]['project']['project_id']
    if 'TCGA' in label:
        label = label.split('-')[-1]
    result[file_name] = label

with open('frozen_dict.json', 'w') as f:
    json.dump(result, f)