import os


def read_case(csv_path):
    case_ids = []
    with open(csv_path) as f:
        _ = f.readline()
        for line in f:
            case_id = line.split(',')[0]
            case_ids.append(case_id)
    return case_ids


def save_split(save_root, case_ids, fold_id):
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    # bool.csv
    with open(os.path.join(save_root, 'splits_{}_bool.csv'.format(fold_id)), 'w') as f:
        f.write(',train,val,test\n')
        for case_id in case_ids:
            f.write('{},False,False,True\n'.format(case_id))
    # splits.csv
    with open(os.path.join(save_root, 'splits_{}.csv'.format(fold_id)), 'w') as f:
        f.write(',train,val,test\n')
        for index, case_id in enumerate(case_ids):
            f.write('{},{},{},{}\n'.format(index, case_id, case_id, case_id))


folds = 5
datasets = ['LUAD', 'UCEC', 'COAD', 'GBM', 'BRCA']

for dataset in datasets:
    dataset='CPTAC_{}'.format(dataset)
    save_root = '../../splits82/{}_survival_100/'.format(dataset)
    csv_path = '../survival_by_case/{}_Splits.csv'.format(dataset)

    case_ids = read_case(csv_path)

    for fold_id in range(folds):
        save_split(save_root, case_ids, fold_id)