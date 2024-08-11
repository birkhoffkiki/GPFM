import os
import json
import numpy as np

root = '/home/jmabq/data/results'


models = ['resnet50', 'phikon', 'ctranspath', 'uni', 'conch', 'plip', 'distill_87499', 'distill_99999', 'distill_174999', 'distill_12499_cls_only',
          'distill_137499_cls_only', 'distill_12499', 'dinov2_vitl', 'distill_379999_cls_only',
          'distill_487499_cls_only']

Keys = {'resnet50': 'ResNet50', 'ctranspath': 'Ctranspath', 'uni': 'UNI', 
        'conch': 'CONCH', 'plip': 'PLIP', 'distill_87499': 'DisFM-87499',
        'phikon': 'Phikon', 'distill_99999': 'DisFM-99999', 'distill_174999': 'DisFM-174999', 'distill_12499_cls_only':'DisFM-CLS-12499',
        'distill_137499_cls_only': 'DisFM-CLS-137499', 'distill_12499': 'DisFM-12499', 'dinov2_vitl': 'dinov2_vitl',
        'distill_379999_cls_only': 'DisFM-CLS-379999', 'distill_487499_cls_only': 'DisFM-CLS-487499'}


def linear_metric(json_path):
    with open(json_path) as f:
        lines = [i for i in f]
    data = lines[-2][11:]
    line = json.loads(data)['linear_prob']
    return (line['top1 acc_mean'], line['f1_score_mean'], line['AUC_mean'])


if __name__ == '__main__':
    datasets = os.listdir(root)
    acc_dict = {m: [] for m in models}
    auc_dict = {m: [] for m in models}
    f1_dict = {m: [] for m in models}
    
    
    for model in models:
        for dataset in datasets:
            print(model, dataset)
            json_path = os.path.join(root, dataset, 'linear',  model, 'results_eval_linear.json')
            acc, f1, auc = linear_metric(json_path)
            acc_dict[model].append(acc)
            f1_dict[model].append(f1)
            auc_dict[model].append(auc)
    
    for m in models:
        avg_acc = sum(acc_dict[m])/len(acc_dict[m])
        avg_f1 = sum(f1_dict[m])/len(f1_dict[m])
        avg_auc = sum(auc_dict[m])/len(auc_dict[m])
        print("{}&{:.3f}±{:.3f}&{:.3f}±{:.3f}&{:.3f}±{:.3f}\\\\".format(Keys[m], avg_acc, np.std(acc_dict[m]),
                    avg_f1, np.std(f1_dict[m]), avg_auc, np.std(auc_dict[m])))
        

