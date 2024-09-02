"""

"""
from torchmetrics import AUROC
from torchmetrics.classification.f_beta import F1Score
from torchmetrics.classification import MulticlassAccuracy
import numpy as np
from scipy import stats
import pandas as pd
import os
import pickle
try:
    from sksurv.metrics import concordance_index_censored
except:
    print('failed to load sksurv lib...')
import torch


def auc_fn(num_classes):
    def fn():
        m = AUROC(num_classes=num_classes, average='macro', task='multiclass')
        return m
    return fn

def acc_fn(num_classes, topk=1):
    def fn():
        m = MulticlassAccuracy(top_k=topk, num_classes=int(num_classes), average='macro'),
        return m
    return fn

def f1_fn(num_classes):
    def fn():
        m =  F1Score(num_classes=int(num_classes), average='macro', task='multiclass'),
        return m
    return fn

        

def C_Index():
    def fn(risks, label):
        event_times, censorships = label
        c_index = concordance_index_censored((1-censorships).astype(bool), event_times, risks, tied_tol=1e-08)[0]
        return c_index
    return fn
    
    
def p_value_bootstrap(metric_fn, model_1_predictions, model_2_predictions, test_set_labels, n_resamples=1000, mode='less'):
    """
    """
    
    model_1_metric = metric_fn()(model_1_predictions, test_set_labels)
    model_2_metric = metric_fn()(model_2_predictions, test_set_labels)
    
    print('Model 1 metric:', model_1_metric)
    print('Model 2 metric:', model_2_metric)
    

    n_samples = len(model_1_predictions)
    print('n_samples:', n_samples)
    all_metric1 = []
    all_metric2 = []
    for i in range(n_resamples):
        bootstrap_indices = np.random.choice(range(n_samples), size=n_samples, replace=True)
        if isinstance(test_set_labels, (list, tuple)):
            new_test_set_labels = [v[bootstrap_indices] for v in test_set_labels]
        else:
            new_test_set_labels = test_set_labels[bootstrap_indices]
            
        new_model_1_predictions = model_1_predictions[bootstrap_indices]
        new_model_2_predictions = model_2_predictions[bootstrap_indices]

        model_1_metric = metric_fn()(new_model_1_predictions, new_test_set_labels)
        model_2_metric = metric_fn()(new_model_2_predictions, new_test_set_labels)
        # add metric
        all_metric1.append(model_1_metric)
        all_metric2.append(model_2_metric)

    t, p_value = stats.wilcoxon(all_metric1, all_metric2, correction=False, alternative=mode)

    return p_value, all_metric1, all_metric2


def parse_survival_results(root):
    IDs = []
    censorships = []
    event_times = []
    risk_scores = []
    for fold in range(5):
        p = os.path.join(root, 'split_{}_results.pkl'.format(fold))
        with open(p, 'rb') as f:
            data = pickle.load(f)
        for slide_id, values in data.items():
            risk = values['risk']
            event = values['survival']
            censor = values['censorship']
            
            censorships.append((censor))
            event_times.append(event)
            risk_scores.append(risk)
            IDs.append(slide_id)
    
                
    censorships = np.array(censorships)
    event_times = np.array(event_times)
    risk_scores = np.array(risk_scores)
    return censorships, event_times, risk_scores, IDs
    

def survival_p_value(result1, result2):
    censorships1, event_times1, risk_socres1, ID1 = parse_survival_results(result1)
    censorships2, event_times2, risk_socres2, ID2 = parse_survival_results(result2)
    for i1, i2 in zip(ID1, ID2):
        if i1 != i2:
            print(i1, i2)
            raise RuntimeError
    # print(censorships2)
    print('chekc mean value:', (censorships1 - censorships2).mean())
    pvalue, m1, m2 = p_value_bootstrap(C_Index, risk_socres1, risk_socres2, [event_times1, censorships1], n_resamples=1000)
    print('P values is:', pvalue)


def WSI_p_value(result1, result2):
    """
    For Fengtao pkl
    """
    with open(result1, 'rb') as f:
        data1 = pickle.load(f)
    
    with open(result2, 'rb') as f:
        data2 = pickle.load(f)
    
    n_classes = data1['labels'].shape[1]
    y_test = data1['labels'].argmax(axis=1)
    y_score1 = data1['logits']
    ids = data1['IDs']
    y_score2 = []
    for case_id in ids:
        index = data2['IDs'].index(case_id)
        # print('model1:', y_test[index])
        # print('model2:', data2['labels'])
        score = data2['logits'][index:index+1, :]
        y_score2.append(score)
        
    y_score2 = np.concatenate(y_score2, axis=0)
    
    y_score1 = torch.from_numpy(y_score1)
    y_score2 = torch.from_numpy(y_score2)
    y_test = torch.from_numpy(y_test)
    
    
    pvalue, m1, m2 = p_value_bootstrap(auc_fn(n_classes), y_score1, y_score2, y_test, n_resamples=1000)
    print('P values is:', pvalue)
   
    
def WSI_p_value_easymil(result1, result2):
    """
    For EasyMIL Eval
    """
    
    data1 = {'labels': [], 'logits': [], 'IDs': []}
    with open(result1) as f:
        all_lines = f.readlines()
        for line in all_lines[1:]:
            items = line.split(',')
            slide_id = items[0]
            gt = int(float(items[1]))
            logits = np.array([float(i) for i in items[3:]])
            data1['IDs'].append(slide_id)
            data1['labels'].append(gt)
            data1['logits'].append(logits)
        data1['labels'] = np.array(data1['labels'], dtype='int')
        data1['logits'] = np.stack(data1['logits'])
        
    data2 = {'labels': [], 'logits': [], 'IDs': []}
    with open(result2) as f:
        all_lines = f.readlines()
        for line in all_lines[1:]:
            items = line.split(',')
            slide_id = items[0]
            gt = int(float(items[1]))
            logits = np.array([float(i) for i in items[3:]])
            data2['IDs'].append(slide_id)
            data2['labels'].append(gt)
            data2['logits'].append(logits)
        data2['labels'] = np.array(data2['labels'], dtype='int')
        data2['logits'] = np.stack(data2['logits'])
    
    
    n_classes = int(data1['labels'].max()+1)
    print('N classes:', n_classes)
    y_test = data1['labels']
    y_score1 = data1['logits']
    ids = data1['IDs']
    y_score2 = []
    for case_id in ids:
        index = data2['IDs'].index(case_id)
        # print('model1:', y_test[index])
        # print('model2:', data2['labels'])
        score = data2['logits'][index:index+1, :]
        y_score2.append(score)
        
    y_score2 = np.concatenate(y_score2, axis=0)
    
    y_score1 = torch.from_numpy(y_score1)
    y_score2 = torch.from_numpy(y_score2)
    y_test = torch.from_numpy(y_test)
    
    
    pvalue, m1, m2 = p_value_bootstrap(auc_fn(n_classes), y_score1, y_score2, y_test, n_resamples=1000)
    print('P values is:', pvalue)


def WSI_p_value_caiyu(result1, result2):
    """
    For Caiyu's result
    """
    
    data1 = {'labels': [], 'logits': [], 'IDs': []}
    with open(result1, 'rb') as f:
        data = pickle.load(f)
    for slide_id, values in data.items():
            gt = values['label']
            logits = values['prob']
            data1['IDs'].append(slide_id)
            data1['labels'].append(gt)
            data1['logits'].append(logits)
    data1['labels'] = np.array(data1['labels'], dtype='int')
    data1['logits'] = np.concatenate(data1['logits'], axis=0)
        
    data2 = {'labels': [], 'logits': [], 'IDs': []}
    with open(result2, 'rb') as f:
        data = pickle.load(f)
    for slide_id, values in data.items():
        gt = values['label']
        logits = values['prob']
        data2['IDs'].append(slide_id)
        data2['labels'].append(gt)
        data2['logits'].append(logits)
    data2['labels'] = np.array(data2['labels'], dtype='int')
    data2['logits'] = np.concatenate(data2['logits'], axis=0)
    
    
    n_classes = int(data1['labels'].max()+1)
    print('N classes:', n_classes)
    y_test = data1['labels']
    y_score1 = data1['logits']
    ids = data1['IDs']
    y_score2 = []
    for case_id in ids:
        index = data2['IDs'].index(case_id)
        # print('model1:', y_test[index])
        # print('model2:', data2['labels'])
        score = data2['logits'][index:index+1, :]
        y_score2.append(score)
        
    y_score2 = np.concatenate(y_score2, axis=0)
    
    y_score1 = torch.from_numpy(y_score1)
    y_score2 = torch.from_numpy(y_score2)
    y_test = torch.from_numpy(y_test)
    
    
    pvalue, m1, m2 = p_value_bootstrap(auc_fn(n_classes), y_score1, y_score2, y_test, n_resamples=1000)
    print('P values is:', pvalue)


def ROI_p_value(result1, result2):
    import torch
    data1 = torch.load(result1, map_location='cpu')
    data2 = torch.load(result2, map_location='cpu')
    logits1 = torch.cat(data1['logits'], axis=0)
    label1 = torch.cat(data1['labels'], axis=0)
    logits2 = torch.cat(data2['logits'], axis=0)
    label2 = torch.cat(data2['labels'], axis=0)
    sum = torch.sum(label1 - label2)
    print('Sum:', sum)
    n_classes = int(label1.max()+1)
    print("n classes:", n_classes)
    pvalue, m1, m2 = p_value_bootstrap(auc_fn(n_classes), logits1, logits2, label1, n_resamples=1000)
    print('P values is:', pvalue)

    


if __name__ == "__main__":
    import random
    
    survival_p_value('/jhcnas3/Pathology/experiments/train/splits82/TCGA_SKCM_survival/att_mil/phikon_s110', 
                     '/jhcnas3/Pathology/experiments/train/splits82/TCGA_SKCM_survival/att_mil/distill_87499_s110')

    # WSI_p_value('/jhcnas2/home/zhoufengtao/subtyping/PANDA/[ABMIL]/[uni]-[2024-05-08]-[03-15-06]/result.pkl',
    #             '/jhcnas2/home/zhoufengtao/subtyping/PANDA/[ABMIL]/[distill_87499]-[2024-05-08]-[03-17-27]/result.pkl')
    
    # WSI_p_value_easymil('/storage/Pathology/results/experiments/train/TCGA_GBMLGG_IDH1/att_mil/uni_s1/split_0_results.pkl',
    #     '/storage/Pathology/results/experiments/train/TCGA_GBMLGG_IDH1/att_mil/distill_87499_s1/split_0_results.pkl')

    # WSI_p_value_caiyu('/storage/Pathology/results/experiments/train/TCGA_GBMLGG_IDH1/att_mil/uni_s1/split_0_results.pkl',
    #     '/storage/Pathology/results/experiments/train/TCGA_GBMLGG_IDH1/att_mil/distill_87499_s1/split_0_results.pkl')
    
    # ROI_p_value('/home/jmabq/data/results_withlogits/PCAM/linear/dinov2_vitl/ep=0_predictions.ckpt', 
    #     '/home/jmabq/data/results_withlogits/PCAM/linear/distill_87499/ep=0_predictions.ckpt'
    # )