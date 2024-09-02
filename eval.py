import argparse
import torch
import os
import pandas as pd
from utils.clam_utils import *
from utils.core_trainer import TrainEngine
from datasets import get_subtying_dataset, get_survival_dataset
from downstream_tasks.metrics import build_linear_metric
import json


# Training settings
parser = argparse.ArgumentParser(description='CLAM Evaluation Script')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--model_size', type=str, default='small', 
                    help='size of model (default: small)')
parser.add_argument('--model_type', type=str, default='clam_sb', 
                    help='type of model (default: clam_sb)')
parser.add_argument('--drop_out', action='store_true', default=False, 
                    help='whether model uses dropout')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False, 
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str)
parser.add_argument('--task_type', type=str, default='subtyping')
parser.add_argument('--backbone', type=str, default='resnet50')
parser.add_argument('--save_dir', type=str)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--in_dim', type=str, default='1024')
parser.add_argument('--n_classes', type=int, default=2)
# new for 712, bootstrapping and balanced evaluation
parser.add_argument('--bootstrap', action='store_true', default=False)
# for external validation
parser.add_argument('--models_dir', type=str, default='')



args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.save_dir = os.path.join(args.save_dir, str(args.save_exp_code))

if args.models_dir == '':
    args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir

print('model dir:', args.models_dir)
assert os.path.isdir(args.models_dir), f"{args.models_dir}"
assert os.path.isdir(args.splits_dir), f"{args.splits_dir}"

settings = {'task': args.task,
            'split': args.split,
            'save_dir': args.save_dir, 
            'models_dir': args.models_dir,
            'model_type': args.model_type,
            'drop_out': args.drop_out,
            'model_size': args.model_size}

print(settings)

with open(os.path.join(args.save_dir, 'eval_experiment.txt'), 'w') as f:
    print(settings, file=f)
f.close()

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)
ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

if __name__ == "__main__":
    # get dataset
    if args.task_type == 'subtyping':
        dataset = get_subtying_dataset(args.task, args.seed, args.data_root_dir)
        
        all_results = []
        all_auc = []
        all_acc = []
        all_f1 = []
        
    elif args.task_type == 'survival':
        dataset = get_survival_dataset(args.task, args.seed, args.data_root_dir)
        
        all_results = []
        all_c_index = []
        
    else:
        raise NotImplementedError(f'{args.task_type} is not implemented...')

    for ckpt_idx in range(len(ckpt_paths)):

        csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
        split_datasets = dataset.return_splits(args.backbone, from_id=False, csv_path=csv_path)
        
        drop_out = 0.25 if args.drop_out else 0.0
        # reset bag loss to solve incompatible problem
        bag_loss = 'nll_surv' if args.task_type == 'survival' else 'ce'
        train_engine = TrainEngine(datasets=split_datasets, fold=ckpt_idx, result_dir=args.results_dir, mil_model_name=args.model_type,
                                   optimizer_name='adam', lr=1e-4, regularization=1e-2, weighted_sample=False,
                                   batch_size=1, task_type=args.task_type, max_epochs=1, in_dim=args.in_dim, 
                                   n_classes=args.n_classes, drop_out=drop_out, dataset_name=args.task, bag_loss=bag_loss)
        
        print('checkpoint path:', ckpt_paths[ckpt_idx])
        if args.task_type == 'subtyping':
            patient_results, test_error, auc, df, f1  = train_engine.eval_model(ckpt_paths[ckpt_idx])
            all_results.append(patient_results)
            all_auc.append(auc)
            all_f1.append(f1)
            all_acc.append(1-test_error)
            df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)

        elif args.task_type == 'survival':
            patient_results, test_error, c_index, df = train_engine.eval_model(ckpt_paths[ckpt_idx])
            all_results.append(patient_results)
            all_c_index.append(c_index)
            df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)
        else:
            raise NotImplementedError(f'{args.task_type} is not implmentated for evaluation, please implemnet it by youself...')

    if args.task_type == 'subtyping':
        final_df = pd.DataFrame({'folds': folds, 'test_auc': all_auc, 'test_acc': all_acc, 'test_f1': all_f1})
    elif args.task_type == 'survival':
        final_df = pd.DataFrame({'folds': folds, 'test_c_index': all_c_index})
        # Calculate average C-Index
        average_c_index = final_df['test_c_index'].mean()
        final_df['average_test_c_index'] = average_c_index
        
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.save_dir, save_name))

    # --------- new evaluation-----------
    if args.bootstrap:
        print('>> Do bootstrapping...')
        if args.task_type == 'subtyping':
            metric_fn = build_linear_metric(num_classes=args.n_classes, bootstrap=args.bootstrap)
        else:
            raise NotImplementedError
        for slide_id, content in patient_results.items():
            prob = torch.from_numpy(content['prob'])
            label = torch.tensor(content['label'])[None]
            metric_fn.update(prob, label)
        result = metric_fn.compute()
        result = {k: v.item() for k, v in result.items()}
        print(result)
        with open(os.path.join(args.save_dir, 'bootstrap_result.json'), 'w') as f:
            json.dump(result, f)

    