import argparse
import os
from utils.file_utils import save_pkl
from utils.core_trainer import TrainEngine
from datasets import get_subtying_dataset, get_survival_dataset

import torch
import pandas as pd
import numpy as np


def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    folds = np.arange(start, end)
    
    # get dataset
    if args.task_type == 'subtyping':
        dataset = get_subtying_dataset(args.task, args.seed)
        all_test_auc = []
        all_val_auc = []
        all_test_acc = []
        all_val_acc = []
        all_test_f1 = []
        all_val_f1 = []
        
    elif args.task_type == 'survival':
        dataset = get_survival_dataset(args.task, args.seed, args.data_root_dir)
        args.n_classes = 4
        latest_test_cindex = []
        latest_val_cindex = []
    else:
        raise NotImplementedError(f'{args.task_type} is not implemented...')
    
    for i in folds:
        seed_torch(args.seed)
        dataset_split = dataset.return_splits(args.backbone, args.patch_size, from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        
        if args.preloading == 'yes':
            for d in dataset_split:
                d.pre_loading()
            
        # init train enginge
        drop_out = 0.25 if args.drop_out else 0.0
        train_engine = TrainEngine(datasets=dataset_split, fold=i, result_dir=args.results_dir, mil_model_name=args.model_type,
                                   optimizer_name=args.opt, lr=args.lr, regularization=args.reg, weighted_sample=args.weighted_sample,
                                   batch_size=args.batch_size, task_type=args.task_type, max_epochs=args.max_epochs,
                                   in_dim=args.in_dim, n_classes=args.n_classes, drop_out=drop_out, dataset_name=args.task, bag_loss=args.bag_loss)
        if args.task_type == 'subtyping':
            results, test_auc, val_auc, test_acc, val_acc, test_f1, val_f1  = train_engine.train_model(i)

            all_test_auc.append(test_auc)
            all_val_auc.append(val_auc)
            all_test_acc.append(test_acc)
            all_val_acc.append(val_acc)
            all_test_f1.append(test_f1)
            all_val_f1.append(val_f1)

        elif args.task_type == 'survival':
            results, cindex_test, cindex_val = train_engine.train_model(i)
            latest_val_cindex.append(cindex_val)
            latest_test_cindex.append(cindex_test)
        else:
            raise NotImplementedError(f'{args.task_type} is not implemented...')
        
        # write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    if args.task_type == 'subtyping':
        final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 
            'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc, 'test_f1': all_test_f1, 'val_f1': all_val_f1})
    elif args.task_type == 'survival':
        final_df = pd.DataFrame({'folds': folds, 'test_cindex': latest_test_cindex, 
            'val_cindex': latest_val_cindex, })
    else:
        raise NotImplementedError(f'{args.task_type} is not implemented...')

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))


# Generic training settings
def parse_args():
    parser = argparse.ArgumentParser(description='Configurations for WSI Training')
    parser.add_argument('--data_root_dir', type=str, default=None, 
                        help='data directory')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='maximum number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--label_frac', type=float, default=1.0,
                        help='fraction of training labels (default: 1.0)')
    parser.add_argument('--reg', type=float, default=1e-5,
                        help='weight decay (default: 1e-5)')
    parser.add_argument('--seed', type=int, default=1, 
                        help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
    parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
    parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
    parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
    parser.add_argument('--split_dir', type=str, default=None, 
                        help='manually specify the set of splits to use, ' 
                        +'instead of infering from the task and label_frac argument (default: None)')
    parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
    parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
    parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
    parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
    parser.add_argument('--drop_out', action='store_true', default=True, help='enable dropout (p=0.25)')
    parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce', 'ce_surv', 'nll_surv', 'cox_surv'], default='ce',
                        help='slide-level classification loss function (default: ce)')
    parser.add_argument('--model_type', type=str, default='clam_sb', 
                        help='type of model (default: clam_sb, clam w/ single attention branch)')
    parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
    parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
    parser.add_argument('--model_size', type=str, choices=['small', 'big', 'small-768'], default='small', help='size of model, does not affect mil')
    parser.add_argument('--task', type=str)
    ### CLAM specific options
    parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                        help='disable instance-level clustering')
    parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                        help='instance-level clustering loss function (default: None)')
    parser.add_argument('--task_type', type=str, choices=['subtyping', 'survival'], default='subtyping', help='The type of task')

    parser.add_argument('--bag_weight', type=float, default=0.7,
                        help='clam: weight coefficient for bag-level loss (default: 0.7)')
    parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--patch_size', type=str, default='')
    parser.add_argument('--preloading', type=str, default='no')
    parser.add_argument('--in_dim', type=str, default='1024')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_classes', type=int, default=2)

    args = parser.parse_args()

    settings = {'num_splits': args.k, 
                'k_start': args.k_start,
                'k_end': args.k_end,
                'task': args.task,
                'max_epochs': args.max_epochs, 
                'results_dir': args.results_dir, 
                'lr': args.lr,
                'experiment': args.exp_code,
                'reg': args.reg,
                'label_frac': args.label_frac,
                'bag_loss': args.bag_loss,
                'seed': args.seed,
                'model_type': args.model_type,
                'model_size': args.model_size,
                "use_drop_out": args.drop_out,
                'weighted_sample': args.weighted_sample,
                'opt': args.opt}

    if args.model_type in ['clam_sb', 'clam_mb']:
        settings.update({'bag_weight': args.bag_weight,
                            'inst_loss': args.inst_loss,
                            'B': args.B})

    print('\nLoad Dataset')
    args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)

    if args.split_dir is None:
        args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
    # else:
    #     args.split_dir = os.path.join('splits', args.split_dir)

    print('split_dir: ', args.split_dir)
    assert os.path.isdir(args.split_dir)
    settings.update({'split_dir': args.split_dir})
    with open(args.results_dir + '/experiment.txt', 'w') as f:
        print(settings, file=f)

    print("################# Settings ###################")
    for key, val in settings.items():
        print("{}:  {}".format(key, val))        
    # set auto resume 
    if args.k_start == -1:
        folds = args.k if args.k_end == -1 else args.k_end
        for i in range(folds):
            filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
            if not os.path.exists(filename):
                args.k_start = i
                break
        print('Training from fold: {}'.format(args.k_start))
    
    return args


def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    

if __name__ == "__main__":
    args = parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(args.seed)

    results = main(args)
    print("finished!")
    print("end script")


