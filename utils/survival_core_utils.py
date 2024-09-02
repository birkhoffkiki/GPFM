import sys 
sys.path.append("..") 
from datasets import dataset_generic
from dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from utils.survival_utils import *

from argparse import Namespace
from collections import OrderedDict
import os
import pickle 

from lifelines.utils import concordance_index
import numpy as np
from sksurv.metrics import concordance_index_censored

import torch

# from utils.coattn_train_utils import *
# from utils.cluster_train_utils import *

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, warmup=5, patience=15, stop_epoch=20, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


class Monitor_CIndex:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.best_score = None

    def __call__(self, val_cindex, model, ckpt_name:str='checkpoint.pt'):

        score = val_cindex

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        else:
            pass

    def save_checkpoint(self, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), ckpt_name)


def train(datasets: tuple, cur: int, args: Namespace):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from torch.utils.tensorboard.writer import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.task_type == 'survival':
        if args.bag_loss == 'ce_surv':
            loss_fn = CrossEntropySurvLoss(alpha=args.alpha_surv)
        elif args.bag_loss == 'nll_surv':
            loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
        elif args.bag_loss == 'cox_surv':
            loss_fn = CoxSurvLoss()
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    if args.reg_type == 'omic':
        reg_fn = l1_reg_all
    elif args.reg_type == 'pathomic':
        reg_fn = l1_reg_modules
    else:
        reg_fn = None

    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    args.fusion = None if args.fusion == 'None' else args.fusion

    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()
        
        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError

    elif args.model_type == 'mean_mil':
        from models.mean_max_mil import MeanMIL
        model = MeanMIL(args.in_dim, args.n_classes, survival = True)
    elif args.model_type == 'max_mil':
        from models.mean_max_mil import MaxMIL
        model = MaxMIL(args.in_dim, args.n_classes, survival = True)
    elif args.model_type == 'att_mil':
        from models.att_mil import DAttention
        model = DAttention(args.in_dim, args.n_classes, dropout=True, act='relu', survival = True)
    elif args.model_type == 'trans_mil':
        from models.trans_mil import TransMIL
        model = TransMIL(args.in_dim, args.n_classes, dropout=True, act='gelu')
    elif args.model_type == 'deep_attn_misl':
        from models.deep_attn_misl import MIL_Cluster_FC
        dropout = 0.25 if args.drop_out else 0
        model = MIL_Cluster_FC(path_input_dim = args.in_dim, n_classes = args.n_classes, dropout = dropout)
    elif args.model_type == 'ds_mil':
        from models.ds_mil import FCLayer, BClassifier, MILNet
        dropout = 0.25 if args.drop_out else 0
        i_classifier = FCLayer(in_size=args.in_dim, out_size=args.n_classes)
        b_classifier = BClassifier(input_size=args.in_dim, output_class=args.n_classes, dropout_v=dropout)
        model = MILNet(i_classifier, b_classifier, survival = True)
    elif args.model_type == 'llama':
        from models.llama_aggregator import WSI_Expert
        model = WSI_Expert(n_classes = args.n_classes, survival = True)
    else:
        raise NotImplementedError(f'{args.model_type} is not implemented ...')
    
    # model.relocate()
    if hasattr(model, "relocate"):
        model.relocate()
    elif torch.cuda.device_count() > 1:
        model = model.to('cuda:0')
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    else:
        model = model.to(torch.device('cuda'))
    print('Done!')
    
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, 
        weighted = args.weighted_sample, mode=args.mode, batch_size=args.batch_size)
    val_loader = get_split_loader(val_split, testing = args.testing, mode=args.mode, batch_size=args.batch_size)
    test_loader = get_split_loader(test_split, testing = args.testing, mode=args.mode, batch_size=args.batch_size)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(warmup=0, patience=20, stop_epoch=50, verbose = True)
    else:
        early_stopping = None

    print('\nSetup Validation C-Index Monitor...', end=' ')
    monitor_cindex = Monitor_CIndex()
    print('Done!')

    for epoch in range(args.max_epochs):
        if args.task_type == 'survival':
            if args.mode == 'coattn':
                train_loop_survival_coattn(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn, reg_fn, args.lambda_reg, args.gc)
                stop = validate_survival_coattn(cur, epoch, model, val_loader, args.n_classes, early_stopping, monitor_cindex, writer, loss_fn, reg_fn, args.lambda_reg, args.results_dir)
            else:
                train_loop_survival(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn, reg_fn, args.lambda_reg, args.gc)
                stop = validate_survival(cur, epoch, model, val_loader, args.n_classes, early_stopping, monitor_cindex, writer, loss_fn, reg_fn, args.lambda_reg, args.results_dir)

            if stop:
                break
            
    # if args.early_stopping:
    #     model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    # else:
    #     torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    print('Done!')
    model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    _, val_cindex = summary_survival(model, val_loader, args.n_classes)
    print('Val c-Index: {:.4f}'.format(val_cindex))
    results_dict, test_cindex = summary_survival(model, test_loader, args.n_classes)
    print('Test c-Index: {:.4f}'.format(test_cindex))
    writer.close()
    return results_dict, test_cindex, val_cindex

def train_loop_survival(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., gc=16):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    train_loss_surv, train_loss = 0., 0.

    print('\n')
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    # for batch_idx, (data_WSI, data_omic, label, event_time, c) in enumerate(loader):
    for batch_idx, batch in enumerate(loader):
        
        if hasattr(model, "num_clusters"):
            data_WSI, cluster_id, data_omic, label, event_time, c = batch
            data_WSI, cluster_id, data_omic, label = data_WSI.to(device, non_blocking=True), cluster_id, data_omic.to(device, non_blocking = True), label.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)
        else:
            data_WSI, data_omic, label, event_time, c = batch
            data_WSI, data_omic = data_WSI.to(device, non_blocking = True), data_omic.to(device, non_blocking = True)
            label = label.to(device, non_blocking = True)
            c = c.to(device, non_blocking=True)
            cluster_id = None

        # hazards, S, Y_hat, _, _ = model(x_path=data_WSI, x_omic=data_omic) # return hazards, S, Y_hat, A_raw, results_dict
        hazards, S, Y_hat, _, _ = model(data_WSI)
        
        loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.item()
        all_event_times[batch_idx] = event_time

        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg

        if (batch_idx + 1) % 100 == 0:
            print('batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}, bag_size: {}'.format(batch_idx, loss_value + loss_reg, label.item(), float(event_time), float(risk), data_WSI.size(0)))
        # backward pass
        loss = loss / gc + loss_reg
        loss.backward()

        if (batch_idx + 1) % gc == 0: 
            optimizer.step()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss_surv /= len(loader)
    train_loss /= len(loader)

    # c_index = concordance_index(all_event_times, all_risk_scores, event_observed=1-all_censorships) 
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss_surv, train_loss, c_index))

    if writer:
        writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index, epoch)


def validate_survival(cur, epoch, model, loader, n_classes, early_stopping=None, monitor_cindex=None, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, (data_WSI, data_omic, label, event_time, c) in enumerate(loader):
        data_WSI, data_omic = data_WSI.to(device), data_omic.to(device)
        label = label.to(device)
        c = c.to(device)

        with torch.no_grad():
            # hazards, S, Y_hat, _, _ = model(x_path=data_WSI, x_omic=data_omic) # return hazards, S, Y_hat, A_raw, results_dict
            hazards, S, Y_hat, _, _ = model(data_WSI) # return hazards, S, Y_hat, A_raw, results_dict

        loss = loss_fn(hazards=hazards, S=S, Y=label, c=c, alpha=0)
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = -torch.sum(S, dim=1).cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.cpu().numpy()
        all_event_times[batch_idx] = event_time

        val_loss_surv += loss_value
        val_loss += loss_value + loss_reg

    val_loss_surv /= len(loader)
    val_loss /= len(loader)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    if writer:
        writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c-index', c_index, epoch)

    if early_stopping:
        assert results_dir
        # early_stopping(epoch, val_loss_surv, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        early_stopping(epoch, -c_index, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def summary_survival(model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0.

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data_WSI, data_omic, label, event_time, c) in enumerate(loader):
        data_WSI, data_omic = data_WSI.to(device), data_omic.to(device)
        label = label.to(device)
        
        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            # hazards, survival, Y_hat, _, _ = model(x_path=data_WSI, x_omic=data_omic)
            hazards, survival, Y_hat, _, _ = model(data_WSI)

        # risk = np.asscalar(-torch.sum(survival, dim=1).cpu().numpy()) 
        risk = np.ndarray.item(-torch.sum(survival, dim=1).cpu().numpy())
        # event_time = np.asscalar(event_time)
        event_time = np.ndarray.item(event_time)
        # c = np.asscalar(c)
        c = np.ndarray.item(c.cpu().numpy())
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c
        all_event_times[batch_idx] = event_time
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'risk': risk, 'disc_label': label.item(), 'survival': event_time, 'censorship': c}})

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    return patient_results, c_index