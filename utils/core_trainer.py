"""
Universal MIL trainer
@Author: MA JIABO, GUO Zhengrui

"""
import numpy as np
import pandas as pd
import torch
import os
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from sklearn.metrics import auc as calc_auc
from torch.utils.tensorboard.writer import SummaryWriter

from datasets.dataset_generic import save_splits
from mil_models import find_mil_model
from .comm_utils import Accuracy_Logger, EarlyStopping
from .clam_utils import print_network, get_split_loader, calculate_error
from utils.survival_utils import CrossEntropySurvLoss, NLLSurvLoss, CoxSurvLoss
try:
    from sksurv.metrics import concordance_index_censored
except:
    print('sksurv is not installed, survival task is not supported.')


class TrainEngine:
    def __init__(self, datasets, fold, result_dir, mil_model_name, optimizer_name, 
                 lr, regularization, weighted_sample, batch_size, task_type, max_epochs,
                 in_dim, n_classes, drop_out, dataset_name=None, bag_loss='ce') -> None:
        self.train_split, self.val_split, self.test_split = datasets
        self.fold = fold
        self.result_dir = result_dir
        self.mil_model_name = mil_model_name
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.regularization = regularization
        self.weighted_sample = weighted_sample
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.drop_out = drop_out
        self.dataset_name = dataset_name
        
        assert task_type in ['subtyping', 'survival'], f'{task_type} is not supported.'
        self.task_type = task_type
        self.bag_loss = bag_loss
    
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        save_splits(datasets, ['train', 'val', 'test'], os.path.join(self.result_dir, 'splits_{}.csv'.format(fold)))
        
        # init logger
        self.init_logger()

        # init models
        self.call_scheduler = self.get_lr_scheduler()
        self.model = self.get_mil_model()
        self.loss_function = self.get_loss_function()
        self.optimizer = self.get_optimizer()
        # change early_stop parameter to speed up training
        patience = self.model.early_stopping_patience if hasattr(self.model, 'early_stopping_patience') else 20
        stop_epoch = self.model.early_stopping_stop_epoch if hasattr(self.model, 'early_stopping_stop_epoch') else 50
        self.early_stopping = EarlyStopping(patience = patience, stop_epoch=stop_epoch, verbose = True)
        # init dataloader
        self.train_loader, self.val_loader, self.test_loader = self.init_data_loaders()
        
        # core, if you implement your training framework, call setup to pass training hyperparameters.
        if hasattr(self.model, 'set_up'):
            extra_args = {'total_iterations': max_epochs*len(self.train_loader)}
            self.model.set_up(lr=lr, max_epochs=max_epochs, weight_decay=regularization, **extra_args)
        
    def init_logger(self):
        print('\nTraining Fold {}!'.format(self.fold))
        writer_dir = os.path.join(self.result_dir, str(self.fold))
        if not os.path.isdir(writer_dir):
            os.mkdir(writer_dir)
        self.writer = SummaryWriter(writer_dir, flush_secs=15)
        print('Done!')
        print("Training on {} samples".format(len(self.train_split)))
        print("Validating on {} samples".format(len(self.val_split)))
        print("Testing on {} samples".format(len(self.test_split)))

    def get_lr_scheduler(self):
        return None

    def get_mil_model(self):
        model = find_mil_model(self.mil_model_name, self.in_dim, self.n_classes, self.drop_out, self.task_type, self.dataset_name)
        if hasattr(model, "relocate"):
            model.relocate()
        else:
            model = model.to(self.device)
        print_network(model)
        return model
        
    def get_loss_function(self):
        """get loss function, if you defined a loss function in MIL model, then we will use it, other use CE loss.

        Returns:
            nn.Loss: loss function
        """
        if hasattr(self.model, 'loss_function'):
            print('The loss function defined in the MIL model is adopted...')
            loss_function = self.model.loss_function
        elif self.task_type == 'subtyping':
            print('Default CE loss function is adopted...')
            loss_function = torch.nn.CrossEntropyLoss()
        elif self.task_type == 'survival':
            if self.bag_loss == 'ce_surv':
                loss_function = CrossEntropySurvLoss(alpha=0.0)
            elif self.bag_loss == 'nll_surv':
                loss_function = NLLSurvLoss(alpha=0.0)
            elif self.bag_loss == 'cox_surv':
                loss_function = CoxSurvLoss()
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return loss_function
    
    def get_optimizer(self):
        print('Init optimizer ...')
        if self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, weight_decay=self.regularization)
        elif self.optimizer_name.lower() == 'sgd':
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, momentum=0.9, weight_decay=self.regularization)
        else:
            raise NotImplementedError
        return optimizer
    
    def init_data_loaders(self):
        print('\nInit Loaders...', end=' ')
        train_loader = get_split_loader(self.train_split, training=True, weighted = self.weighted_sample,
                                        batch_size=self.batch_size)
        val_loader = get_split_loader(self.val_split)
        test_loader = get_split_loader(self.test_split)
        return train_loader, val_loader, test_loader
        
    def train_model(self, fold):
        if self.task_type == 'subtyping':
            train_loop_func = self.train_loop_subtyping
            validate_func = self.validate_subtyping
            test_func = self.summary_subtyping
        elif self.task_type == 'survival':
            train_loop_func = self.train_loop_survival
            validate_func = self.validate_survival
            test_func = self.summary_survival
            
        else:
            raise NotImplementedError(f'Training loop and val loop have not been implemented for {self.task_type}')

        for epoch in range(self.max_epochs):
            train_loop_func(epoch)
            stop = validate_func(epoch)
            if stop: 
                break
        # load saved model
        msg = self.model.load_state_dict(torch.load(os.path.join(self.result_dir, "s_{}_checkpoint.pt".format(fold))))
        print('Loading trained model...')
        print(msg)
        
        if self.task_type == 'subtyping':
            # test_func on val loader
            _, val_error, val_auc, _, _, val_f1 = test_func(self.val_loader)
            # test on test loader
            results_dict, test_error, test_auc, acc_logger, _, test_f1 = test_func(self.test_loader)
            print('Test error: {:.4f}, ROC AUC: {:.4f}, F1 Score: {:.4f}'.format(test_error, test_auc, test_f1))
            print('val error: {:.4f}, ROC AUC: {:.4f}, F1 Score: {:.4f}'.format(val_error, val_auc, val_f1))

            for i in range(self.n_classes):
                acc, correct, count = acc_logger.get_summary(i)
                print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
                self.writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

            self.writer.add_scalar('final/val_error', val_error, 0)
            self.writer.add_scalar('final/val_auc', val_auc, 0)
            self.writer.add_scalar('final/test_error', test_error, 0)
            self.writer.add_scalar('final/test_auc', test_auc, 0)
            self.writer.close()
            return results_dict, test_auc, val_auc, 1-test_error, 1-val_error, test_f1, val_f1

        elif self.task_type == "survival":
            _, _, val_cindex, _ = test_func(self.val_loader)
            print('Val c-Index: {:.4f}'.format(val_cindex))
            results_dict, _, test_cindex, _ = test_func(self.test_loader)
            print('Test c-Index: {:.4f}'.format(test_cindex))
            
            self.writer.add_scalar('final/val_cindex', val_cindex, 0)
            self.writer.add_scalar('final/test_cindex', test_cindex, 0)
            self.writer.close()
            return results_dict, test_cindex, val_cindex
        
        else:
            raise NotImplementedError(f'Training has not been implemented for {self.task_type}')

    def train_loop_subtyping(self, epoch):   
        self.model.train()
        acc_logger = Accuracy_Logger(n_classes=self.n_classes)
        train_loss = 0.
        train_error = 0.

        print('\n')
        print('Epoch: {}'.format(epoch))
        
        for batch_idx, batch in enumerate(self.train_loader):
            iteration = epoch*len(self.train_loader) + batch_idx
            kwargs = {}
            data = batch['features']
            label = batch['label']
            kwargs['iteration'] = iteration
            kwargs['image_call'] = batch['image_call']
            # Core 1: If you model need specific pre-process of data and label, please implement following function,
            # we will call to keep code clean
            if hasattr(self.model, 'process_data'):
                data, label = self.model.process_data(data, label, self.device)
            else:
                data = data.to(self.device)
                label = label.to(self.device)
            # core 2: If you model has special optimizing strategy, e.g., using mutilpe optimizers, please
            # define your own update parameter code in one_step function. You may also need to define optimizes in you MIL model.
            if hasattr(self.model, 'one_step'):
                outputs = self.model.one_step(data, label, **kwargs)
                loss = outputs['loss']
                if 'call_scheduler' in outputs.keys():
                    self.call_scheduler = outputs['call_scheduler']
                logits, Y_prob, Y_hat = outputs['wsi_logits'], outputs['wsi_prob'], outputs['wsi_label']

            else:
                # use univer code to update param
                kwargs['label'] = label
                outputs = self.model(data, **kwargs)
                logits, Y_prob, Y_hat = outputs['wsi_logits'], outputs['wsi_prob'], outputs['wsi_label']

                if hasattr(self.model, 'loss_function'):
                    loss = self.loss_function(logits, label, **outputs)
                else:
                    loss = self.loss_function(logits, label)
                    
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                outputs['loss'] = loss
            
            # to support batch size greater than 1
            if isinstance(label, torch.Tensor):
                acc_logger.log(Y_hat, label)
            else:
                for i in range(len(data)):
                    acc_logger.log(Y_hat[i], label[i])

            loss_value = loss.item()
            if torch.isnan(loss):
                print('logits:', logits)
                print('Y_prob:', Y_prob)
                print('loss:', loss)
                raise RuntimeError('Found Nan number')
            
            if (batch_idx + 1) % 20 == 0:
                bag_size = data[0].shape[0] if isinstance(data, list) else data.shape[0]
                print('batch {}'.format(batch_idx), end=',')
                for k, v in outputs.items():
                    if 'loss' in k:
                        print('{}:{:.4f}'.format(k, v.item()), end=',')
                print(' label: {}, bag_size: {}'.format(label.item(), bag_size), flush=True)
                    
                    
            error = calculate_error(Y_hat, label)
            train_loss += loss_value
            train_error += error

        # calculate loss and error for epoch
        train_loss /= len(self.train_loader)
        train_error /= len(self.train_loader)

        print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
        if self.writer:
            self.writer.add_scalar('train/loss', train_loss, epoch)
            self.writer.add_scalar('train/error', train_error, epoch)
            
        for i in range(self.n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
            if self.writer:
                self.writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

        if self.call_scheduler is not None:
            self.call_scheduler()

    def validate_subtyping(self, epoch):
        self.model.eval()
        acc_logger = Accuracy_Logger(n_classes=self.n_classes)
        val_loss = 0.
        val_error = 0.   

        prob = np.zeros((len(self.val_loader), self.n_classes))
        labels = np.zeros(len(self.val_loader))    
        Y_hats = np.zeros(len(self.val_loader))    
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # print('Evaluating: [{}/{}]'.format(batch_idx+1, len(self.val_loader)), flush=True)
                kwargs = {}
                data = batch['features']
                label = batch['label']
                kwargs['image_call'] = batch['image_call']
                if hasattr(self.model, 'process_data'):
                    data, label = self.model.process_data(data, label, self.device)
                else:
                    data = data.to(self.device)
                    label = label.to(self.device)

                if hasattr(self.model, 'wsi_predict'):
                    outputs = self.model.wsi_predict(data, **kwargs)
                else:
                    # use univer code to update param
                    outputs = self.model(data)

                logits, Y_prob, Y_hat = outputs['wsi_logits'], outputs['wsi_prob'], outputs['wsi_label']
                acc_logger.log(Y_hat, label)
                # if hasattr(self.model, 'loss_function'):
                #     loss = self.loss_function(logits, label, **outputs)
                # else:
                #     loss = self.loss_function(logits, label)
                try:
                    loss = self.loss_function(logits, label, **outputs)
                except:
                    loss = self.loss_function(logits, label)
                    
                prob[batch_idx] = Y_prob.cpu().numpy()
                labels[batch_idx] = label.item()
                Y_hats[batch_idx] = Y_hat.item()
                
                val_loss += loss.item()
                error = calculate_error(Y_hat, label)
                val_error += error
                
        val_error /= len(self.val_loader)
        val_loss /= len(self.val_loader)
        if self.n_classes == 2:
            auc = roc_auc_score(labels, prob[:, 1])
        else:
            auc = roc_auc_score(labels, prob, multi_class='ovr')
        # f1 score
        f1 = f1_score(labels, Y_hats, average='macro')
 
        if self.writer:
            self.writer.add_scalar('val/loss', val_loss, epoch)
            self.writer.add_scalar('val/auc', auc, epoch)
            self.writer.add_scalar('val/error', val_error, epoch)

        print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}, f1 score: {:.4f}'.format(val_loss, val_error, auc, f1))
        for i in range(self.n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

        # val_error is better than val_loss
        self.early_stopping(epoch, val_error, self.model, ckpt_name = os.path.join(self.result_dir, "s_{}_checkpoint.pt".format(self.fold)))
        
        if self.early_stopping.early_stop:
            print("Early stopping")
            return True
        else:
            return False

    def summary_subtyping(self, loader=None):
        if loader is None:
            loader = self.test_loader
        
        acc_logger = Accuracy_Logger(n_classes=self.n_classes)
        self.model.eval()
        test_error = 0.

        all_probs = np.zeros((len(loader), self.n_classes))
        all_labels = np.zeros(len(loader))
        all_preds = np.zeros(len(loader))

        slide_ids = loader.dataset.slide_data['slide_id']
        patient_results = {}
        
        # Create lists to store all predictions and labels
        all_Y_hat = []
        all_label = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                data = batch['features']
                label = batch['label']
                if hasattr(self.model, 'process_data'):
                    data, label = self.model.process_data(data, label, self.device)
                else:
                    data = data.to(self.device)
                    label = label.to(self.device)

                if hasattr(self.model, 'wsi_predict'):
                    outputs = self.model.wsi_predict(data, **batch)
                else:
                    # use univer code to update param
                    outputs = self.model(data)

                logits, Y_prob, Y_hat = outputs['wsi_logits'], outputs['wsi_prob'], outputs['wsi_label']
                slide_id = slide_ids.iloc[batch_idx]
                acc_logger.log(Y_hat, label)

                probs = Y_prob.cpu().numpy()
                all_probs[batch_idx] = probs
                all_labels[batch_idx] = label.item()
                all_preds[batch_idx] = Y_hat.item()
                
                patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
                error = calculate_error(Y_hat, label)
                test_error += error
                # Append current predictions and labels to the lists
                all_Y_hat.append(Y_hat.cpu().numpy())
                all_label.append(label.cpu().numpy())

        test_error /= len(loader)

        if self.n_classes == 2:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
            aucs = []
        else:
            aucs = []
            binary_labels = label_binarize(all_labels, classes=[i for i in range(self.n_classes)])
            for class_idx in range(self.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(calc_auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))

            auc = np.nanmean(np.array(aucs))
            
        # Convert the lists of all predictions and labels to numpy arrays
        all_Y_hat = np.concatenate(all_Y_hat)
        all_label = np.concatenate(all_label)

        # Calculate the F1 score
        f1 = f1_score(all_label, all_Y_hat, average='macro')
        results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
        for c in range(self.n_classes):
            results_dict.update({'p_{}'.format(c): all_probs[:,c]})
        print(results_dict)
        df = pd.DataFrame(results_dict)

        return patient_results, test_error, auc, acc_logger, df, f1

    def train_loop_survival(self, epoch):
        self.model.train()
        
        train_loss = 0.
        train_error = 0.

        print('\n')
        print('Epoch: {}'.format(epoch))
        all_risk_scores = np.zeros((len(self.train_loader)))
        all_censorships = np.zeros((len(self.train_loader)))
        all_event_times = np.zeros((len(self.train_loader)))

        for batch_idx, batch in enumerate(self.train_loader):
            iteration = epoch*len(self.train_loader) + batch_idx
            kwargs = {}
            data = batch['features']
            label = batch['label']
            event_time = batch['event_time']
            c = batch['c']
            kwargs['iteration'] = iteration
            kwargs['image_call'] = batch['image_call']
            # Core 1: If you model need specific pre-process of data and label, please implement following function,
            # we will call to keep code clean
            if hasattr(self.model, 'process_data'):
                data, label = self.model.process_data(data, label, self.device)
                c = torch.tensor(c).to(self.device, non_blocking=True)
            else:
                data = data.to(self.device)
                label = label.to(self.device)
                c = torch.tensor(c).to(self.device, non_blocking=True)
            kwargs['c'] = c
            # core 2: If you model has special optimizing strategy, e.g., using mutilpe optimizers, please
            # define your own update parameter code in one_step function. You may also need to define optimizes in you MIL model.
            if hasattr(self.model, 'one_step'):
                outputs = self.model.one_step(data, label, **kwargs)
                loss = outputs['loss']
                if 'call_scheduler' in outputs.keys():
                    self.call_scheduler = outputs['call_scheduler']
                hazards, S, Y_hat = outputs['wsi_logits'], outputs['wsi_prob'], outputs['wsi_label']

            else:
                # use univer code to update param
                kwargs['label'] = label
                outputs = self.model(data, **kwargs)
                hazards, S, Y_hat = outputs['wsi_logits'], outputs['wsi_prob'], outputs['wsi_label']

                if hasattr(self.model, 'loss_function'):
                    loss = self.loss_function(hazards=hazards, S=S, Y=label, c=c, **outputs)
                else:
                    loss = self.loss_function(hazards=hazards, S=S, Y=label, c=c)
                    
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.item()
            all_event_times[batch_idx] = event_time

            loss_value = loss.item()
            if torch.isnan(loss):
                print('hazards:', hazards)
                print('S:', S)
                print('loss:', loss)
                raise RuntimeError('Found Nan number')
                    
            if (batch_idx + 1) % 20 == 0:
                if isinstance(data, (list, tuple)):
                    data = data[0]
                print('batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}, bag_size: {}'.format(batch_idx, 
                        loss_value, label.item(), float(event_time), float(risk), data.size(0)), flush=True)
                    
            train_loss += loss_value
            error = calculate_error(Y_hat, label)
            train_error += error

        # calculate loss for epoch
        train_loss /= len(self.train_loader)
        train_error /= len(self.train_loader)
            
        c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

        print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss, train_error, c_index))
        if self.writer:
            self.writer.add_scalar('train/loss', train_loss, epoch)
            self.writer.add_scalar('train/error', train_error, epoch)
            self.writer.add_scalar('train/c_index', c_index, epoch)

        if self.call_scheduler is not None:
            self.call_scheduler()
    
    def validate_survival(self, epoch):
        self.model.eval()

        val_loss = 0.
        val_error = 0.   
        
        all_risk_scores = np.zeros((len(self.val_loader)))
        all_censorships = np.zeros((len(self.val_loader)))
        all_event_times = np.zeros((len(self.val_loader)))

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                kwargs = {}
                data = batch['features']
                label = batch['label']
                event_time = batch['event_time']
                c = batch['c']
                kwargs['image_call'] = batch['image_call']
                if hasattr(self.model, 'process_data'):
                    data, label = self.model.process_data(data, label, self.device)
                    c = torch.tensor(c).to(self.device, non_blocking=True)
                else:
                    data = data.to(self.device)
                    label = label.to(self.device)
                    c = torch.tensor(c).to(self.device, non_blocking=True)

                if hasattr(self.model, 'wsi_predict'):
                    outputs = self.model.wsi_predict(data, **kwargs)
                else:
                    # use univer code to update param
                    outputs = self.model(data)

                hazards, S, Y_hat = outputs['wsi_logits'], outputs['wsi_prob'], outputs['wsi_label']

                # if hasattr(self.model, 'loss_function'):
                #     loss = self.loss_function(hazards=hazards, S=S, Y=label, c=c, alpha=0, **outputs)
                # else:
                #     loss = self.loss_function(hazards=hazards, S=S, Y=label, c=c, alpha=0)
                try:
                    loss = self.loss_function(hazards=hazards, S=S, Y=label, c=c, **outputs)
                except:
                    loss = self.loss_function(hazards=hazards, S=S, Y=label, c=c)
                    
                risk = -torch.sum(S, dim=1).cpu().numpy()
                all_risk_scores[batch_idx] = risk
                all_censorships[batch_idx] = c.cpu().numpy()
                all_event_times[batch_idx] = event_time
                
                val_loss += loss.item()
                error = calculate_error(Y_hat, label)
                val_error += error
                
        val_error /= len(self.val_loader)
        val_loss /= len(self.val_loader)
        c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
 
        if self.writer:
            self.writer.add_scalar('val/loss', val_loss, epoch)
            self.writer.add_scalar('val/error', val_error, epoch)
            self.writer.add_scalar('val/c-index', c_index, epoch)

        print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, c-index: {:.4f}'.format(val_loss, val_error, c_index)) 

        # val_error is better than val_loss
        self.early_stopping(epoch, -c_index, self.model, ckpt_name = os.path.join(self.result_dir, "s_{}_checkpoint.pt".format(self.fold)))
        
        if self.early_stopping.early_stop:
            print("Early stopping")
            return True
        else:
            return False
    
    def summary_survival(self, loader = None):
        if loader is None:
            loader = self.test_loader
        
        self.model.eval()
        test_error = 0.
        all_risk_scores = np.zeros((len(loader)))
        all_censorships = np.zeros((len(loader)))
        all_event_times = np.zeros((len(loader)))

        slide_ids = loader.dataset.slide_data['slide_id']
        patient_results = {}
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                
                data = batch['features']
                label = batch['label']
                event_time = batch['event_time']
                c = batch['c']
                
                if hasattr(self.model, 'process_data'):
                    data, label = self.model.process_data(data, label, self.device)
                    c = torch.tensor(c).to(self.device, non_blocking=True)
                else:
                    data = data.to(self.device)
                    label = label.to(self.device)
                    c = torch.tensor(c).to(self.device, non_blocking=True)

                if hasattr(self.model, 'wsi_predict'):
                    outputs = self.model.wsi_predict(data, **batch)
                else:
                    # use univer code to update param
                    outputs = self.model(data)
                    
                hazards, survival, Y_hat = outputs['wsi_logits'], outputs['wsi_prob'], outputs['wsi_label']
            
                slide_id = slide_ids.iloc[batch_idx]
                 
                risk = np.ndarray.item(-torch.sum(survival, dim=1).cpu().numpy())
                # event_time = np.ndarray.item(event_time)
                event_time = float(event_time) #!
                c = np.ndarray.item(c.cpu().numpy())
                all_risk_scores[batch_idx] = risk
                all_censorships[batch_idx] = c
                all_event_times[batch_idx] = event_time
                
                patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'risk': risk, 'hazards': hazards, 'disc_label': label.item(), 'survival': event_time, 'censorship': c}})

                error = calculate_error(Y_hat, label)
                test_error += error
        
        test_error /= len(loader)
        c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        df = pd.DataFrame(patient_results)
        return patient_results, test_error, c_index, df
        
    def eval_model(self, ckpt_path):
        if hasattr(self.model, 'load_model'):
            print('Using built-in API to load the ckpt...')
            self.model.load_model(ckpt_path)
        else:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            msg = self.model.load_state_dict(ckpt)
            print('loading results:', msg)
            
        if self.task_type == 'subtyping':
            func = self.summary_subtyping
            patient_results, test_error, auc, _, df, f1 = func(self.test_loader)
            return patient_results, test_error, auc, df, f1
        elif self.task_type == 'survival':
            func = self.summary_survival
            patient_results, test_error, c_index, df = func(self.test_loader)
            return patient_results, test_error, c_index, df
        else:
            raise NotImplementedError(f'{self.task_type} is not supported, please implement it here.') 
          


