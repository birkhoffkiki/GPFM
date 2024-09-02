import torch
import numpy as np
import torch.nn as nn
import random
import torch.nn.functional as F
from .mil import BaseMILModel


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class Pdropout(nn.Module):
    def __init__(self,p=0,ic=1):
        super(Pdropout,self).__init__()
        if not(0 <= p <= 1):
            raise ValueError("Drop rate must be in range [0,1]")
        self.p = p 
        
    def forward(self,input):
        if not self.training:
            return input
        else:
            importances = torch.mean(input,dim=1,keepdim=True)
            importances = torch.sigmoid(importances)
            #print(importances)
            mask = self.generate_mask(importances,input)
            
            #print(mask)
            input = input*mask
            return input
        
    def generate_mask(self,importance,input):
        n,f = input.shape
        interpolation = self.non_linear_interpolation(self.p,0,n).to(input.device)
        mask = torch.zeros_like(importance)
        mask = mask.to(input.device)
        _, indx = torch.sort(importance,dim=0)
        #print(indx)
        idx = indx.view(-1)
        mask.index_add_(0,idx,interpolation)
        #mask 
        sampler = torch.rand(mask.shape[0],mask.shape[1]).to(input.device)

        mask = (sampler < mask).float()
        mask = 1 - mask
        return mask
    
    def non_linear_interpolation(self,max,min,num):
        e_base = 20
        log_e = 1.5
        res = (max - min)/log_e* np.log10((np.linspace(0, np.power(10,(log_e)) - 1, num)+ 1)) + min
        res = torch.from_numpy(res).float()
        return res


class LinearScheduler():
    def __init__(self, model, start_value, stop_value, nr_steps):
        self.model = model
        self.i = 0
        self.dropoutLayers = []
        self.drop_values = np.linspace(start=start_value, stop=stop_value, num=int(nr_steps))
        self.drop_values = self.dropvalue_sampler(start_value,stop_value,int(nr_steps))
        for name, layer in model.named_modules():
            if isinstance(layer, Pdropout):
                #print(name, layer)
                self.dropoutLayers.append(layer)
    def step(self):
        #for name, layer in model.named_modules():
        #dropout = []
        if self.i < len(self.drop_values):
            for i in range(len(self.dropoutLayers)):
                self.dropoutLayers[i].p = self.drop_values[self.i]
                # print('Dropout ratio:', self.dropoutLayers[i].p)

        self.i += 1

    def dropvalue_sampler(self,min,max,num):
        e_base = 20
        log_e = 1.5
        res = (max - min)/log_e* np.log10((np.linspace(0, np.power(10,(log_e)) - 1, num)+ 1)) + min
        #res =  (max - min)*(0.5*(1-np.cos(np.linspace(0, math.pi, num)))) + min
        #res = (max-min)/e_base *(np.power(10,(np.linspace(0, np.log10(e_base+1), num))) - 1) + min

        return res


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class Simple(BaseMILModel):
    def __init__(self, in_dim=1024, n_classes=1, act='relu', task='subtyping'):
        super(Simple, self).__init__(task=task)
        self.L = 512
        self.D = 128
        self.K = 1
        self.teacher_update_iter = 40
        self.n_classes = n_classes
        self.use_ema = False
        
        act_fun = nn.GELU if act == 'gelu' else nn.ReLU
        dropout_p = 0.2
        dropout_fn = nn.Dropout

        self.feature = nn.Sequential(nn.Linear(in_dim, self.L), act_fun(), dropout_fn(dropout_p))
        
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.classifier = nn.Sequential(nn.Linear(self.L*self.K, n_classes+1))
        self.drop_max = 0
        self.loss_fn = nn.CrossEntropyLoss()
        # self.loss_fn = FocalLoss(gamma=2.0)
        
        if self.use_ema:
            self.ema_feature = nn.Sequential(nn.Linear(in_dim, self.L), act_fun(), dropout_fn(dropout_p))
            self.ema_attention = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh(), nn.Linear(self.D, self.K))
            self.ema_classifier = nn.Sequential(nn.Linear(self.L*self.K, n_classes+1))
            self.freezer_layer(self.ema_feature)
            self.freezer_layer(self.ema_attention)
            self.freezer_layer(self.ema_classifier)
            
    
    def freezer_layer(self, module: nn.Module):
        module.eval()
        for p in module.parameters():
            p.requires_grad = False

    def set_up(self, lr, max_epochs, weight_decay, **args):
        total_iterations = args['total_iterations']
        self.drop_max = total_iterations//10
        params = filter(lambda p: p.requires_grad, self.parameters())
        self.opt_wsi = torch.optim.Adam(params, lr=lr,  weight_decay=weight_decay)
        self.dropout_scheduler = LinearScheduler(self, 0, 0.2, self.drop_max)
        self.mom_scheduler = np.linspace(0.994, 0.9999, total_iterations)
    
    def one_step(self, data, label, **args):
        # p = 1/(self.n_classes + 1)
        p = 0.2
        augmentation = random.random() < p
        # augmentation = False
        if augmentation:
            label = label - label + self.n_classes
            
        outputs = self.forward(data, augmentation=augmentation)
        logits = outputs['wsi_logits']
        loss = self.loss_fn(logits, label)
        
        self.opt_wsi.zero_grad()
        loss.backward()
        self.opt_wsi.step()
        if args['iteration'] < self.drop_max:
            self.dropout_scheduler.step()

        outputs['loss'] = loss

        if self.use_ema:
            if args['iteration'] % self.teacher_update_iter:
                m = self.mom_scheduler[args['iteration']]
                self.update_teacher(m)
        return outputs


    def forward(self, x, augmentation=False):
        feature = self.feature(x)
        if augmentation:
            noise = torch.normal(0, 0.1, feature.shape, device=x.device)
            feature = feature + noise
            
        A = self.attention(feature)
        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.mm(A, feature)  # KxL
        
        logits = self.classifier(M)
        wsi_logits, wsi_prob, wsi_label = self.task_adapter(logits)
        outputs = {
            'wsi_logits': wsi_logits,
            'wsi_prob': wsi_prob,
            'wsi_label': wsi_label,
        }
        return outputs
        
    def wsi_predict(self, x, **args):
        if self.use_ema:
            feature_fn = self.ema_feature
            attention_fn = self.ema_attention
            classifier_fn = self.ema_classifier
        else:
            feature_fn = self.feature
            attention_fn = self.attention
            classifier_fn = self.classifier
            
        feature = feature_fn(x)
        A = attention_fn(feature)
        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.mm(A, feature)  # KxL
        logits = classifier_fn(M)[:, :self.n_classes]
        
        wsi_logits, wsi_prob, wsi_label = self.task_adapter(logits)
        outputs = {
            'wsi_logits': wsi_logits,
            'wsi_prob': wsi_prob,
            'wsi_label': wsi_label,
        }
        return outputs


    def update_teacher(self, m):
        with torch.no_grad():
            for param_q, param_k in zip(self.feature.parameters(), self.ema_feature.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            for param_q, param_k in zip(self.attention.parameters(), self.ema_attention.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            for param_q, param_k in zip(self.classifier.parameters(), self.ema_classifier.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        
if __name__ == '__main__':

    mean_model = Simple(n_classes=2)
    x = torch.randn(100, 1024)
    y = mean_model(x)
    print(y)