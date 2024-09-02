from torch import nn
import torch
import random
from torch.nn import functional as F

from .mamba_simple import MoEMambaBlock, MambaBlock
from ..mil import BaseMILModel
from ..loss import get_loss_function

class MoE(BaseMILModel):
    def __init__(self, in_dim=1024, n_classes=1, depth=4, experts=2, task='subtyping'):
        super(MoE, self).__init__(task=task)
        self.n_classes = n_classes
        self.expert_loss_ratio = 1.0
        self.task_type = task
        self.noise_std = 0.1
        self.augmentation_ratio = 0.0
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, in_dim))
        # self.backbone = MoEMambaBlock(in_dim, depth=depth, d_state=16, num_experts=experts, ffn_mult=1, use_aux_loss=True)
        self.backbone = MambaBlock(in_dim, depth=depth, d_state=16)
        self.max_pool = nn.AdaptiveAvgPool1d(1)
        
        self.union_classifier = nn.Linear(in_dim, n_classes)
        
        self.loss_fn = get_loss_function(task_type=task, loss_name='CrossEntropy')
        
            
    def set_up(self, lr, max_epochs, weight_decay, **args):
        params = filter(lambda p: p.requires_grad, self.parameters())
        self.opt_wsi = torch.optim.Adam(params, lr=lr,  weight_decay=weight_decay)
    
    # def process_data(self, data, label, device):
    #     data = [i.to(device) for i in data]
    #     label = label.to(device)
    #     return data, label
    
    def one_step(self, data, label, **args):
        augmentation = random.random() < self.augmentation_ratio if self.task_type == 'subtyping' else False 
        outputs = self.forward(data, augmentation=augmentation)
        logits = outputs['wsi_logits']
    
        if self.task_type == 'subtyping':
            # print(logits.shape, flush=True)
            # print(label.shape)
            loss = self.loss_fn(logits, label)
        elif self.task_type == 'survival':
            hazards, S, _ = outputs['wsi_logits'], outputs['wsi_prob'], outputs['wsi_label']
            c = args['c']
            loss = self.loss_fn(hazards=hazards, S=S, Y=label, c=c)
        else:
            raise NotImplementedError
    
        self.opt_wsi.zero_grad()
        loss.backward()
        self.opt_wsi.step()
        outputs['loss'] = loss
        return outputs

    def forward_features(self, x):
        # cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((x, cls_token), dim=1)
        hidden_states = self.backbone(x)
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.max_pool(hidden_states)[:, :, 0]
        return hidden_states


    def forward(self, x, augmentation=False):
        x = x[None, :, :]
        feat = self.forward_features(x)
        logits = self.union_classifier(feat)
        wsi_logits, wsi_prob, wsi_label = self.task_adapter(logits)
        outputs = {
            'wsi_logits': wsi_logits,
            'wsi_prob': wsi_prob,
            'wsi_label': wsi_label,
        }
        return outputs
        
    def wsi_predict(self, x, **args):
        x = x[None, :, :]

        feat = self.forward_features(x)
        logits = self.union_classifier(feat)
        wsi_logits, wsi_prob, wsi_label = self.task_adapter(logits)
        
        outputs = {
            'wsi_logits': wsi_logits,
            'wsi_prob': wsi_prob,
            'wsi_label': wsi_label,
        }
        return outputs
