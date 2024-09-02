# https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py
# https://arxiv.org/pdf/1802.04712.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mil import BaseMILModel

class DAttention(BaseMILModel):
    def __init__(self, in_dim, n_classes, dropout, act, task_type='subtyping'):
        super(DAttention, self).__init__(task=task_type)
        self.L = 512
        self.D = 128
        self.K = 1
        self.feature = [nn.Linear(in_dim, 512)]
        
        if act.lower() == 'gelu':
            self.feature += [nn.GELU()]
        else:
            self.feature += [nn.ReLU()]

        if dropout:
            self.feature += [nn.Dropout(0.25)]

        self.feature = nn.Sequential(*self.feature)

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, n_classes),
        )

    def forward(self, x, **kwargs):
        feature = self.feature(x)
        feature = feature.squeeze()
        A = self.attention(feature)
        A = torch.transpose(A, -1, -2)  # KxN
        A_raw = A
        # if return_attention is in kwargs, return the attention weights
        if 'attention_only' in kwargs:
            return A
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.mm(A, feature)  # KxL
        
        logits = self.classifier(M)
        
        wsi_logits, wsi_prob, wsi_label = self.task_adapter(logits)

        outputs = {
                'wsi_logits': wsi_logits,
                'wsi_prob': wsi_prob,
                'wsi_label': wsi_label,
                'attention': A_raw,
            }
            
        return outputs
    
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature = self.feature.to(device)
        self.attention = self.attention.to(device)
        self.classifier = self.classifier.to(device)