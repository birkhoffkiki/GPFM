import torch
import torch.nn as nn
import torch.nn.functional as F
from .mil import BaseMILModel


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class MeanMIL(BaseMILModel):
    def __init__(self, in_dim=1024, n_classes=1, dropout=True, act='relu', task_type='subtyping'):
        super(MeanMIL, self).__init__(task=task_type)

        head = [nn.Linear(in_dim,512)]

        if act.lower() == 'relu':
            head += [nn.ReLU()]
        elif act.lower() == 'gelu':
            head += [nn.GELU()]

        if dropout:
            head += [nn.Dropout(0.25)]
            
        head += [nn.Linear(512,n_classes)]
        
        self.head = nn.Sequential(*head)
        self.apply(initialize_weights)

    def forward(self, x, **kwargs):
        if len(x.shape) == 3 and x.shape[0] > 1:
            raise RuntimeError('Batch size must be 1, current batch size is:{}'.format(x.shape[0]))
        if len(x.shape) == 3 and x.shape[0] == 1:
            x = x[0]
        logits = self.head(x)
        logits = torch.mean(logits, dim=0, keepdim=True)
        
        wsi_logits, wsi_prob, wsi_label = self.task_adapter(logits)

        outputs = {
            'wsi_logits': wsi_logits,
            'wsi_prob': wsi_prob,
            'wsi_label': wsi_label,
        }
        return outputs
    
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.head = self.head.to(device)



class MaxMIL(BaseMILModel):
    def __init__(self, in_dim=1024, n_classes=1,dropout=True,act='relu', task_type='subtyping'):
        super(MaxMIL, self).__init__(task=task_type)

        head = [nn.Linear(in_dim,512)]

        if act.lower() == 'relu':
            head += [nn.ReLU()]
        elif act.lower() == 'gelu':
            head += [nn.GELU()]

        if dropout:
            head += [nn.Dropout(0.25)]
        head += [nn.Linear(512,n_classes)]
        self.head = nn.Sequential(*head)
        self.apply(initialize_weights)
        
    def forward(self, x, **kwargs):
        if len(x.shape) == 3 and x.shape[0] > 1:
            raise RuntimeError('Batch size must be 1, current batch size is:{}'.format(x.shape[0]))
        if len(x.shape) == 3 and x.shape[0] == 1:
            x = x[0]
        
        logits = self.head(x)
        logits, _ = torch.max(logits, dim=0, keepdim=True)
        
        wsi_logits, wsi_prob, wsi_label = self.task_adapter(logits)

        outputs = {
            'wsi_logits': wsi_logits,
            'wsi_prob': wsi_prob,
            'wsi_label': wsi_label,
        }
        return outputs     
    
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.head = self.head.to(device)


if __name__ == '__main__':
    mean_model = MeanMIL(n_classes=2)
    x = torch.randn(100, 1024)
    y = mean_model(x)
    print(y)