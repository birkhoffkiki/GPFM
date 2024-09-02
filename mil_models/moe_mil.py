import torch
import random
import numpy as np
import torch.nn.functional as F
from torch import Tensor, nn

from .mil import BaseMILModel


class Expert(nn.Module):
    def __init__(self, in_dim):
        super(Expert, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1
        self.feature = nn.Sequential(*[nn.Linear(in_dim, self.L), nn.GELU(), nn.Dropout(0.25)])

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K))

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.squeeze()
        A = self.attention(feature)
        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.mm(A, feature)  # KxL
        return M
    
    
class SwitchGate(nn.Module):
    """
    SwitchGate module for MoE (Mixture of Experts) model.
    """

    def __init__(
        self,
        dim,
        num_experts: int,
        capacity_factor: float = 1.0,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.epsilon = epsilon
        self.w_gate = nn.Linear(dim, num_experts)

    def forward(self, x: Tensor, use_aux_loss=False):
        """
        Forward pass of the SwitchGate module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Gate scores.
        """
        # Compute gate scores
        gate_scores = F.softmax(self.w_gate(x), dim=-1)

        # Determine the top-1 expert for each token
        capacity = int(self.capacity_factor * x.size(0))

        top_k_scores, top_k_indices = gate_scores.topk(1, dim=-1)

        # Mask to enforce sparsity
        mask = torch.zeros_like(gate_scores).scatter_(
            1, top_k_indices, 1
        )

        # Combine gating scores with the mask
        masked_gate_scores = gate_scores * mask

        # Denominators
        denominators = (
            masked_gate_scores.sum(0, keepdim=True) + self.epsilon
        )

        # Norm gate scores to sum to the capacity
        gate_scores = (masked_gate_scores / denominators) * capacity

        if use_aux_loss:
            load = gate_scores.sum(0)  # Sum over all examples
            importance = gate_scores.sum(1)  # Sum over all experts

            # Aux loss is mean suqared difference between load and importance
            loss = ((load - importance) ** 2).mean()

            return gate_scores, loss

        return gate_scores, None


class SwitchMoE(nn.Module):
    """
    A module that implements the Switched Mixture of Experts (MoE) architecture.

    Args:
        dim (int): The input dimension.
        hidden_dim (int): The hidden dimension of the feedforward network.
        output_dim (int): The output dimension.
        num_experts (int): The number of experts in the MoE.
        capacity_factor (float, optional): The capacity factor that controls the capacity of the MoE. Defaults to 1.0.
        mult (int, optional): The multiplier for the hidden dimension of the feedforward network. Defaults to 4.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        dim (int): The input dimension.
        hidden_dim (int): The hidden dimension of the feedforward network.
        output_dim (int): The output dimension.
        num_experts (int): The number of experts in the MoE.
        capacity_factor (float): The capacity factor that controls the capacity of the MoE.
        mult (int): The multiplier for the hidden dimension of the feedforward network.
        experts (nn.ModuleList): The list of feedforward networks representing the experts.
        gate (SwitchGate): The switch gate module.

    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        capacity_factor: float = 1.0,
        use_aux_loss: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.use_aux_loss = use_aux_loss
        
        self.experts = nn.ModuleList([Expert(dim) for _ in range(num_experts)])

        self.gate = SwitchGate(
            dim,
            num_experts,
            capacity_factor,
        )

    def forward(self, x: Tensor):
        """
        Forward pass of the SwitchMoE module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor of the MoE.

        """
        # (batch_size, seq_len, num_experts)
        gate_scores, loss = self.gate(
            x, use_aux_loss=self.use_aux_loss
        )

        # Dispatch to experts
        expert_outputs = [expert(x) for expert in self.experts]

        # Check if any gate scores are nan and handle
        if torch.isnan(gate_scores).any():
            print("NaN in gate scores")
            gate_scores[torch.isnan(gate_scores)] = 0

        # Stack and weight outputs
        stacked_expert_outputs = torch.stack(
            expert_outputs, dim=-1
        )  # (batch_size, seq_len, output_dim, num_experts)
        
        if torch.isnan(stacked_expert_outputs).any():
            stacked_expert_outputs[
                torch.isnan(stacked_expert_outputs)
            ] = 0

        # Combine expert outputs and gating scores
        gate_scores = torch.mean(gate_scores, dim=1, keepdim=True)
        gate_scores = torch.softmax(gate_scores, dim=-1)
        
        moe_output = torch.sum(
            gate_scores * stacked_expert_outputs, dim=-1
        )
        return moe_output, loss


class softCrossEntropy(nn.Module):
    def __init__(self):
        super(softCrossEntropy, self).__init__()
        return

    def forward(self, inputs, target):
        """
        :param inputs: predictions
        :param target: target labels
        :return: loss
        """
        log_likelihood = - F.log_softmax(inputs, dim=1)
        sample_num, class_num = target.shape
        loss = torch.sum(torch.mul(log_likelihood, target))/sample_num
        return loss


# MoE archive
class MoE(BaseMILModel):
    def __init__(self, in_dim=1024, n_classes=1, task='subtyping',
                 samples_per_cls=None, num_experts=8, use_aux_loss=True):
        super(MoE, self).__init__(task=task)
        self.L = 512
        self.n_classes = n_classes
        self.expert_loss_ratio = 1.0
        self.samples_per_cls = samples_per_cls
        self.task_type = task

        if not isinstance(in_dim, (list, tuple)):
            in_dim = [in_dim]
        
        self.feature = nn.ModuleList()
        for idm in in_dim:
            self.feature.append(nn.Sequential(nn.Linear(idm, self.L), nn.GELU(), nn.Dropout(0.25)))
        
        self.swith_moe = SwitchMoE(self.L, num_experts=num_experts, use_aux_loss=use_aux_loss)
        self.union_classifier = nn.Linear(self.L, n_classes+1)

        if task == 'subtyping':
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        elif task == 'survival':
            from utils.survival_utils import NLLSurvLoss
            self.loss_fn = NLLSurvLoss(alpha=0.0)
        else:
            raise NotImplementedError
            
            
    def set_up(self, lr, max_epochs, weight_decay, **args):
        params = filter(lambda p: p.requires_grad, self.parameters())
        self.opt_wsi = torch.optim.Adam(params, lr=lr,  weight_decay=weight_decay)
    
    def process_data(self, data, label, device):
        if not isinstance(data, (tuple, list)):
            data = [data]
        data = [i.to(device) for i in data]
        label = label.to(device)
        return data, label
    
    def one_step(self, data, label, **args):
                
        if len(self.samples_per_cls) == 2:
            p = 0.1
        else:
            p = 1/(self.n_classes + 1)
        
        augmentation = random.random() < p if self.task_type == 'subtyping' else False
        # augmentation = False
        if augmentation:
            label = label - label + self.n_classes
            
        outputs = self.forward(data, augmentation=augmentation)
        logits = outputs['wsi_logits']
    
        if self.task_type == 'subtyping':
            loss = self.loss_fn(logits, label)
        elif self.task_type == 'survival':
            hazards, S, _ = outputs['wsi_logits'], outputs['wsi_prob'], outputs['wsi_label']
            c = args['c']
            loss = self.loss_fn(hazards=hazards, S=S, Y=label, c=c)
        else:
            raise NotImplementedError
        loss = loss + outputs['aux_loss']*0.5
        
        self.opt_wsi.zero_grad()
        loss.backward()
        self.opt_wsi.step()
        outputs['loss'] = loss
        return outputs


    def forward(self, x, augmentation=False):
        if not isinstance(x, (list, tuple)):
            x = [x]

        features = []
        for fn, i in zip(self.feature, x):
            feature = fn(i)
            features.append(feature)
            
        feat = torch.cat(features, dim=0)[None]
        feat, aux_loss = self.swith_moe(feat)
        logits = self.union_classifier(feat)
        wsi_logits, wsi_prob, wsi_label = self.task_adapter(logits)
        
        outputs = {
            'wsi_logits': wsi_logits,
            'wsi_prob': wsi_prob,
            'wsi_label': wsi_label,
            'aux_loss': aux_loss,
        }
        return outputs
        
    def wsi_predict(self, x, **args):
        if not isinstance(x, (list, tuple)):
            x = [x]

        features = []
        for fn, i in zip(self.feature, x):
            feature = fn(i)
            features.append(feature)
            
        feat = torch.cat(features, dim=0)[None]
        feat, _ = self.swith_moe(feat)
        logits = self.union_classifier(feat)[:, :self.n_classes]
        wsi_logits, wsi_prob, wsi_label = self.task_adapter(logits)
        
        outputs = {
            'wsi_logits': wsi_logits,
            'wsi_prob': wsi_prob,
            'wsi_label': wsi_label,
            # 'exp_logits': exp_logits,
        }
        return outputs




# class MoE(BaseMILModel):
#     def __init__(self, in_dim=1024, n_classes=1, act='relu', task='subtyping',
#                  samples_per_cls=None, noise_std=0.1):
#         super(MoE, self).__init__(task=task)
#         self.L = 512
#         self.D = 128
#         self.K = 1
#         self.n_classes = n_classes
#         self.noise_std = noise_std
#         self.expert_loss_ratio = 1.0
#         self.samples_per_cls = samples_per_cls
#         self.mix_up = False
#         self.cut_flag = True
#         self.task_type = task
        
#         assert len(in_dim) == 2, 'only support 2 experts currently.'
        
#         act_fun = nn.GELU if act == 'gelu' else nn.ReLU
#         dropout_p = 0.2
#         dropout_fn = nn.Dropout

#         if not isinstance(in_dim, (list, tuple)):
#             raise ValueError('The in_dim should be a list')

#         self.feature = nn.ModuleList()
#         for idm in in_dim:
#             self.feature.append(nn.Sequential(nn.Linear(idm, self.L),
#                                 act_fun(), dropout_fn(dropout_p)))
#         self.drop_out_layer = nn.Dropout(0.2)
#         self.attention = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh(), nn.Linear(self.D, self.K))
#         self.union_classifier = nn.Linear(self.L, n_classes+1)
#         if task == 'subtyping':
#             self.loss_fn = nn.CrossEntropyLoss()
#         elif task == 'survival':
#             from utils.survival_utils import NLLSurvLoss
#             self.loss_fn = NLLSurvLoss(alpha=0.0)
        
#         self.memory_bank = {i:[] for i in range(n_classes)}
            
#     def set_up(self, lr, max_epochs, weight_decay, **args):
#         params = filter(lambda p: p.requires_grad, self.parameters())
#         self.opt_wsi = torch.optim.Adam(params, lr=lr*0.25,  weight_decay=weight_decay)
    
#     def process_data(self, data, label, device):
#         data = [i.to(device) for i in data]
#         label = label.to(device)
#         return data, label
    
#     def mix_two_tensor(self, ori_data, target_data, ratio = 0.2):
#         if not isinstance(ori_data, (list, tuple)):
#             ori_data, target_data = [ori_data], [target_data]
#         mix_data = []
#         for ori, target in zip(ori_data, target_data):
#             length = target.shape[0]
#             cut_num = random.randint(0, int(length*ratio))
#             perm = torch.randperm(length)
#             cut_target = target[perm][:cut_num, :]
            
#             leng_ori = ori.shape[0]
#             left_num = leng_ori - random.randint(0, int(length*ratio))
#             perm = torch.randperm(leng_ori)
#             left_ori = ori[perm][:left_num, :]
#             data = torch.cat([left_ori, cut_target], dim=0)
#             mix_data.append(data)
#         return mix_data

#     def cut_pair(self, data):
#         d1, d2 = data
#         perm = torch.randperm(d1.shape[0])
#         ratio = random.randint(1, 9)/10
#         half = int(d1.shape[0]*ratio)
#         d1 = d1[perm][:half, :]
#         d2 = d2[perm][half:, :]
#         return [d1, d2]
    
    
#     def one_step(self, data, label, **args):
#         cls = label.item()
#         if self.mix_up and len(self.memory_bank[cls]) < 40:
#             self.memory_bank[cls].append([i.clone().cpu() for i in data]+[label])
            
#         if self.mix_up and random.random() < 0.5:
#             cls = random.randint(0, self.n_classes-1)
#             ratio = 0.2 if cls == label.item() else 0.02
            
#             random.shuffle(self.memory_bank[cls])
#             if len(self.memory_bank[cls]) > 0:
#                 target = [i.cuda() for i in self.memory_bank[cls].pop()]
#                 data = self.mix_two_tensor(data, target, ratio=ratio)
#         if self.cut_flag:
#             data = self.cut_pair(data)
                
#         if len(self.samples_per_cls) == 2:
#             p = 0.1
#         else:
#             p = 1/(self.n_classes + 1)
        
#         augmentation = random.random() < p
#         # augmentation = False
#         if augmentation:
#             label = label - label + self.n_classes
            
#         outputs = self.forward(data, augmentation=augmentation)
#         logits = outputs['wsi_logits']
        
#         if self.task_type == 'subtyping':
#             loss = self.loss_fn(logits, label)
#         elif self.task_type == 'survival':
#             hazards, S, _ = outputs['wsi_logits'], outputs['wsi_prob'], outputs['wsi_label']
#             c = args['c']
#             loss = self.loss_fn(hazards=hazards, S=S, Y=label, c=c)
#         else:
#             raise NotImplementedError
        
#         self.opt_wsi.zero_grad()
#         loss.backward()
#         self.opt_wsi.step()
#         outputs['loss'] = loss
#         return outputs


#     def forward(self, x, augmentation=False):
#         # feat fn
#         features = []
#         for fn, i in zip(self.feature, x):
#             feature = fn(i)
#             if augmentation:
#                 noise = torch.normal(0, self.noise_std, feature.shape, device=i.device)
#                 feature = feature + noise
            
#             features.append(feature)
            
#         UM = torch.cat(features, dim=0)

#         UA = self.attention(UM)
#         UA = torch.transpose(UA, -1, -2)  # KxN
#         UA = F.softmax(UA, dim=-1)  # softmax over N
#         UM = torch.mm(UA, UM)  # KxL
        
#         logits = self.union_classifier(UM)
#         wsi_logits, wsi_prob, wsi_label = self.task_adapter(logits)
        
#         outputs = {
#             'wsi_logits': wsi_logits,
#             'wsi_prob': wsi_prob,
#             'wsi_label': wsi_label,
#             'features': features,
#         }
#         return outputs
        
#     def wsi_predict(self, x, **args):
#         # feat fn
#         features = []
#         for fn, i in zip(self.feature, x):
#             feature = fn(i)            
#             features.append(feature)
            
#         # cal attention
#         UM = torch.cat(features, dim=0)        
#         UA = self.attention(UM)
#         UA = torch.transpose(UA, -1, -2)  # KxN
#         UA = F.softmax(UA, dim=-1)  # softmax over N
#         UM = torch.mm(UA, UM)  # KxL

#         logits = self.union_classifier(UM)[:, :self.n_classes]
#         wsi_logits, wsi_prob, wsi_label = self.task_adapter(logits)
        
#         outputs = {
#             'wsi_logits': wsi_logits,
#             'wsi_prob': wsi_prob,
#             'wsi_label': wsi_label,
#             # 'exp_logits': exp_logits,
#         }
#         return outputs


