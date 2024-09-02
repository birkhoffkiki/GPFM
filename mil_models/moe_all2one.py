import torch
import random
import torch.nn.functional as F
from torch import Tensor, nn
torch.autograd.set_detect_anomaly(True)
from torch_geometric.nn import global_mean_pool, global_max_pool, GlobalAttention
from .mil import BaseMILModel


class WiKG(BaseMILModel):
    def __init__(self, dim_hidden=512, topk=6, n_classes=2, agg_type='bi-interaction', dropout=0.3,
                 pool='attn', task_type='subtyping'):
        super().__init__(task=task_type)

        
        self.W_head = nn.Linear(dim_hidden, dim_hidden)
        self.W_tail = nn.Linear(dim_hidden, dim_hidden)

        self.scale = dim_hidden ** -0.5
        self.topk = topk
        self.agg_type = agg_type

        self.gate_U = nn.Linear(dim_hidden, dim_hidden // 2)
        self.gate_V = nn.Linear(dim_hidden, dim_hidden // 2)
        self.gate_W = nn.Linear(dim_hidden // 2, dim_hidden)

        if self.agg_type == 'gcn':
            self.linear = nn.Linear(dim_hidden, dim_hidden)
        elif self.agg_type == 'sage':
            self.linear = nn.Linear(dim_hidden * 2, dim_hidden)
        elif self.agg_type == 'bi-interaction':
            self.linear1 = nn.Linear(dim_hidden, dim_hidden)
            self.linear2 = nn.Linear(dim_hidden, dim_hidden)
        else:
            raise NotImplementedError
        
        self.activation = nn.LeakyReLU()
        self.message_dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(dim_hidden)
        self.fc = nn.Linear(dim_hidden, n_classes)

        if pool == "mean":
            self.readout = global_mean_pool 
        elif pool == "max":
            self.readout = global_max_pool 
        elif pool == "attn":
            att_net=nn.Sequential(nn.Linear(dim_hidden, dim_hidden // 2), nn.LeakyReLU(), nn.Linear(dim_hidden//2, 1))     
            self.readout = GlobalAttention(att_net)


    def forward(self, x1, x2):
        # B, N, C = x.shape
        x1 = (x1 + x1.mean(dim=1, keepdim=True)) * 0.5  
        x2 = (x2 + x2.mean(dim=1, keepdim=True)) * 0.5  

        e_h = self.W_head(x1)
        e_t = self.W_tail(x2)

        # construct neighbour
        attn_logit = (e_h * self.scale) @ e_t.transpose(-2, -1)  # 1
        topk_weight, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)

        # add an extra dimension to the index tensor, making it available for advanced indexing, aligned with the dimensions of e_t
        topk_index = topk_index.to(torch.long)

        # expand topk_index dimensions to match e_t
        topk_index_expanded = topk_index.expand(e_t.size(0), -1, -1)  # shape: [1, 10000, 4]

        # create a RANGE tensor to help indexing
        batch_indices = torch.arange(e_t.size(0)).view(-1, 1, 1).to(topk_index.device)  # shape: [1, 1, 1]

        Nb_h = e_t[batch_indices, topk_index_expanded, :]  # shape: [1, 10000, 4, 512]

        # use SoftMax to obtain probability
        topk_prob = F.softmax(topk_weight, dim=2)
        eh_r = torch.mul(topk_prob.unsqueeze(-1), Nb_h) + torch.matmul((1 - topk_prob).unsqueeze(-1), e_h.unsqueeze(2))  # 1 pixel wise   2 matmul

        # gated knowledge attention
        e_h_expand = e_h.unsqueeze(2).expand(-1, -1, self.topk, -1)
        gate = torch.tanh(e_h_expand + eh_r)
        
        ka_weight = torch.einsum('ijkl,ijkm->ijk', Nb_h, gate)

        ka_prob = F.softmax(ka_weight, dim=2).unsqueeze(dim=2)
        e_Nh = torch.matmul(ka_prob, Nb_h).squeeze(dim=2)

        if self.agg_type == 'gcn':
            embedding = e_h + e_Nh
            embedding = self.activation(self.linear(embedding))
        elif self.agg_type == 'sage':
            embedding = torch.cat([e_h, e_Nh], dim=2)
            embedding = self.activation(self.linear(embedding))
        elif self.agg_type == 'bi-interaction':
            sum_embedding = self.activation(self.linear1(e_h + e_Nh))
            bi_embedding = self.activation(self.linear2(e_h * e_Nh))
            embedding = sum_embedding + bi_embedding

        
        h = self.message_dropout(embedding)

        h = self.readout(h.squeeze(0), batch=None)
        h = self.norm(h)
        h = self.fc(h)
        
        # adapter
        wsi_logits, wsi_prob, wsi_label = self.task_adapter(h)

        outputs = {
            'wsi_logits': wsi_logits,
            'wsi_prob': wsi_prob,
            'wsi_label': wsi_label,
        }
        return outputs
    



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
        # self.w_gate = nn.Linear(dim, num_experts)
        self.w_gate = nn.Sequential(
            nn.Linear(dim, dim//2), nn.GELU(),
            nn.Linear(dim//2, dim), nn.GELU(),
            nn.Linear(dim, num_experts)
            )

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
        self.linear_layers = nn.ModuleList([nn.Linear(dim, 1) for _ in range(num_experts)])


        self.gate = SwitchGate(dim, num_experts, capacity_factor)

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
        expert_predictions = [layer(i) for i, layer in zip(expert_outputs, self.linear_layers)]
        
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
        return moe_output, loss, expert_predictions


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


class ExpertLoss(nn.Module):
    def __init__(self, num_experts) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, expert_predictions: list, label: int):
        gt = F.one_hot(label, self.num_experts).float()
        expert_predictions = torch.cat(expert_predictions, dim=1)
        loss = self.bce(expert_predictions, gt)
        
        return loss


def data_mixer(data: list, low_ratio=0.1, high_ratio=1.0):
    result = []
    ratio = random.uniform(low_ratio, high_ratio)
    for d in data:
        perm = torch.randperm(len(d))
        num = int(ratio*len(d))
        result.append(d[perm][:num])
    return result
    
    
class MoE(BaseMILModel):
    def __init__(self, in_dim=1024, n_classes=1, task='subtyping',
                 samples_per_cls=None, num_experts=8, use_aux_loss=False):
        super(MoE, self).__init__(task=task)
        self.L = 512
        self.n_classes = n_classes
        self.expert_loss_ratio = 1.0
        self.samples_per_cls = samples_per_cls
        self.task_type = task
        self.use_data_mixer = True

        if not isinstance(in_dim, (list, tuple)):
            in_dim = [in_dim]
        
        self.feature = nn.ModuleList()
        for idm in in_dim:
            self.feature.append(nn.Sequential(nn.Linear(idm, self.L), nn.LeakyReLU()))
        
        self.kfuser = WiKG(dim_hidden=self.L, n_classes=n_classes, task_type=task)
        # self.swith_moe = SwitchMoE(self.L, num_experts=n_classes, use_aux_loss=use_aux_loss)
        # self.union_classifier = nn.Linear(self.L, n_classes)

        if task == 'subtyping':
            self.loss_fn = nn.CrossEntropyLoss()
        elif task == 'survival':
            from utils.survival_utils import NLLSurvLoss
            self.loss_fn = NLLSurvLoss(alpha=0.0)
        else:
            raise NotImplementedError
        
        # self.expert_cri = ExpertLoss(n_classes)
            
            
    def set_up(self, lr, max_epochs, weight_decay, **args):
        self.opt_wsi = torch.optim.Adam(self.parameters(), lr=lr,  weight_decay=weight_decay)
    
    def process_data(self, data, label, device):
        if not isinstance(data, (tuple, list)):
            data = [data]
        data = [i.to(device) for i in data]
        label = label.to(device)
        return data, label
    
    def one_step(self, data, label, **args):
        if not isinstance(data, (list, tuple)):
            x = [x]
        if self.use_data_mixer:
            data = data_mixer(data, low_ratio=0.9, high_ratio=1.0)
            
        outputs = self.forward(data)
        logits = outputs['wsi_logits']
    
        if self.task_type == 'subtyping':
            loss = self.loss_fn(logits, label)
        elif self.task_type == 'survival':
            hazards, S, _ = outputs['wsi_logits'], outputs['wsi_prob'], outputs['wsi_label']
            c = args['c']
            loss = self.loss_fn(hazards=hazards, S=S, Y=label, c=c)
        else:
            raise NotImplementedError
        
        # # expert ?
        # expert_predictions = outputs['expert_predictions']
        # expert_loss = self.expert_cri(expert_predictions, label) * 0.5
        
        # loss = loss + expert_loss
        
        self.opt_wsi.zero_grad()
        loss.backward()
        self.opt_wsi.step()
        outputs['loss'] = loss
        # outputs['expert_loss'] = expert_loss
        return outputs


    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            x = [x]

        features = []
        for fn, i in zip(self.feature, x):
            feature = fn(i)
            features.append(feature[None])
            
        # feat = torch.cat(features, dim=0)[None]
        # feat, aux_loss, expert_predictions = self.swith_moe(feat)
        # logits = self.union_classifier(feat)
        # wsi_logits, wsi_prob, wsi_label = self.task_adapter(logits)
        
        # outputs = {
        #     'wsi_logits': wsi_logits,
        #     'wsi_prob': wsi_prob,
        #     'wsi_label': wsi_label,
        #     'expert_predictions': expert_predictions, 
        # }
        outputs = self.kfuser(features[0], features[1])
        return outputs
        
    # def wsi_predict(self, x, **args):
    #     if not isinstance(x, (list, tuple)):
    #         x = [x]

    #     features = []
    #     for fn, i in zip(self.feature, x):
    #         feature = fn(i)
    #         features.append(feature)
            
    #     feat = torch.cat(features, dim=0)[None]
    #     feat, _ = self.swith_moe(feat)
    #     logits = self.union_classifier(feat)[:, :self.n_classes]
    #     wsi_logits, wsi_prob, wsi_label = self.task_adapter(logits)
        
    #     outputs = {
    #         'wsi_logits': wsi_logits,
    #         'wsi_prob': wsi_prob,
    #         'wsi_label': wsi_label,
    #         # 'exp_logits': exp_logits,
    #     }
    #     return outputs

