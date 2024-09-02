import torch
import torch.nn.functional as F
from torch import Tensor, nn
from mamba_ssm.modules.mamba_simple import Mamba
import math
from .ffn import FeedForward


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output
    

class SwitchGate(nn.Module):
    """
    SwitchGate module for MoE (Mixture of Experts) model.

    Args:
        dim (int): Input dimension.
        num_experts (int): Number of experts.
        capacity_factor (float, optional): Capacity factor for sparsity. Defaults to 1.0.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(
        self,
        dim,
        num_experts: int,
        capacity_factor: float = 1.0,
        epsilon: float = 1e-6,
        *args,
        **kwargs,
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
        hidden_dim: int,
        output_dim: int,
        num_experts: int,
        capacity_factor: float = 1.0,
        mult: int = 4,
        use_aux_loss: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.mult = mult
        self.use_aux_loss = use_aux_loss

        self.experts = nn.ModuleList([FeedForward(dim, dim, mult, *args, **kwargs) for _ in range(num_experts)])

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
        moe_output = torch.sum(
            gate_scores.unsqueeze(-2) * stacked_expert_outputs, dim=-1
        )

        return moe_output, loss


class MambaBlock(nn.Module):
    """
    MoEMambaBlock is a module that combines MambaBlock and SwitchMoE layers.

    Args:
        dim (int): The input dimension.
        depth (int): The number of MambaBlock layers.
        d_state (int): The dimension of the state.
        causal (bool, optional): Whether to use causal attention. Defaults to True.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        shared_qk (bool, optional): Whether to share the query and key projections. Defaults to True.
        exact_window_size (bool, optional): Whether to use exact window size for attention. Defaults to False.
        heads (int, optional): The number of attention heads. Defaults to None.
        dim_head (int, optional): The dimension of each attention head. Defaults to None.
        m_expand (int, optional): The expansion factor for the hidden dimension. Defaults to 4.
        num_experts (int, optional): The number of experts in the SwitchMoE layer. Defaults to 4.
    """

    def __init__(
        self,
        dim,
        depth,
        d_state: int,
        m_expand: int = 4,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.d_state = d_state

        self.layers = nn.ModuleList([])
        self.norm = RMSNorm(dim)

        self.hidden_dim = dim * m_expand

        for _ in range(depth):
            self.layers.append(
                Mamba(
                    d_model=dim,
                    d_state=d_state,
                    *args,
                    **kwargs,
                )
            )


    def forward(self, x):
        """
        Forward pass of the MoEMambaBlock module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        for mamba in self.layers:
            x = mamba(x)
        return x


class MoEMambaBlock(nn.Module):
    """
    MoEMambaBlock is a module that combines MambaBlock and SwitchMoE layers.

    Args:
        dim (int): The input dimension.
        depth (int): The number of MambaBlock layers.
        d_state (int): The dimension of the state.
        causal (bool, optional): Whether to use causal attention. Defaults to True.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        shared_qk (bool, optional): Whether to share the query and key projections. Defaults to True.
        exact_window_size (bool, optional): Whether to use exact window size for attention. Defaults to False.
        heads (int, optional): The number of attention heads. Defaults to None.
        dim_head (int, optional): The dimension of each attention head. Defaults to None.
        m_expand (int, optional): The expansion factor for the hidden dimension. Defaults to 4.
        num_experts (int, optional): The number of experts in the SwitchMoE layer. Defaults to 4.
    """

    def __init__(
        self,
        dim,
        depth,
        d_state: int,
        ffn_mult = 0.5,
        m_expand: int = 4,
        num_experts: int = 4,
        use_aux_loss = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.d_state = d_state
        self.num_experts = num_experts

        self.layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        self.norm = RMSNorm(dim)

        self.hidden_dim = dim * m_expand

        for _ in range(depth):
            self.layers.append(
                Mamba(
                    d_model=dim,
                    d_state=d_state,
                    *args,
                    **kwargs,
                )
            )

            self.ffn_layers.append(
                SwitchMoE(
                    dim=dim,
                    hidden_dim=self.hidden_dim,
                    mult=ffn_mult,
                    output_dim=dim,
                    num_experts=num_experts,
                    use_aux_loss=use_aux_loss
                )
            )

    def forward(self, x):
        """
        Forward pass of the MoEMambaBlock module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        for mamba, moe in zip(self.layers, self.ffn_layers):
            x = mamba(x)
            x, _ = moe(x)
        return x
