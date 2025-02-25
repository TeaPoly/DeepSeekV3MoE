#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2024 Lucky Wong
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

"""
Modified from https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/blob/main/modeling_deepseek.py
             https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
"""

from typing import Literal, Tuple

import torch
import torch.distributed as dist


class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss,
    which includes the gradient of the aux loss during backpropagation.
    """

    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(
                1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


class DeepSeekV3MoEGate(torch.nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features.
        topk (int): Number of top experts activated for each input.
        n_groups (int): Number of groups for routing.
        topk_groups (int): Number of groups to route inputs to.
        score_func (str): Scoring function ('softmax' or 'sigmoid').
        route_scale (float): Scaling factor for routing weights.
        weight (torch.nn.Parameter): Learnable weights for the gate.
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
    """

    def __init__(
        self,
        n_routed_experts: int,
        n_activated_experts: int,
        n_expert_groups: int = 1,
        n_limited_groups: int = 1,
        score_func: Literal["softmax", "sigmoid"] = "sigmoid",
        route_scale: float = 1.0,
        use_bias: bool = True,
        bias_update_speed: float = 0.001,
        aux_loss_alpha: float = 0.001,
    ):
        """
        Initializes the DeepSeekV3Gate module.
        """
        super().__init__()
        self.n_routed_experts = n_routed_experts
        self.topk = n_activated_experts
        self.n_groups = n_expert_groups
        self.topk_groups = n_limited_groups
        self.score_func = score_func
        self.route_scale = route_scale
        self.aux_loss_alpha = aux_loss_alpha

        # Gating with auxiliary-loss-free balancing
        self.bias = (
            torch.nn.Parameter(torch.zeros(
                n_routed_experts), requires_grad=False)
            if use_bias
            else None
        )
        self.bias_update_speed = bias_update_speed

    def forward(self, linear: torch.nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        bsz, seq_len, h = x.shape
        # compute gating score
        x = x.view(-1, h)
        scores = linear(x)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores

        # Adjust top_indices to include all selected experts
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = torch.zeros_like(scores[..., 0]).scatter_(1, indices, True)
            scores = (scores * mask.unsqueeze(-1)).flatten(1)
        indices = torch.topk(scores, self.topk, dim=-1)[1]

        # Complementary Sequence-Wise Auxiliary Loss
        if self.training and self.aux_loss_alpha > 0.0:
            # Note that the bias term is only used for routing.
            scores_for_aux = original_scores  # scores
            aux_topk = self.topk
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = indices.view(bsz, -1)
            scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
            ce = torch.zeros(bsz, self.n_routed_experts, device=x.device)
            ce.scatter_add_(
                1,
                topk_idx_for_aux_loss,
                torch.ones(bsz, seq_len * aux_topk, device=x.device),
            ).div_(seq_len * aux_topk / self.n_routed_experts)
            aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                dim=1
            ).mean() * self.aux_loss_alpha
        else:
            aux_loss = None

        if self.training and self.bias is not None:
            with torch.no_grad():
                counts = torch.bincount(
                    indices.flatten(), minlength=self.n_routed_experts
                )
                self.update_bias(counts)

        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights.type_as(x), indices, aux_loss

    def update_bias(self, counts):
        # Sum counts across all processes using distributed communication
        dist.all_reduce(counts, dist.ReduceOp.SUM)
        avg_count = counts.float().mean()
        error = avg_count - counts.float()
        # Update the bias parameter using the sign of error scaled by bias_update_speed
        self.bias.add_(self.bias_update_speed * torch.sign(error))


class GatedMLP(torch.nn.Module):
    def __init__(
        self,
        idim: int,
        hidden_units: int,
        dropout_rate: float,
        activation: torch.nn.Module = torch.nn.SiLU,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.gate = torch.nn.Linear(idim, hidden_units, bias=False)
        self.activation = activation
        self.w_1 = torch.nn.Linear(idim, hidden_units, bias=bias)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.w_2 = torch.nn.Linear(hidden_units, idim, bias=bias)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(self.activation(self.gate(xs)) * self.w_1(xs)))


class DeepSeekV3MoELayer(torch.nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Args:
        n_expert: number of expert.
        n_expert_activated: The actual number of experts used for each frame
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (torch.nn.Module): Activation function
    """

    def __init__(
        self,
        idim: int,
        hidden_units: int,
        dropout_rate: float,
        n_shared_expert: int = 1,
        activation: torch.nn.Module = torch.nn.ReLU(),
        bias: bool = False,
        n_expert: int = 8,
        n_expert_activated: int = 2,
        n_expert_groups: int = 1,
        n_limited_groups: int = 1,
        score_func: Literal["softmax", "sigmoid"] = "sigmoid",
        route_scale: float = 1.0,
        auxiliary_loss_free: bool = True,
        aux_loss_alpha: float = 0.001,
    ):
        super(DeepSeekV3MoELayer, self).__init__()

        self.gate = torch.nn.Linear(idim, n_expert, bias=False)

        self.experts = torch.nn.ModuleList(
            GatedMLP(
                idim,
                hidden_units,
                dropout_rate,
                activation,
                bias=bias,
            )
            for _ in range(n_expert)
        )
        self.shared_experts = None
        if n_shared_expert > 0:
            self.shared_experts = GatedMLP(
                idim,
                n_shared_expert * hidden_units,
                dropout_rate,
                activation,
                bias=bias,
            )
        self.n_routed_experts = n_expert
        self.gate_control = DeepSeekV3MoEGate(
            n_expert,
            n_expert_activated,
            n_expert_groups=n_expert_groups,
            n_limited_groups=n_limited_groups,
            score_func=score_func,
            route_scale=route_scale,
            use_bias=auxiliary_loss_free,
            aux_loss_alpha=aux_loss_alpha,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        B, L, D = x.size()

        weights, indices, aux_loss = self.gate_control(self.gate, x)
        x = x.view(-1, D)

        y = torch.zeros_like(x)
        expert_counts = torch.bincount(
            indices.flatten(), minlength=self.n_routed_experts
        )
        counts = expert_counts.tolist()
        for i, expert in enumerate(self.experts):
            if counts[i] == 0:
                continue
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        if self.training and aux_loss is not None:
            y = AddAuxiliaryLoss.apply(y, aux_loss)

        if self.shared_experts is not None:
            z = self.shared_experts(x)
            y = (z + y).view(B, L, D)
        else:
            y = y.view(B, L, D)
        return y
