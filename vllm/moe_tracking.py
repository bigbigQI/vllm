# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MoE Expert Selection Tracking

This module provides functionality to track which experts are selected
during MoE (Mixture of Experts) model inference. This is useful for
analyzing model behavior, debugging, and visualization.
"""

from dataclasses import dataclass, field

import torch


@dataclass
class ExpertSelectionInfo:
    """Information about expert selection for a single MoE layer.
    
    Args:
        layer_name: Name of the MoE layer (e.g., "model.layers.0.block_sparse_moe")
        topk_ids: Expert IDs selected for each token [num_tokens, top_k]
    """
    
    layer_name: str
    topk_ids: torch.Tensor  # [num_tokens, top_k] 
    
    def to_cpu(self) -> "ExpertSelectionInfo":
        """Move tensors to CPU for serialization."""
        return ExpertSelectionInfo(
            layer_name=self.layer_name,
            topk_ids=self.topk_ids.cpu(),
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for easier serialization."""
        return {
            "layer_name": self.layer_name,
            "topk_ids": self.topk_ids.tolist(),
        }
