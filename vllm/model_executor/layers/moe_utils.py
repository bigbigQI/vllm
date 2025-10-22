# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Data structures for MoE expert selection tracking."""

from dataclasses import dataclass, field


@dataclass
class TokenExpertSelection:
    """Expert selection information for a single token at a specific layer.
    
    Args:
        token_index: The global position of the token in the sequence.
        layer_name: The name of the MoE layer (e.g., "model.layers.5.mlp").
        selected_expert_ids: The IDs of the selected experts (top-k).
    """
    token_index: int
    layer_name: str
    selected_expert_ids: list[int]

    def to_dict(self) -> dict:
        """Convert to dictionary format for output."""
        return {
            "token_index": self.token_index,
            "layer_name": self.layer_name,
            "selected_experts": self.selected_expert_ids,
        }


@dataclass
class RequestExpertSelections:
    """Expert selection information for all tokens in a request.
    
    The selections are organized by token index, with each token containing
    a mapping from layer name to selected expert IDs.
    
    Args:
        request_id: The unique identifier for the request.
        token_selections: Dictionary mapping token index to a dictionary of
            layer name -> selected expert IDs.
            Example: {0: {"layer_5": [1, 3], "layer_10": [2, 4]}, 1: {...}}
    """
    request_id: str
    token_selections: dict[int, dict[str, list[int]]] = field(default_factory=dict)

    def add_token_layer_selection(
        self,
        token_index: int,
        layer_name: str,
        expert_ids: list[int],
    ) -> None:
        """Add expert selection for a token at a specific layer.
        
        Args:
            token_index: The global position of the token in the sequence.
            layer_name: The name of the MoE layer.
            expert_ids: The list of selected expert IDs.
        """
        if token_index not in self.token_selections:
            self.token_selections[token_index] = {}
        self.token_selections[token_index][layer_name] = expert_ids

    def to_dict(self) -> dict:
        """Convert to complete dictionary format for output.
        
        Returns:
            Dictionary with the following structure:
            {
                "request_id": "xxx",
                "num_tokens": 20,
                "selections": [
                    {
                        "token_index": 0,
                        "layers": {
                            "model.layers.5.mlp": [1, 3, 5],
                            "model.layers.10.mlp": [2, 4, 6]
                        }
                    },
                    ...
                ]
            }
        """
        selections = []
        for token_idx in sorted(self.token_selections.keys()):
            selections.append(
                {"token_index": token_idx, "layers": self.token_selections[token_idx]}
            )

        return {
            "request_id": self.request_id,
            "num_tokens": len(self.token_selections),
            "selections": selections,
        }


