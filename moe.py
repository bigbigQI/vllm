#!/usr/bin/env python3
"""
Example script demonstrating MoE Expert Tracking in vLLM.

This script shows how to enable and use the MoE expert tracking feature
to see which experts are selected during inference for Mixture of Experts models.

Usage:
    # Enable MoE expert tracking via environment variable
    export VLLM_ENABLE_MOE_EXPERT_TRACKING=1
    
    # Run the script
    python examples/moe_expert_tracking_example.py

Author: vLLM Team
Date: 2025
"""

import os
from vllm import LLM, SamplingParams

def main():
    # VLLM_ENABLE_MOE_EXPERT_TRACKING
    # Enable MoE expert tracking
    os.environ["VLLM_ENABLE_MOE_EXPERT_TRACKING"] = "1"
    
    # Initialize the model - use a MoE model like Mixtral
    # Note: Replace with an actual MoE model you have access to
    model_name = "/apps/models/Qwen3-30B-A3B-Instruct-2507"  # Example MoE model

    # # Load tokenizer to decode specific tokens
    # from transformers import AutoTokenizer
    
    # print(f"Loading tokenizer for model: {model_name}")
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # # Token IDs to decode
    # token_ids = [22555, 0, 6771, 594, 1438]
    
    # print("\nDecoding specified token IDs:")
    # print("-" * 40)
    # for token_id in token_ids:
    #     try:
    #         token_text = tokenizer.decode([token_id])
    #         print(f"Token ID {token_id}: '{token_text}'")
    #     except Exception as e:
    #         print(f"Token ID {token_id}: Error decoding - {e}")
    # print("-" * 40)
    # print()

    # exit()
    
    print(f"Loading MoE model: {model_name}")
    print("Note: MoE expert tracking is ENABLED")
    print()
    
    # Create LLM instance
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,  # Adjust based on your hardware
        max_model_len=512,  # Shorter for demo purposes
        enforce_eager=True,
    )
    
    # Define prompts
    prompts = [
        "Explain the concept of quantum computing in simple terms.",
    ]

    # Set sampling parameters: decode only one token
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=5,
    )

    print("=" * 80)
    print("Running inference (decoding only one token)...")
    print("=" * 80)
    print()

    # Generate responses
    outputs = llm.generate(prompts, sampling_params)
    print("generate done")
    print()

    # Print prompt and generated token count
    for i, output in enumerate(outputs):
        print(f"Prompt {i+1}: {output.prompt}")
        print(f"Prompt tokens: {len(output.prompt_token_ids)}")
        print(f"Generated tokens: {len(output.outputs[0].token_ids)}")
        print(f"Generated tokens: {output.outputs[0].token_ids}")
        print(f"Generated text: {output.outputs[0].text}")
        print()
        
        # Print MoE expert selections for the first layer
        if hasattr(output, 'moe_expert_selections') and output.moe_expert_selections:
            print("=" * 80)
            print("MoE Expert Selections (First Layer Only)")
            print("=" * 80)
            
            # Get the first layer name (model.layers.0.mlp.experts)
            first_layer_name = "model.layers.0.mlp.experts"
            
            # Print expert selections for each token
            total_tokens = len(output.prompt_token_ids) + len(output.outputs[0].token_ids)
            print(f"Total tokens: {total_tokens}")
            print()
            
            for token_idx in range(total_tokens):
                if token_idx in output.moe_expert_selections:
                    token_experts = output.moe_expert_selections[token_idx]
                    if first_layer_name in token_experts:
                        expert_ids = token_experts[first_layer_name]
                        print(f"Token {token_idx}: Experts {expert_ids}")
            
            print("=" * 80)
        else:
            print("Warning: MoE expert selections not available!")
            print("Make sure VLLM_ENABLE_MOE_EXPERT_TRACKING=1 is set.")
        print()


    # # Process and display results
    # for i, output in enumerate(outputs):
    #     print(f"\n{'='*80}")
    #     print(f"Prompt {i+1}: {output.prompt}")
    #     print(f"{'='*80}")
        
    #     # Display generated text
    #     generated_text = output.outputs[0].text
    #     print(f"\nGenerated Text:\n{generated_text}")
        
    #     # Display MoE expert selection information (if available)
    #     if hasattr(output, 'moe_expert_selections') and output.moe_expert_selections:
    #         print(f"\n{'-'*80}")
    #         print("MoE Expert Selection Information:")
    #         print(f"{'-'*80}")
            
    #         for layer_info in output.moe_expert_selections:
    #             layer_name = layer_info['layer_name']
    #             topk_ids = layer_info['topk_ids']
    #             topk_weights = layer_info['topk_weights']
    #             num_experts = layer_info['num_experts']
                
    #             print(f"\nLayer: {layer_name}")
    #             print(f"  Total Experts: {num_experts}")
    #             print(f"  Number of Tokens: {len(topk_ids)}")
                
    #             # Show expert selection for first few tokens
    #             num_tokens_to_show = min(5, len(topk_ids))
    #             print(f"  Expert selections (first {num_tokens_to_show} tokens):")
                
    #             for token_idx in range(num_tokens_to_show):
    #                 experts = topk_ids[token_idx]
    #                 weights = topk_weights[token_idx]
    #                 print(f"    Token {token_idx}: Experts {experts} with weights {weights}")
                
    #             # Calculate expert usage statistics
    #             from collections import Counter
    #             expert_usage = Counter()
    #             for token_experts in topk_ids:
    #                 for expert_id in token_experts:
    #                     expert_usage[expert_id] += 1
                
    #             print(f"  Expert usage distribution:")
    #             for expert_id in sorted(expert_usage.keys()):
    #                 count = expert_usage[expert_id]
    #                 percentage = (count / (len(topk_ids) * len(topk_ids[0]))) * 100
    #                 print(f"    Expert {expert_id}: {count} times ({percentage:.1f}%)")
    #     else:
    #         print("\nNote: MoE expert selection information is not available.")
    #         print("Make sure VLLM_ENABLE_MOE_EXPERT_TRACKING=1 is set before creating the LLM instance.")
    
    # print(f"\n{'='*80}")
    # print("Inference complete!")
    # print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

