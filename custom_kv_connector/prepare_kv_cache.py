#!/usr/bin/env python3
"""
Script to extract and save KV cache from a vLLM model.

This script runs inference on a prompt and saves the resulting KV cache
to a file that can be loaded by CustomFileKVConnector.

Usage:
    python prepare_kv_cache.py \
        --model <model_path> \
        --prompt "Your prompt text here" \
        --output kv_cache_dir/kv_cache.pt \
        --num-tokens 100
"""

import argparse
import torch
from pathlib import Path
from typing import List, Optional

from vllm import LLM, SamplingParams
from vllm.attention.backends.abstract import AttentionBackend


def extract_kv_cache_from_vllm(
    model_path: str,
    prompt: str,
    num_tokens_to_cache: Optional[int] = None,
    dtype: str = "auto",
    tensor_parallel_size: int = 1,
) -> tuple[List[int], torch.Tensor, torch.Tensor]:
    """
    Extract KV cache by running inference with vLLM.
    
    Args:
        model_path: Path to the model
        prompt: The prompt text to compute KV cache for
        num_tokens_to_cache: Number of tokens to cache (if None, cache all)
        dtype: Model dtype
        tensor_parallel_size: TP size
    
    Returns:
        (prompt_token_ids, keys, values)
        keys shape: (num_layers, 1, num_heads, seq_len, head_dim)
        values shape: (num_layers, 1, num_heads, seq_len, head_dim)
    """
    print(f"Loading model: {model_path}")
    
    # Initialize vLLM
    llm = LLM(
        model=model_path,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=True,  # Disable CUDA graphs for easier extraction
        max_model_len=4096,
    )
    
    # Tokenize prompt
    tokenizer = llm.get_tokenizer()
    prompt_token_ids = tokenizer.encode(prompt)
    
    print(f"Prompt: {prompt}")
    print(f"Prompt tokens: {len(prompt_token_ids)}")
    print(f"Token IDs: {prompt_token_ids[:20]}..." if len(prompt_token_ids) > 20 else f"Token IDs: {prompt_token_ids}")
    
    # Limit to num_tokens_to_cache if specified
    if num_tokens_to_cache and num_tokens_to_cache < len(prompt_token_ids):
        prompt_token_ids = prompt_token_ids[:num_tokens_to_cache]
        prompt_text = tokenizer.decode(prompt_token_ids)
        print(f"Truncated to {num_tokens_to_cache} tokens")
    else:
        prompt_text = prompt
    
    # Run prefill (just prompt, no generation)
    sampling_params = SamplingParams(
        max_tokens=1,  # Generate just 1 token to trigger prefill
        temperature=0.0,
    )
    
    print("Running prefill to compute KV cache...")
    outputs = llm.generate(prompt_text, sampling_params)
    
    # Extract KV cache from the model
    # Note: This is a simplified approach. For production use, you might need
    # to hook into vLLM's internal KV cache management system.
    
    print("\n⚠️  Note: Direct KV cache extraction from vLLM requires accessing")
    print("    internal state. For this demo, we'll simulate the KV cache structure.")
    print("    In production, you'd need to modify vLLM internals or use the")
    print("    approach shown in the alternative method below.\n")
    
    # Get model config
    model_config = llm.llm_engine.model_config
    num_layers = model_config.get_num_layers(llm.llm_engine.parallel_config)
    num_heads = model_config.get_num_attention_heads(llm.llm_engine.parallel_config)
    num_kv_heads = model_config.get_num_kv_heads(llm.llm_engine.parallel_config)
    head_dim = model_config.get_head_size()
    
    seq_len = len(prompt_token_ids)
    
    print(f"Model config:")
    print(f"  Layers: {num_layers}")
    print(f"  Attention heads: {num_heads}")
    print(f"  KV heads: {num_kv_heads}")
    print(f"  Head dimension: {head_dim}")
    print(f"  Sequence length: {seq_len}")
    
    # Create dummy KV cache structure for demonstration
    # In production, you'd extract the actual KV cache from vLLM
    print("\nGenerating simulated KV cache structure...")
    keys = torch.randn(num_layers, 1, num_kv_heads, seq_len, head_dim, dtype=torch.float16)
    values = torch.randn(num_layers, 1, num_kv_heads, seq_len, head_dim, dtype=torch.float16)
    
    return prompt_token_ids, keys, values


def extract_kv_cache_alternative(
    model_path: str,
    prompt: str,
    output_path: str,
    num_tokens: Optional[int] = None,
) -> None:
    """
    Alternative method: Use HuggingFace Transformers directly to extract real KV cache.
    
    This gives you actual KV values computed by the model.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"\n{'='*60}")
    print("Alternative Method: Using HuggingFace Transformers")
    print(f"{'='*60}\n")
    
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_token_ids = inputs['input_ids'][0].tolist()
    
    if num_tokens and num_tokens < len(prompt_token_ids):
        inputs['input_ids'] = inputs['input_ids'][:, :num_tokens]
        inputs['attention_mask'] = inputs['attention_mask'][:, :num_tokens]
        prompt_token_ids = prompt_token_ids[:num_tokens]
    
    print(f"Prompt tokens: {len(prompt_token_ids)}")
    
    # Forward pass with KV cache output
    print("Computing KV cache...")
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, return_dict=True)
    
    # Extract past_key_values (the KV cache)
    past_key_values = outputs.past_key_values  # List of tuples
    
    # Convert to our format
    num_layers = len(past_key_values)
    keys_list = []
    values_list = []
    
    for layer_idx, (key, value) in enumerate(past_key_values):
        # key/value shape: (batch, num_heads, seq_len, head_dim)
        # Convert to: (1, num_heads, seq_len, head_dim)
        keys_list.append(key.cpu())
        values_list.append(value.cpu())
    
    # Stack: (num_layers, batch, num_heads, seq_len, head_dim)
    keys = torch.stack(keys_list, dim=0)
    values = torch.stack(values_list, dim=0)
    
    print(f"\nExtracted KV cache:")
    print(f"  Keys shape: {keys.shape}")
    print(f"  Values shape: {values.shape}")
    
    # Save
    save_data = {
        'prompt_token_ids': prompt_token_ids,
        'keys': keys.half(),  # Convert to fp16 to save space
        'values': values.half(),
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(save_data, output_path)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    
    print(f"\n✓ Saved KV cache to: {output_path}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Tokens cached: {len(prompt_token_ids)}")
    print(f"  Layers: {num_layers}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract and save KV cache from a model"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt text to compute KV cache for"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="kv_cache_dir/kv_cache.pt",
        help="Output file path for KV cache"
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=None,
        help="Number of tokens to cache (default: all)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["transformers", "vllm"],
        default="transformers",
        help="Method to use for extraction (default: transformers)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Model dtype (for vLLM method)"
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallel size (for vLLM method)"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("KV Cache Extraction Tool")
    print(f"{'='*60}\n")
    
    if args.method == "transformers":
        extract_kv_cache_alternative(
            model_path=args.model,
            prompt=args.prompt,
            output_path=args.output,
            num_tokens=args.num_tokens,
        )
    else:
        # vLLM method (with simulated KV for demo)
        prompt_token_ids, keys, values = extract_kv_cache_from_vllm(
            model_path=args.model,
            prompt=args.prompt,
            num_tokens_to_cache=args.num_tokens,
            dtype=args.dtype,
            tensor_parallel_size=args.tp,
        )
        
        # Save
        save_data = {
            'prompt_token_ids': prompt_token_ids,
            'keys': keys,
            'values': values,
        }
        
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_data, output_path)
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\n✓ Saved to: {output_path}")
        print(f"  File size: {file_size_mb:.2f} MB")


if __name__ == "__main__":
    main()

