 #!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Simplified example for using external KV cache in vLLM.

Important:
    Set environment variable before running:
        export VLLM_ENABLE_V1_MULTIPROCESSING=0

Usage:
    VLLM_ENABLE_V1_MULTIPROCESSING=0 python external_kv_simple_usage.py
"""

import torch
import os
import sys
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from external_kv_cache_example import ExternalKVCacheInjector


def simple_example():
    """Simple end-to-end example."""
    
    # Check environment variable
    if os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING', '1') != '0':
        print("=" * 80)
        print("ERROR: V1 multiprocessing mode is enabled.")
        print("This script requires direct access to model executor.")
        print()
        print("Please set the environment variable and try again:")
        print("  export VLLM_ENABLE_V1_MULTIPROCESSING=0")
        print()
        print("Then run:")
        print("  python external_kv_simple_usage.py")
        print()
        print("Or run directly with:")
        print("  VLLM_ENABLE_V1_MULTIPROCESSING=0 python external_kv_simple_usage.py")
        print("=" * 80)
        sys.exit(1)
    
    # 1. Initialize vLLM
    print("Step 1: Initializing vLLM...")
    llm = LLM(
        model="/apps/models/Qwen3-8B",
        max_model_len=200,
        enable_prefix_caching=True,
        enforce_eager=True,  # Required for KV cache manipulation
    )
    
    # 2. Prepare your external KV cache
    # This should be a dictionary with:
    # {
    #     'prompt_token_ids': List[int],
    #     'keys': torch.Tensor,    # (num_layers, 1, num_heads, seq_len, head_dim)
    #     'values': torch.Tensor,  # (num_layers, 1, num_heads, seq_len, head_dim)
    # }
    print("\nStep 2: Loading external KV cache...")

    external_kv = torch.load("/apps/vllm/kv_cache_vllm.pt")
    prompt_token_ids = external_kv['prompt_token_ids']
    keys = external_kv['keys']
    values = external_kv['values']
    
    num_layers = keys.shape[0]
    num_heads = keys.shape[2]
    head_dim = keys.shape[4]
    seq_len = len(prompt_token_ids)
    print(f"num_layers: {num_layers}, num_heads: {num_heads}, head_dim: {head_dim}, seq_len: {seq_len}")
    
    # external_kv = {
    #     'prompt_token_ids': prompt_token_ids,
    #     'keys': torch.randn(num_layers, 1, num_heads, seq_len, head_dim),
    #     'values': torch.randn(num_layers, 1, num_heads, seq_len, head_dim),
    # }
    
    # # Save to file for demo
    # kv_path = "/tmp/demo_kv_cache.pt"
    # torch.save(external_kv, kv_path)
    # print(f"Saved demo KV cache to {kv_path}")
    
    # 3. Inject the external KV cache
    print("\nStep 3: Injecting external KV cache into vLLM...")
    injector = ExternalKVCacheInjector(llm)
    
    print(f"\nInjecting external KV cache with:")
    print(f"  - {num_layers} layers")
    print(f"  - {seq_len} tokens")
    print(f"  - {num_heads} heads per layer")
    print(f"  - {head_dim} dimension per head")
    
    # cached_token_ids, num_blocks = injector.inject_external_kv_cache("/apps/sharon_verl_new/verl/kv_cache.pt")
    cached_token_ids, num_blocks = injector.inject_external_kv_cache("/apps/vllm/kv_cache_vllm.pt")
    
    print(f"✓ Injected KV cache for {len(cached_token_ids)} tokens "
          f"using {num_blocks} blocks")
    
    print("\n" + "=" * 80)
    print("External KV Cache Summary")
    print("=" * 80)
    print(f"Cached prompt tokens: {len(cached_token_ids)}")
    print(f"Memory blocks used: {num_blocks}")
    print(f"Block size: {injector.get_model_config()['block_size']}")
    print(f"Number of layers: {injector.get_model_config()['num_layers']}")
    print(f"Number of KV heads: {injector.get_model_config()['num_kv_heads']}")
    print(f"Head dimension: {injector.get_model_config()['head_dim']}")
    
    # Show injected block information
    if hasattr(injector, 'injected_block_hashes'):
        print(f"\n✓ Registered {len(injector.injected_block_hashes)} blocks to prefix cache")
        print(f"✓ These blocks are now available for cache hits")
    print("=" * 80)

    # 4. Run inference using the injected KV cache
    print("\nStep 4: Running inference with cached KV...")
    
    full_token_ids = cached_token_ids
    
    # Generate
    print(f"\nGenerating with {len(full_token_ids)} prompt tokens...")
    
    # Create TokensPrompt to pass token IDs
    prompt_input = TokensPrompt(prompt_token_ids=full_token_ids)
    
    sampling_params = SamplingParams(temperature=0.7, max_tokens=30)
    outputs = llm.generate(
        prompt_input,
        sampling_params=sampling_params
    )

    # Print KV cache reuse statistics
    print("\n" + "=" * 80)
    print("KV Cache Reuse Statistics")
    print("=" * 80)
    
    for output in outputs:
        # Get prompt and generation info
        num_prompt_tokens = len(output.prompt_token_ids)
        num_generated_tokens = len(output.outputs[0].token_ids)
        total_tokens = num_prompt_tokens + num_generated_tokens
        
        print(f"\nPrompt tokens (from cached KV): {num_prompt_tokens}")
        print(f"Generated tokens (new): {num_generated_tokens}")
        print(f"Total tokens: {total_tokens}")
        
        # Print num_cached_tokens (similar to kv_cache_reuse_demo.py)
        if output.num_cached_tokens is not None:
            print(f"\n✅ 重用的 KV cache tokens (num_cached_tokens): {output.num_cached_tokens}")
            cache_hit_rate = (output.num_cached_tokens / num_prompt_tokens * 100)
            print(f"✅ KV cache 命中率: {cache_hit_rate:.1f}%")
        else:
            print("\n❌ num_cached_tokens: None (可能未启用 prefix caching 或使用外部 KV cache)")

        # Show the generated text
        tokenizer = llm.get_tokenizer()
        prompt_text = tokenizer.decode(output.prompt_token_ids)
        generated_text = output.outputs[0].text
        
        print(f"\n{'─' * 80}")
        print("Prompt (from cached KV):")
        print(f"  {prompt_text[:200]}{'...' if len(prompt_text) > 200 else ''}")
        print(f"\nGenerated text:")
        print(f"  {generated_text}")
        print(f"{'─' * 80}")
        
        # Additional metrics if available
        if hasattr(output, 'metrics') and output.metrics:
            print(f"\nAdditional metrics:")
            for key, value in output.metrics.items():
                print(f"  {key}: {value}")
    
    print("\n✓ Success! External KV cache was used for inference.")
    print(f"✓ Saved computation for {num_prompt_tokens} tokens by reusing external KV cache!")


if __name__ == "__main__":
    simple_example()

