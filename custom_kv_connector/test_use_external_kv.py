#!/usr/bin/env python3
"""
Test script for CustomFileKVConnector.

This script:
1. Prepares a KV cache file
2. Starts vLLM with the custom connector
3. Sends requests and verifies KV cache is being used
4. Compares outputs with and without cached KV
"""

import argparse
import sys
import time
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Register the connector
import custom_kv_connector.register_connector  # noqa: F401

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def test_without_cache(model_path: str, prompt: str) -> tuple[str, float]:
    """Run inference without KV cache connector (baseline)"""
    print("\n" + "="*60)
    print("STEP 2: Baseline Inference (without cached KV)")
    print("="*60 + "\n")
    
    llm = LLM(
        model=model_path,
        enforce_eager=True,
        max_model_len=2048,
    )
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=20,
    )
    
    print(f"Prompt: {prompt[:100]}...")
    print("Running inference without cache...")
    
    start_time = time.time()
    outputs = llm.generate(prompt, sampling_params)
    elapsed = time.time() - start_time
    
    generated_text = outputs[0].outputs[0].text
    
    print(f"✓ Generated in {elapsed:.2f}s")
    print(f"Output: {generated_text}")
    
    return generated_text, elapsed


def test_with_cache(
    model_path: str,
    prompt: str,
    kv_cache_file: str,
    cached_token_ids: list[int]
) -> tuple[str, float]:
    """Run inference with KV cache connector"""
    
    # Configure vLLM to use our connector
    llm = LLM(
        model=model_path,
        enforce_eager=True,
        max_model_len=2048,
        kv_transfer_config={
            "kv_connector": "CustomFileKVConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "kv_cache_dir": str(Path(kv_cache_file).parent)
            }
        }
    )
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=20,
        # Pass the KV cache file path via extra_args
        extra_args={
            "kv_transfer_params": {
                "kv_cache_file": kv_cache_file
            }
        }
    )
    
    print(f"Prompt: {prompt[:100]}...")
    print(f"KV cache file: {kv_cache_file}")
    print(f"Cached tokens: {len(cached_token_ids)}")
    print("Running inference...")
    
    start_time = time.time()
    outputs = llm.generate(prompt, sampling_params)
    elapsed = time.time() - start_time
    
    generated_text = outputs[0].outputs[0].text
    
    # Get cache hit statistics
    request_output = outputs[0]
    num_cached = getattr(request_output, 'num_cached_tokens', None)
    print(f"Number of cached tokens: {num_cached}")
    
    print(f"✓ Generated in {elapsed:.2f}s")
    print(f"Output: {generated_text}")
    
    return generated_text, elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Test CustomFileKVConnector"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/apps/models/Qwen3-8B-Base",
        help="Model to test with (default: /apps/models/Qwen3-8B-Base)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\nIn triangle $ABC$, $\\sin \\angle A = \\frac{4}{5}$ and $\\angle A < 90^\\circ$. Let $D$ be a point outside triangle $ABC$ such that $\\angle BAD = \\angle DAC$ and $\\angle BDC = 90^\\circ$. Suppose that $AD = 1$ and that $\\frac{BD}{CD} = \\frac{3}{2}$. ",
        help="Test prompt"
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default="/apps/sharon_verl_new/verl/kv_cache.pt",
        help="Path to save KV cache file"
    )

    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("CustomFileKVConnector Test Suite")

    try:
        # # Step 1: Prepare KV cache
        # cached_token_ids, continuation_prompt = prepare_test_kv_cache(
        #     model_path=args.model,
        #     prompt=args.prompt,
        #     output_path=args.cache_file,
        #     num_tokens_to_cache=args.cache_tokens
        # )
        kv_cache = torch.load(args.cache_file)
        cached_token_ids = kv_cache['prompt_token_ids']
        
        # Step 3: Test with cache
        cached_output, cached_time = test_with_cache(
            model_path=args.model,
            prompt=args.prompt,
            kv_cache_file=args.cache_file,
            cached_token_ids=cached_token_ids
        )

        baseline_output, baseline_time = test_without_cache(
            model_path=args.model,
            prompt=args.prompt
        )
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

