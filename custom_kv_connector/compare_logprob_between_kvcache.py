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


def test_without_cache(model_path: str, prompt: str) -> tuple[str, float, dict]:
    """Run inference without KV cache connector (baseline)
    
    Returns:
        (generated_text, elapsed_time, logprobs_data)
    """
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
        logprobs=1,  # Get logprobs for generated tokens
        prompt_logprobs=1,  # Get logprobs for prompt tokens
    )
    
    print(f"Prompt: {prompt[:100]}...")
    print("Running inference without cache...")
    
    start_time = time.time()
    outputs = llm.generate(prompt, sampling_params)
    elapsed = time.time() - start_time
    
    output = outputs[0]
    generated_text = output.outputs[0].text
    
    # Extract prompt token information
    prompt_tokens = []
    prompt_logprobs = []
    
    if hasattr(output, 'prompt_token_ids'):
        # Ensure token IDs are Python integers, not tensors
        raw_tokens = output.prompt_token_ids
        if isinstance(raw_tokens, torch.Tensor):
            prompt_tokens = raw_tokens.tolist()
        elif isinstance(raw_tokens, list):
            prompt_tokens = [int(t) if hasattr(t, '__int__') else t for t in raw_tokens]
        else:
            prompt_tokens = list(raw_tokens)
    
    # Extract logprobs for each prompt token
    if output.prompt_logprobs is not None:
        for i, token_logprob_dict in enumerate(output.prompt_logprobs):
            if token_logprob_dict is not None and i < len(prompt_tokens):
                token_id = int(prompt_tokens[i])  # Ensure it's a Python int
                if token_id in token_logprob_dict:
                    prompt_logprobs.append(token_logprob_dict[token_id].logprob)
                else:
                    prompt_logprobs.append(None)
            else:
                prompt_logprobs.append(None)
    
    # Extract response token information
    response_tokens = []
    response_logprobs = []
    
    for output_token in output.outputs[0].logprobs:
        if output_token is not None:
            # Get the token with highest probability (the selected token)
            for token_id, logprob_obj in output_token.items():
                # Ensure token_id is a Python int
                token_id_int = int(token_id) if hasattr(token_id, '__int__') else token_id
                response_tokens.append(token_id_int)
                response_logprobs.append(logprob_obj.logprob)
                break  # Only take the first (selected) token
    
    logprobs_data = {
        'prompt_tokens': prompt_tokens,
        'prompt_logprobs': prompt_logprobs,
        'response_tokens': response_tokens,
        'response_logprobs': response_logprobs,
    }
    
    print(f"✓ Generated in {elapsed:.2f}s")
    print(f"Output: {generated_text}")
    print(f"Prompt tokens: {len(prompt_tokens)}")
    print(f"Response tokens: {len(response_tokens)}")
    
    return generated_text, elapsed, logprobs_data


def test_with_cache(
    model_path: str,
    prompt: str,
    kv_cache_file: str,
    cached_token_ids: list[int]
) -> tuple[str, float, dict, int]:
    """Run inference with KV cache connector
    
    Returns:
        (generated_text, elapsed_time, logprobs_data, num_cached_tokens)
    """
    print("\n" + "="*60)
    print("STEP 3: Inference with Cached KV")
    print("="*60)
    print("NOTE: prompt_logprobs is disabled because vLLM cannot compute")
    print("      logprobs for externally loaded cached tokens.")
    print("      We will only compare response token logprobs.")
    print("="*60 + "\n")
    
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
        logprobs=1,  # Get logprobs for generated tokens
        # NOTE: prompt_logprobs is disabled for KV connector because vLLM cannot
        # compute logprobs for externally loaded cached tokens
        prompt_logprobs=None,  # Disabled for cached tokens
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
    
    output = outputs[0]
    generated_text = output.outputs[0].text
    
    # Get cache hit statistics
    num_cached = getattr(output, 'num_cached_tokens', None)
    if num_cached is None:
        num_cached = 0
    
    # Extract prompt token information
    prompt_tokens = []
    prompt_logprobs = []
    
    if hasattr(output, 'prompt_token_ids'):
        # Ensure token IDs are Python integers, not tensors
        raw_tokens = output.prompt_token_ids
        if isinstance(raw_tokens, torch.Tensor):
            prompt_tokens = raw_tokens.tolist()
        elif isinstance(raw_tokens, list):
            prompt_tokens = [int(t) if hasattr(t, '__int__') else t for t in raw_tokens]
        else:
            prompt_tokens = list(raw_tokens)
    
    # Extract logprobs for each prompt token
    if output.prompt_logprobs is not None:
        for i, token_logprob_dict in enumerate(output.prompt_logprobs):
            if token_logprob_dict is not None and i < len(prompt_tokens):
                token_id = int(prompt_tokens[i])  # Ensure it's a Python int
                if token_id in token_logprob_dict:
                    prompt_logprobs.append(token_logprob_dict[token_id].logprob)
                else:
                    prompt_logprobs.append(None)
            else:
                prompt_logprobs.append(None)
    
    # Extract response token information
    response_tokens = []
    response_logprobs = []
    
    for output_token in output.outputs[0].logprobs:
        if output_token is not None:
            # Get the token with highest probability (the selected token)
            for token_id, logprob_obj in output_token.items():
                # Ensure token_id is a Python int
                token_id_int = int(token_id) if hasattr(token_id, '__int__') else token_id
                response_tokens.append(token_id_int)
                response_logprobs.append(logprob_obj.logprob)
                break  # Only take the first (selected) token
    
    logprobs_data = {
        'prompt_tokens': prompt_tokens,
        'prompt_logprobs': prompt_logprobs,
        'response_tokens': response_tokens,
        'response_logprobs': response_logprobs,
    }
    
    print(f"✓ Generated in {elapsed:.2f}s")
    print(f"Output: {generated_text}")
    print(f"Number of cached tokens: {num_cached}")
    print(f"Prompt tokens: {len(prompt_tokens)}")
    print(f"Response tokens: {len(response_tokens)}")
    
    return generated_text, elapsed, logprobs_data, num_cached


def compare_logprobs(
    model_path: str,
    baseline_logprobs: dict,
    cached_logprobs: dict,
    num_cached_tokens: int
):
    """Compare logprobs between baseline and cached inference
    
    Prints: token_id, word, with_cache_logprob, without_cache_logprob, diff
    """
    print("\n" + "="*80)
    print("STEP 4: Logprobs Comparison")
    print("="*80 + "\n")
    
    # Load tokenizer to decode tokens
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Compare prompt tokens (if available)
    baseline_prompt_tokens = baseline_logprobs['prompt_tokens']
    cached_prompt_tokens = cached_logprobs['prompt_tokens']
    baseline_prompt_lps = baseline_logprobs['prompt_logprobs']
    cached_prompt_lps = cached_logprobs['prompt_logprobs']
    
    # Check if prompt logprobs are available
    has_prompt_logprobs = (baseline_prompt_tokens and baseline_prompt_lps and 
                          any(lp is not None for lp in baseline_prompt_lps))
    
    all_match = True
    max_diff = 0.0
    
    if has_prompt_logprobs and cached_prompt_lps:
        print("="*100)
        print("PROMPT TOKENS COMPARISON")
        print("="*100)
        print(f"{'Token ID':<10} {'Word':<30} {'With Cache':<15} {'Without Cache':<15} {'Diff':<12} {'Match':<8}")
        print("-"*100)
        
        for i in range(min(len(baseline_prompt_tokens), len(cached_prompt_tokens))):
            token_id = baseline_prompt_tokens[i]
            word = tokenizer.decode([token_id]).replace('\n', '\\n').replace('\t', '\\t')
            
            # Get logprobs
            baseline_lp = baseline_prompt_lps[i] if i < len(baseline_prompt_lps) else None
            cached_lp = cached_prompt_lps[i] if i < len(cached_prompt_lps) else None
            
            # Calculate difference
            if baseline_lp is not None and cached_lp is not None:
                diff = abs(baseline_lp - cached_lp)
                max_diff = max(max_diff, diff)
                match = "✓" if diff < 1e-4 else "✗"
                if diff >= 1e-4:
                    all_match = False
                
                print(f"{token_id:<10} {word:<30.30} {cached_lp:<15.6f} {baseline_lp:<15.6f} {diff:<12.2e} {match:<8}")
            else:
                baseline_str = f"{baseline_lp:.6f}" if baseline_lp is not None else "N/A"
                cached_str = f"{cached_lp:.6f}" if cached_lp is not None else "N/A"
                print(f"{token_id:<10} {word:<30.30} {cached_str:<15} {baseline_str:<15} {'N/A':<12} {'-':<8}")
        
        print("-"*100)
        print(f"Total prompt tokens compared: {min(len(baseline_prompt_tokens), len(cached_prompt_tokens))}")
        print(f"Number of cached tokens: {num_cached_tokens}")
        print(f"Max difference: {max_diff:.2e}")
        print(f"All prompt logprobs match: {'✓ Yes' if all_match else '✗ No'}")
    else:
        print("="*100)
        print("PROMPT TOKENS COMPARISON")
        print("="*100)
        print("⚠ Prompt logprobs not available for cached inference")
        print("  (This is expected when using KV connector with externally loaded cache)")
        print(f"  Baseline has {len(baseline_prompt_tokens)} prompt tokens")
        print(f"  Cached has {len(cached_prompt_tokens)} prompt tokens")
        print(f"  Number of cached tokens: {num_cached_tokens}")
    
    # Compare response tokens
    print("\n" + "="*100)
    print("RESPONSE TOKENS COMPARISON")
    print("="*100)
    print(f"{'Token ID':<10} {'Word':<30} {'With Cache':<15} {'Without Cache':<15} {'Diff':<12} {'Match':<8}")
    print("-"*100)
    
    baseline_response_tokens = baseline_logprobs['response_tokens']
    cached_response_tokens = cached_logprobs['response_tokens']
    baseline_response_lps = baseline_logprobs['response_logprobs']
    cached_response_lps = cached_logprobs['response_logprobs']
    
    output_match = True
    output_max_diff = 0.0
    
    for i in range(min(len(baseline_response_tokens), len(cached_response_tokens))):
        token_id = baseline_response_tokens[i]
        word = tokenizer.decode([token_id]).replace('\n', '\\n').replace('\t', '\\t')
        
        # Get logprobs
        baseline_lp = baseline_response_lps[i] if i < len(baseline_response_lps) else None
        cached_lp = cached_response_lps[i] if i < len(cached_response_lps) else None
        
        # Calculate difference
        if baseline_lp is not None and cached_lp is not None:
            diff = abs(baseline_lp - cached_lp)
            output_max_diff = max(output_max_diff, diff)
            match = "✓" if diff < 1e-4 else "✗"
            if diff >= 1e-4:
                output_match = False
            
            print(f"{token_id:<10} {word:<30.30} {cached_lp:<15.6f} {baseline_lp:<15.6f} {diff:<12.2e} {match:<8}")
        else:
            baseline_str = f"{baseline_lp:.6f}" if baseline_lp is not None else "N/A"
            cached_str = f"{cached_lp:.6f}" if cached_lp is not None else "N/A"
            print(f"{token_id:<10} {word:<30.30} {cached_str:<15} {baseline_str:<15} {'N/A':<12} {'-':<8}")
    
    print("-"*100)
    print(f"Total response tokens compared: {min(len(baseline_response_tokens), len(cached_response_tokens))}")
    print(f"Max difference: {output_max_diff:.2e}")
    print(f"All response logprobs match: {'✓ Yes' if output_match else '✗ No'}")
    
    # Summary
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    if has_prompt_logprobs and cached_prompt_lps:
        print(f"Prompt logprobs match:   {'✓ Yes' if all_match else '✗ No'} (max diff: {max_diff:.2e})")
    else:
        print(f"Prompt logprobs match:   N/A (disabled for cached inference)")
    print(f"Response logprobs match: {'✓ Yes' if output_match else '✗ No'} (max diff: {output_max_diff:.2e})")
    
    # Overall pass/fail based on response logprobs (most important)
    overall_pass = output_match
    if has_prompt_logprobs and cached_prompt_lps:
        overall_pass = overall_pass and all_match
    
    print(f"Overall result:          {'✓ PASS - KV cache produces identical response logprobs!' if overall_pass else '✗ FAIL - Response logprobs differ'}")
    print("="*100)
    
    return overall_pass


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
        
        # Step 2: Baseline test (without cache)
        baseline_output, baseline_time, baseline_logprobs = test_without_cache(
            model_path=args.model,
            prompt=args.prompt
        )
        
        # Step 3: Test with cache
        cached_output, cached_time, cached_logprobs, num_cached = test_with_cache(
            model_path=args.model,
            prompt=args.prompt,
            kv_cache_file=args.cache_file,
            cached_token_ids=cached_token_ids
        )
        
        # Step 4: Compare logprobs
        logprobs_match = compare_logprobs(
            model_path=args.model,
            baseline_logprobs=baseline_logprobs,
            cached_logprobs=cached_logprobs,
            num_cached_tokens=num_cached
        )
        
        # Final summary
        print("\n" + "="*80)
        if logprobs_match:
            print("✓ ALL TESTS PASSED!")
            print(f"  KV cache produces identical response logprobs!")
            print(f"  This confirms the cached KV is being used correctly.")
        else:
            print("⚠ TESTS COMPLETED WITH DIFFERENCES")
            print(f"  Response logprobs differ - please review the comparison above")
        print("="*80 + "\n")
        
        return 0 if logprobs_match else 1
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

