#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Example script for using external KV cache in vLLM.

This script demonstrates how to:
1. Load external KV cache from file
2. Convert it to vLLM's internal format
3. Inject it into vLLM engine
4. Run inference using the injected KV cache

The external KV cache format should be:
{
    'prompt_token_ids': list[int],
    'keys': torch.Tensor,    # Shape: (num_layers, batch_size, num_heads, seq_len, head_dim)
    'values': torch.Tensor,  # Shape: (num_layers, batch_size, num_heads, seq_len, head_dim)
}

Important:
    This script requires disabling V1 multiprocessing mode.
    Set environment variable before running:
        export VLLM_ENABLE_V1_MULTIPROCESSING=0
"""

import torch
from typing import List, Optional, Tuple, Callable, Any
import argparse
import os
import sys

from vllm import LLM, SamplingParams
from vllm.attention.layer import Attention
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.core.kv_cache_utils import hash_block_tokens, BlockHash
from vllm.v1.core.block_pool import make_block_hash_with_group_id
from vllm.utils import get_hash_fn_by_name


class ExternalKVCacheInjector:
    """Helper class to inject external KV cache into vLLM engine."""
    
    def __init__(self, llm: LLM):
        """
        Initialize the injector with a vLLM LLM instance.
        
        Args:
            llm: The vLLM LLM instance
        
        Note:
            This requires VLLM_ENABLE_V1_MULTIPROCESSING=0 to be set.
        """
        self.llm = llm
        self.engine = llm.llm_engine
        
        # Try to access model executor
        # V1 multiprocessing mode (SyncMPClient) doesn't allow direct access
        if hasattr(self.engine, 'model_executor'):
            # V0 or V1 non-multiprocessing mode
            self.model_executor = self.engine.model_executor
        elif hasattr(self.engine, 'engine_core'):
            # V1 in-process mode
            engine_core = self.engine.engine_core
            if hasattr(engine_core, 'engine_core'):
                # This is multiprocessing mode - cannot access
                raise RuntimeError(
                    "Cannot access model executor in V1 multiprocessing mode.\n"
                    "Please set environment variable: VLLM_ENABLE_V1_MULTIPROCESSING=0\n"
                    "and restart your script."
                )
            self.model_executor = engine_core.model_executor
        else:
            raise RuntimeError(
                f"Cannot access model executor from engine type: {type(self.engine)}\n"
                f"Engine attributes: {dir(self.engine)}\n"
                "Please set environment variable: VLLM_ENABLE_V1_MULTIPROCESSING=0"
            )
            
    def get_model_config(self):
        """Get model configuration including layer info."""
        vllm_config = self.engine.vllm_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        parallel_config = vllm_config.parallel_config
        
        # Get attention layers from model
        attention_layers = {}
        if hasattr(self.model_executor, 'driver_worker'):
            # For multi-GPU setup
            worker = self.model_executor.driver_worker
            model_runner = worker.model_runner
        else:
            # For single-GPU setup
            model_runner = getattr(self.model_executor, 'model_runner', None)
            
        if model_runner is None:
            raise RuntimeError("Cannot access model runner")
            
        # Get attention context
        forward_context = model_runner.compilation_config.static_forward_context
        
        for layer_name in forward_context:
            layer = forward_context[layer_name]
            if isinstance(layer, Attention):
                attention_layers[layer_name] = layer
                
        return {
            'num_layers': model_config.hf_text_config.num_hidden_layers,
            'num_kv_heads': model_config.get_num_kv_heads(parallel_config),
            'head_dim': model_config.get_head_size(),
            'block_size': cache_config.block_size,
            'attention_layers': attention_layers,
            'model_runner': model_runner,
        }
    
    def convert_external_kv_to_vllm_format(
        self,
        external_keys: torch.Tensor,
        external_values: torch.Tensor,
        prompt_token_ids: List[int],
    ) -> Tuple[dict, int]:
        """
        Convert external KV cache to vLLM's paged KV cache format.
        
        Args:
            external_keys: External keys tensor 
                Shape: (num_layers, batch_size, num_heads, seq_len, head_dim)
            external_values: External values tensor
                Shape: (num_layers, batch_size, num_heads, seq_len, head_dim)
            prompt_token_ids: Token IDs of the prompt
            
        Returns:
            A tuple of (converted_kv_caches, num_blocks_needed)
        """
        model_config = self.get_model_config()
        num_layers = external_keys.shape[0]
        batch_size = external_keys.shape[1]
        num_heads = external_keys.shape[2]
        seq_len = external_keys.shape[3]
        head_dim = external_keys.shape[4]
        
        block_size = model_config['block_size']
        num_kv_heads = model_config['num_kv_heads']
        
        # Validate shapes
        assert batch_size == 1, "Currently only support batch_size=1"
        assert external_values.shape == external_keys.shape, \
            "Keys and values must have the same shape"
        assert head_dim == model_config['head_dim'], \
            f"Head dim mismatch: {head_dim} vs {model_config['head_dim']}"
        assert seq_len == len(prompt_token_ids), \
            f"Sequence length mismatch: {seq_len} vs {len(prompt_token_ids)}"
            
        # Calculate number of blocks needed
        num_blocks_needed = (seq_len + block_size - 1) // block_size
        
        print(f"External KV cache info:")
        print(f"  Num layers: {num_layers}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Num heads: {num_heads}")
        print(f"  Head dim: {head_dim}")
        print(f"  Block size: {block_size}")
        print(f"  Num blocks needed: {num_blocks_needed}")
        
        # Get KV cache tensors from model runner
        model_runner = model_config['model_runner']
        kv_caches = model_runner.kv_caches
        
        if not kv_caches or len(kv_caches) == 0:
            raise RuntimeError("KV caches not initialized in model runner")
        
        print(f"\nConverting external KV cache to vLLM format...")
        print(f"Total number of KV cache tensors: {len(kv_caches)}")
        
        # Debug: print first few tensor shapes
        for i in range(min(3, len(kv_caches))):
            tensor = kv_caches[i]
            if isinstance(tensor, torch.Tensor):
                print(f"  KV cache[{i}] shape: {tensor.shape}, ndim: {tensor.ndim}")
            else:
                print(f"  KV cache[{i}] type: {type(tensor)}")
        
        # Convert layer by layer
        attention_layers = model_config['attention_layers']
        layer_names = sorted(attention_layers.keys(), 
                           key=lambda x: extract_layer_index(x))
        
        print(f"\nAttention layers to process: {len(layer_names)}")
        
        for layer_idx, layer_name in enumerate(layer_names):
            if layer_idx >= len(kv_caches):
                print(f"Warning: Layer {layer_name} (idx={layer_idx}) "
                      f"exceeds KV cache tensor count ({len(kv_caches)})")
                break
                
            kv_cache_tensor = kv_caches[layer_idx]
            print(f"\nProcessing layer {layer_name} (idx={layer_idx}):")
            print(f"  KV cache tensor shape: {kv_cache_tensor.shape}")
            print(f"  KV cache tensor ndim: {kv_cache_tensor.ndim}")
            
            # Determine KV cache format based on shape
            # Common formats:
            # 1. [2, num_blocks, block_size, num_kv_heads, head_dim] - dim 0 is [key, value]
            # 2. [num_blocks, block_size, num_kv_heads, head_dim] - single tensor (might be key only or interleaved)
            # 3. [num_blocks, block_size, num_kv_heads*2, head_dim] - interleaved key/value in head dimension
            
            # Get external K and V for this layer
            if layer_idx >= num_layers:
                print(f"  Skipping: layer_idx {layer_idx} >= num_layers {num_layers}")
                continue
                
            ext_k = external_keys[layer_idx, 0]  # Remove batch dim
            ext_v = external_values[layer_idx, 0]  # Remove batch dim
            # ext_k/v shape: (num_heads, seq_len, head_dim)
            
            # Determine format and convert
            if kv_cache_tensor.ndim == 5:
                # Format: [2, num_blocks, block_size, num_kv_heads, head_dim]
                print(f"  Using 5D format [2, num_blocks, block_size, num_kv_heads, head_dim]")
                for block_idx in range(num_blocks_needed):
                    start_pos = block_idx * block_size
                    end_pos = min(start_pos + block_size, seq_len)
                    block_len = end_pos - start_pos
                    
                    k_block = ext_k[:, start_pos:end_pos, :]  # (num_heads, block_len, head_dim)
                    v_block = ext_v[:, start_pos:end_pos, :]
                    
                    # Write to KV cache [0] for key, [1] for value
                    kv_cache_tensor[0, block_idx, :block_len, :, :] = \
                        k_block.permute(1, 0, 2)  # (block_len, num_heads, head_dim)
                    kv_cache_tensor[1, block_idx, :block_len, :, :] = \
                        v_block.permute(1, 0, 2)
                        
                    if block_len < block_size:
                        kv_cache_tensor[0, block_idx, block_len:, :, :] = 0
                        kv_cache_tensor[1, block_idx, block_len:, :, :] = 0
                        
            elif kv_cache_tensor.ndim == 4:
                # Format: [num_blocks, block_size, num_kv_heads, head_dim]
                # This might be a single tensor per layer, need to check if there are separate K/V tensors
                print(f"  Using 4D format [num_blocks, block_size, num_kv_heads, head_dim]")
                print(f"  Warning: 4D format detected. This format is ambiguous.")
                print(f"  Assuming this is key cache and value cache is in a separate tensor.")
                print(f"  If this doesn't work, the KV cache structure might be different.")
                
                # Try to access both key and value caches from forward context
                attention_layer = attention_layers[layer_name]
                if hasattr(attention_layer, 'kv_cache') and len(attention_layer.kv_cache) > 0:
                    layer_kv = attention_layer.kv_cache[0]  # [ve_idx]
                    print(f"  Layer KV cache type: {type(layer_kv)}")
                    print(f"  Layer KV cache shape: {layer_kv.shape if isinstance(layer_kv, torch.Tensor) else 'N/A'}")
                    
                # For now, raise an error with helpful information
                raise NotImplementedError(
                    f"4D KV cache format detected but handling is not fully implemented.\n"
                    f"KV cache shape: {kv_cache_tensor.shape}\n"
                    f"Please check if:\n"
                    f"  1. Keys and values are in separate tensors in the kv_caches list\n"
                    f"  2. Or they are interleaved in a different dimension\n"
                    f"\nDebug info:\n"
                    f"  Total kv_caches length: {len(kv_caches)}\n"
                    f"  Current layer_idx: {layer_idx}\n"
                    f"  Expected num_layers: {num_layers}"
                )
            else:
                raise ValueError(
                    f"Unexpected KV cache tensor dimensions: {kv_cache_tensor.ndim}\n"
                    f"Shape: {kv_cache_tensor.shape}"
                )
                    
            print(f"  Converted {num_blocks_needed} blocks")
        
        print(f"\n{'=' * 80}")
        print("Registering blocks to vLLM's caching system...")
        print(f"{'=' * 80}")
        
        # Step 2: Register blocks to block pool and compute hashes
        self._register_blocks_to_cache(
            prompt_token_ids=prompt_token_ids,
            num_blocks=num_blocks_needed,
            block_size=block_size
        )
            
        return kv_caches, num_blocks_needed
    
    def _register_blocks_to_cache(
        self,
        prompt_token_ids: List[int],
        num_blocks: int,
        block_size: int,
    ) -> List[BlockHash]:
        """
        Register the injected KV cache blocks to vLLM's block pool.
        This computes block hashes and registers them in the cache.
        
        Args:
            prompt_token_ids: Token IDs of the prompt
            num_blocks: Number of blocks to register
            block_size: Size of each block
            
        Returns:
            List of block hashes
        """
        # Get block pool from KV cache manager
        # Access path: engine -> engine_core (InprocClient) -> engine_core (EngineCore) -> scheduler
        if hasattr(self.engine, 'engine_core'):
            # V1 with InprocClient
            engine_core_client = self.engine.engine_core
            if hasattr(engine_core_client, 'engine_core'):
                # InprocClient has engine_core attribute
                actual_engine_core = engine_core_client.engine_core
            else:
                # Direct EngineCore
                actual_engine_core = engine_core_client
        else:
            # V0 or other modes
            raise RuntimeError("Cannot find EngineCore to access scheduler")
        
        kv_cache_manager = actual_engine_core.scheduler.kv_cache_manager
        block_pool = kv_cache_manager.block_pool
        
        if not kv_cache_manager.enable_caching:
            print("Warning: Prefix caching is disabled, blocks will not be cached")
            return []
        
        # Get hash function
        cache_config = self.engine.vllm_config.cache_config
        hash_algo_name = cache_config.prefix_caching_hash_algo
        hash_function = get_hash_fn_by_name(hash_algo_name)
        
        # Calculate number of full blocks
        num_full_blocks = len(prompt_token_ids) // block_size
        num_cached_tokens = num_full_blocks * block_size
        num_uncached_tokens = len(prompt_token_ids) - num_cached_tokens
        
        print(f"Computing block hashes for {num_full_blocks} full blocks...")
        print(f"  Total tokens: {len(prompt_token_ids)}")
        print(f"  Block size: {block_size}")
        print(f"  Full blocks: {num_full_blocks} (cacheable)")
        print(f"  Tokens in full blocks: {num_cached_tokens} (will be cached)")
        if num_uncached_tokens > 0:
            print(f"  ⚠️  Incomplete block: {num_uncached_tokens} token(s) (will NOT be cached)")
            print(f"  ⚠️  NOTE: vLLM only caches FULL blocks. The last {num_uncached_tokens} token(s)")
            print(f"           will need to be recomputed during inference.")
            print(f"  💡 TIP: For optimal performance, generate external KV cache with")
            print(f"           token count that's a multiple of block_size ({block_size})")
        
        # Compute block hashes
        block_hashes: List[BlockHash] = []
        parent_hash = None
        
        for block_idx in range(num_full_blocks):
            start_pos = block_idx * block_size
            end_pos = start_pos + block_size
            
            # Get token IDs for this block
            block_token_ids = prompt_token_ids[start_pos:end_pos]
            
            # Compute hash for this block
            block_hash = hash_block_tokens(
                hash_function=hash_function,
                parent_block_hash=parent_hash,
                curr_block_token_ids=block_token_ids,
                extra_keys=None  # No LoRA or multi-modal for now
            )
            block_hashes.append(block_hash)
            parent_hash = block_hash
            
            print(f"  Block {block_idx}: hash computed for tokens [{start_pos}:{end_pos}]")
        
        print(f"\n✓ Computed {len(block_hashes)} block hashes for {num_cached_tokens} cacheable tokens")
        
        # Allocate blocks from block pool and register them
        print(f"\nAllocating and registering {len(block_hashes)} blocks...")
        
        allocated_blocks = []
        kv_cache_group_ids = list(range(len(kv_cache_manager.kv_cache_config.kv_cache_groups)))
        
        for block_idx, block_hash in enumerate(block_hashes):
            # Allocate blocks for each KV cache group
            blocks_per_group = []
            
            for group_id in kv_cache_group_ids:
                # Try to get an already cached block first
                block_hash_with_group = make_block_hash_with_group_id(block_hash, group_id)
                existing_block = block_pool.cached_block_hash_to_block.get_one_block(
                    block_hash_with_group
                )
                
                if existing_block:
                    print(f"  Block {block_idx} (group {group_id}): reusing existing block {existing_block.block_id}")
                    blocks_per_group.append(existing_block)
                else:
                    # Allocate a new block
                    if block_pool.get_num_free_blocks() > 0:
                        new_block = block_pool.free_block_queue.popleft()
                        new_block.ref_cnt = 1
                        
                        # Register the block in cache
                        block_pool.cached_block_hash_to_block.insert(
                            block_hash_with_group, new_block
                        )
                        
                        print(f"  Block {block_idx} (group {group_id}): allocated and cached block {new_block.block_id}")
                        blocks_per_group.append(new_block)
                    else:
                        print(f"  Warning: No free blocks available for block {block_idx} (group {group_id})")
                        # This is a problem - we need to evict or handle this
                        raise RuntimeError(f"Insufficient free blocks to register external KV cache")
            
            allocated_blocks.append(blocks_per_group)
        
        print(f"\n✓ Successfully registered {len(allocated_blocks)} blocks to cache")
        print(f"✓ Block hashes: {len(block_hashes)}")
        print(f"✓ Cacheable tokens: {num_cached_tokens} out of {len(prompt_token_ids)} total")
        
        # Store the block hashes for later use
        self.injected_block_hashes = block_hashes
        self.injected_blocks = allocated_blocks
        self.num_cached_tokens = num_cached_tokens
        self.num_uncached_tokens = num_uncached_tokens
        
        return block_hashes
    
    def inject_external_kv_cache(
        self,
        external_kv_path: str,
    ) -> Tuple[List[int], int, int, int]:
        """
        Load and inject external KV cache into vLLM.
        
        Args:
            external_kv_path: Path to the saved external KV cache file
            
        Returns:
            A tuple of (prompt_token_ids, num_full_blocks, num_cached_tokens, num_uncached_tokens)
            - prompt_token_ids: List of token IDs
            - num_full_blocks: Number of full blocks that were cached
            - num_cached_tokens: Number of tokens that are actually cached (full blocks only)
            - num_uncached_tokens: Number of tokens that were NOT cached (incomplete block)
            
        Note:
            vLLM only caches FULL blocks. If your token count is not a multiple of
            block_size, the last few tokens will NOT be cached and will need to be
            recomputed during inference.
        """
        # Load external KV cache
        print(f"Loading external KV cache from {external_kv_path}...")
        loaded = torch.load(external_kv_path, map_location='cpu')
        
        prompt_token_ids = loaded['prompt_token_ids']
        external_keys = loaded['keys']
        external_values = loaded['values']
        print(f" external key dtype: {external_keys.dtype}, external values dtype: {external_values.dtype}")
        print(f" external key sample: {external_keys[0, 0, 0, 0:10, :]}")
        print(f" external values sample: {external_values[0:1, 0, 0, 0:10, :]}")
        
        print(f"Loaded KV cache with {len(prompt_token_ids)} tokens")
        
        # Move to GPU if needed
        if torch.cuda.is_available():
            external_keys = external_keys.cuda()
            external_values = external_values.cuda()
        
        # Convert and inject
        converted_kv, num_blocks = self.convert_external_kv_to_vllm_format(
            external_keys, external_values, prompt_token_ids
        )
        
        print(f"\nSuccessfully injected external KV cache!")
        print(f"Used {num_blocks} blocks for {len(prompt_token_ids)} tokens")
        
        return prompt_token_ids, num_blocks
    
    def get_injected_block_hashes(self) -> List[BlockHash]:
        """
        Get the block hashes that were computed during injection.
        These can be used to create requests that will recognize the cached blocks.
        
        Returns:
            List of block hashes
        """
        if not hasattr(self, 'injected_block_hashes'):
            raise RuntimeError("No KV cache has been injected yet. Call inject_external_kv_cache() first.")
        return self.injected_block_hashes
    
    def get_num_cached_tokens(self) -> int:
        """
        Get the number of tokens that are cached.
        
        Returns:
            Number of cached tokens
        """
        if not hasattr(self, 'injected_block_hashes'):
            return 0
        model_config = self.get_model_config()
        return len(self.injected_block_hashes) * model_config['block_size']


def create_external_kv_cache_example(
    model_name: str,
    output_path: str,
    prompt: str = "Once upon a time",
):
    """
    Create an example external KV cache file for testing.
    
    This runs vLLM once to generate a KV cache and saves it in the
    external format.
    """
    print(f"Creating example external KV cache...")
    print(f"Model: {model_name}")
    print(f"Prompt: {prompt}")
    
    # Initialize vLLM
    llm = LLM(model=model_name, max_model_len=512, enforce_eager=True)
    tokenizer = llm.get_tokenizer()
    
    # Tokenize prompt
    prompt_token_ids = tokenizer.encode(prompt)
    print(f"Prompt has {len(prompt_token_ids)} tokens")
    
    # Run a dummy inference to populate KV cache
    sampling_params = SamplingParams(temperature=0, max_tokens=1)
    outputs = llm.generate(prompt_token_ids=prompt_token_ids, 
                          sampling_params=sampling_params)
    
    # Access KV cache from model runner
    injector = ExternalKVCacheInjector(llm)
    model_config = injector.get_model_config()
    model_runner = model_config['model_runner']
    kv_caches = model_runner.kv_caches
    
    # Extract KV cache in external format
    ve_idx = 0
    ve_kv_cache = kv_caches[ve_idx]
    
    num_layers = len(ve_kv_cache)
    seq_len = len(prompt_token_ids)
    
    # Get first layer to determine shapes
    first_kv = ve_kv_cache[0]
    # Shape: [2, num_blocks, block_size, num_kv_heads, head_dim]
    num_kv_heads = first_kv.shape[3]
    head_dim = first_kv.shape[4]
    block_size = first_kv.shape[2]
    
    num_blocks_used = (seq_len + block_size - 1) // block_size
    
    print(f"\nExtracting KV cache:")
    print(f"  Num layers: {num_layers}")
    print(f"  Num KV heads: {num_kv_heads}")
    print(f"  Head dim: {head_dim}")
    print(f"  Block size: {block_size}")
    
    # Convert to external format
    external_keys = torch.zeros(
        (num_layers, 1, num_kv_heads, seq_len, head_dim),
        dtype=first_kv.dtype,
        device=first_kv.device
    )
    external_values = torch.zeros_like(external_keys)
    
    for layer_idx in range(num_layers):
        kv_cache_tensor = ve_kv_cache[layer_idx]
        
        # Extract from paged format
        for block_idx in range(num_blocks_used):
            start_pos = block_idx * block_size
            end_pos = min(start_pos + block_size, seq_len)
            block_len = end_pos - start_pos
            
            # Extract K and V
            k_block = kv_cache_tensor[0, block_idx, :block_len, :, :]
            v_block = kv_cache_tensor[1, block_idx, :block_len, :, :]
            
            # k/v_block shape: (block_len, num_kv_heads, head_dim)
            external_keys[layer_idx, 0, :, start_pos:end_pos, :] = \
                k_block.permute(1, 0, 2)  # (num_kv_heads, block_len, head_dim)
            external_values[layer_idx, 0, :, start_pos:end_pos, :] = \
                v_block.permute(1, 0, 2)
    
    # Move to CPU for saving
    external_keys = external_keys.cpu()
    external_values = external_values.cpu()
    
    # Save
    save_dict = {
        'prompt_token_ids': prompt_token_ids,
        'keys': external_keys,
        'values': external_values,
    }
    
    torch.save(save_dict, output_path)
    print(f"\nSaved external KV cache to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    return output_path


def main():
    # Check environment variable
    if os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING', '1') != '0':
        print("=" * 80)
        print("ERROR: V1 multiprocessing mode is enabled.")
        print("This script requires direct access to model executor.")
        print()
        print("Please set the environment variable and try again:")
        print("  export VLLM_ENABLE_V1_MULTIPROCESSING=0")
        print()
        print("Or run the script with:")
        print("  VLLM_ENABLE_V1_MULTIPROCESSING=0 python", " ".join(sys.argv))
        print("=" * 80)
        sys.exit(1)
    
    parser = argparse.ArgumentParser(
        description="Example for using external KV cache in vLLM"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/apps/models/Qwen3-8B",
        help="Model name or path"
    )
    parser.add_argument(
        "--kv-cache-path",
        type=str,
        default="/apps/sharon_verl_new/verl/kv_cache.pt",
        help="Path to external KV cache file. If not provided, will create an example."
    )
    parser.add_argument(
        "--create-example",
        action="store_true",
        help="Create an example KV cache file"
    )
    parser.add_argument(
        "--example-output",
        type=str,
        default="example_kv_cache.pt",
        help="Output path for example KV cache"
    )
    parser.add_argument(
        "--continuation-prompt",
        type=str,
        default=" and they lived",
        help="Continuation prompt to test KV cache reuse"
    )
    
    args = parser.parse_args()
    
    # # Step 1: Create example if needed
    # if args.create_example or args.kv_cache_path is None:
    #     kv_cache_path = create_external_kv_cache_example(
    #         model_name=args.model,
    #         output_path=args.example_output,
    #         prompt="Once upon a time"
    #     )
    # else:
    kv_cache_path = args.kv_cache_path
    
    print("\n" + "=" * 80)
    print("Testing external KV cache injection")
    print("=" * 80)
    
    # Step 2: Load vLLM and inject KV cache
    print("\nInitializing vLLM...")
    llm = LLM(model=args.model, max_model_len=512, enforce_eager=True)
    
    print("\nInjecting external KV cache...")
    injector = ExternalKVCacheInjector(llm)
    prompt_token_ids, num_blocks = injector.inject_external_kv_cache(kv_cache_path)
    
    # Step 3: Test inference with injected KV cache
    print("\n" + "=" * 80)
    print("Running inference with injected KV cache")
    print("=" * 80)
    
    tokenizer = llm.get_tokenizer()
    original_prompt = tokenizer.decode(prompt_token_ids)
    print(f"\nOriginal prompt (from KV cache): {original_prompt!r}")
    
    # Continue from the cached prompt
    continuation_tokens = tokenizer.encode(args.continuation_prompt, 
                                          add_special_tokens=False)
    full_token_ids = prompt_token_ids + continuation_tokens
    
    full_prompt = tokenizer.decode(full_token_ids)
    print(f"Full prompt (with continuation): {full_prompt!r}")
    
    # Generate
    sampling_params = SamplingParams(temperature=0.7, max_tokens=50)
    outputs = llm.generate(prompt_token_ids=full_token_ids, 
                          sampling_params=sampling_params)
    
    # Print results
    print("\nGenerated output:")
    print("-" * 80)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"{full_prompt}{generated_text}")
    print("-" * 80)
    
    print("\n✓ Successfully used external KV cache for inference!")


if __name__ == "__main__":
    main()

