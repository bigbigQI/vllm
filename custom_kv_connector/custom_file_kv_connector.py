# SPDX-License-Identifier: Apache-2.0
"""
Custom File KV Connector for vLLM

This connector loads pre-computed KV cache from local files to skip
the computation of cached tokens during inference.

File format:
{
    'prompt_token_ids': List[int],
    'keys': Tensor(num_layers, batch_size, num_heads, seq_len, head_dim),
    'values': Tensor(num_layers, batch_size, num_heads, seq_len, head_dim)
}
"""

import hashlib
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class LoadRequestMeta:
    """Metadata for a request that needs KV loading"""
    request_id: str
    num_external_tokens: int
    block_ids: list[int]
    kv_file_path: str


@dataclass
class CustomFileConnectorMetadata(KVConnectorMetadata):
    """Metadata passed from scheduler to worker"""
    load_requests: list[LoadRequestMeta]

    def __init__(self):
        self.load_requests = []


class CustomFileKVConnector(KVConnectorBase_V1):
    """
    Custom File KV Connector that loads pre-computed KV cache from disk.
    
    This connector enables:
    1. Loading partial prompt KV cache from files
    2. Skipping computation for cached tokens
    3. Only computing the remaining uncached tokens
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        
        self._block_size = vllm_config.cache_config.block_size
        self._num_layers = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self._num_kv_heads = vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        )
        self._head_size = vllm_config.model_config.get_head_size()
        
        # Scheduler side: track requests needing external KV
        self._requests_need_load: dict[str, tuple[str, int]] = {}
        # key: request_id, value: (kv_file_path, num_cached_tokens)
        
        # Scheduler side: track block allocations
        self._request_block_allocations: dict[str, list[int]] = {}
        
        # Worker side: cache loaded KV to avoid redundant disk reads
        self._loaded_kv_cache: dict[str, dict] = {}
        
        # Get KV cache directory from config
        transfer_config = vllm_config.kv_transfer_config
        self._kv_cache_dir = transfer_config.get_from_extra_config(
            "kv_cache_dir", "./kv_cache"
        )
        Path(self._kv_cache_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"CustomFileKVConnector initialized (role={role.name}, "
            f"cache_dir={self._kv_cache_dir}, block_size={self._block_size})"
        )

    # ========================================
    # Scheduler Side Methods
    # ========================================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[Optional[int], bool]:
        """
        Check if there's a matching external KV cache for this request.
        
        Returns:
            (num_tokens_to_load, is_async): Number of tokens that can be loaded
                from external cache and whether loading is async.
        """
        # Only check on first call (when no tokens computed yet)
        if num_computed_tokens > 0:
            return 0, False
        
        # Check if request already has KV transfer params specifying cache file
        kv_file_path = None
        if request.kv_transfer_params:
            kv_file_path = request.kv_transfer_params.get("kv_cache_file")
        
        # Otherwise, try to find cache file automatically
        if kv_file_path is None:
            kv_file_path = self._find_kv_cache_file(request)
        
        if kv_file_path is None:
            return 0, False
        
        # Load KV cache file to get available tokens
        try:
            loaded = torch.load(kv_file_path, map_location='cpu')
            cached_token_ids = loaded['prompt_token_ids']
            
            # Check if request prompt matches cached tokens
            request_token_ids = request.prompt_token_ids
            if request_token_ids is None:
                return 0, False
            
            num_matched = self._count_matched_prefix(
                cached_token_ids, request_token_ids
            )
            
            if num_matched == 0:
                return 0, False
            
            # Align to block boundary
            num_matched_aligned = (num_matched // self._block_size) * self._block_size
            
            if num_matched_aligned > 0:
                # Record this request for loading
                self._requests_need_load[request.request_id] = (
                    kv_file_path, num_matched_aligned
                )
                logger.info(
                    f"Request {request.request_id}: KV cache hit! "
                    f"{num_matched_aligned}/{len(request_token_ids)} tokens cached "
                    f"(file: {Path(kv_file_path).name})"
                )
                return num_matched_aligned, False  # Synchronous load
            
        except Exception as e:
            logger.warning(f"Failed to load KV cache from {kv_file_path}: {e}")
            return 0, False
        
        return 0, False

    def update_state_after_alloc(
        self, 
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int
    ):
        """
        Update state after scheduler allocates blocks for external KV.
        Record the block IDs so worker knows where to inject KV.
        """
        if num_external_tokens > 0 and request.request_id in self._requests_need_load:
            # Extract block IDs from KVCacheBlocks
            # blocks.blocks is a tuple of lists, one per KV cache group
            # Each element in the list is a KVCacheBlock object with block_id attribute
            if blocks and len(blocks.blocks) > 0:
                # Use blocks from first KV cache group
                kv_cache_blocks = blocks.blocks[0]
                # Extract integer block IDs from KVCacheBlock objects
                block_ids = [block.block_id for block in kv_cache_blocks]
                self._request_block_allocations[request.request_id] = block_ids
                logger.debug(
                    f"Request {request.request_id}: allocated {len(block_ids)} blocks "
                    f"for {num_external_tokens} external tokens (IDs: {block_ids[:5]}...)"
                )

    def build_connector_meta(
        self, 
        scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """
        Build metadata to pass to workers.
        This tells workers which requests need KV loading and where to inject.
        """
        metadata = CustomFileConnectorMetadata()
        
        # Only process new requests being scheduled
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            
            if req_id in self._requests_need_load:
                kv_file_path, num_tokens = self._requests_need_load[req_id]
                
                # Get block IDs from allocation
                block_ids = self._request_block_allocations.get(req_id, [])
                
                if block_ids:
                    metadata.load_requests.append(LoadRequestMeta(
                        request_id=req_id,
                        num_external_tokens=num_tokens,
                        block_ids=block_ids[:],  # Copy the list
                        kv_file_path=kv_file_path
                    ))
                    logger.debug(
                        f"Added {req_id} to load queue: {num_tokens} tokens, "
                        f"{len(block_ids)} blocks"
                    )
        
        # Clean up processed requests
        for load_req in metadata.load_requests:
            self._requests_need_load.pop(load_req.request_id, None)
            self._request_block_allocations.pop(load_req.request_id, None)
        
        if metadata.load_requests:
            logger.info(
                f"Connector metadata: {len(metadata.load_requests)} requests "
                f"will load external KV"
            )
        
        return metadata

    # ========================================
    # Worker Side Methods
    # ========================================

    def start_load_kv(
        self, 
        forward_context: "ForwardContext",
        **kwargs: Any
    ) -> None:
        """
        Load external KV cache into vLLM's paged buffer before forward pass.
        Called at the start of model execution.
        """
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, CustomFileConnectorMetadata):
            return
        
        if not metadata.load_requests:
            return
        
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return
        
        logger.info(f"Loading KV cache for {len(metadata.load_requests)} requests")
        
        # Load KV for each request
        for load_req in metadata.load_requests:
            try:
                self._load_kv_for_request(load_req, forward_context)
            except Exception as e:
                logger.error(
                    f"Failed to load KV for request {load_req.request_id}: {e}",
                    exc_info=True
                )

    def _load_kv_for_request(
        self,
        load_req: LoadRequestMeta,
        forward_context: "ForwardContext"
    ):
        """Load KV cache for a single request"""
        # Load file if not already cached
        if load_req.kv_file_path not in self._loaded_kv_cache:
            try:
                loaded = torch.load(load_req.kv_file_path, map_location='cuda')
                self._loaded_kv_cache[load_req.kv_file_path] = loaded
                logger.debug(f"Loaded KV file: {load_req.kv_file_path}")
            except Exception as e:
                logger.error(f"Failed to load KV file: {e}")
                return
        
        loaded = self._loaded_kv_cache[load_req.kv_file_path]
        keys = loaded['keys']  # (num_layers, batch, num_heads, seq_len, head_dim)
        values = loaded['values']
        
        num_file_layers = keys.shape[0]
        num_tokens_available = keys.shape[3]
        
        # Validate dimensions
        if num_file_layers < self._num_layers:
            logger.warning(
                f"KV file has {num_file_layers} layers but model has "
                f"{self._num_layers} layers"
            )
        
        if num_tokens_available < load_req.num_external_tokens:
            logger.warning(
                f"KV file has only {num_tokens_available} tokens but "
                f"{load_req.num_external_tokens} requested"
            )
            load_req.num_external_tokens = num_tokens_available
        
        # Extract only the tokens we need (batch_size=0, first N tokens)
        keys_to_inject = keys[:, 0, :, :load_req.num_external_tokens, :]
        values_to_inject = values[:, 0, :, :load_req.num_external_tokens, :]
        
        # Inject into each attention layer
        layer_idx = 0
        for layer_name, layer in forward_context.no_compile_layers.items():
            # Only process attention layers (those with kv_cache attribute)
            kv_cache_attr = getattr(layer, 'kv_cache', None)
            if kv_cache_attr is None:
                continue
            
            if layer_idx >= num_file_layers:
                break
            
            # Get the paged KV cache for this layer
            kv_cache = kv_cache_attr[forward_context.virtual_engine]
            
            # Inject KV into paged format
            self._inject_continuous_kv_to_paged(
                kv_cache=kv_cache,
                keys_continuous=keys_to_inject[layer_idx],
                values_continuous=values_to_inject[layer_idx],
                block_ids=load_req.block_ids,
                num_tokens=load_req.num_external_tokens
            )
            
            layer_idx += 1
        
        logger.info(
            f"✓ Loaded {load_req.num_external_tokens} tokens across "
            f"{layer_idx} layers for request {load_req.request_id}"
        )

    def _inject_continuous_kv_to_paged(
        self,
        kv_cache: torch.Tensor,
        keys_continuous: torch.Tensor,  # (num_heads, seq_len, head_dim)
        values_continuous: torch.Tensor,  # (num_heads, seq_len, head_dim)
        block_ids: list[int],
        num_tokens: int
    ):
        """
        Convert continuous KV cache to paged format and inject into vLLM's buffer.
        
        Args:
            kv_cache: vLLM's paged KV cache tensor
            keys_continuous: Continuous keys (num_heads, seq_len, head_dim)
            values_continuous: Continuous values (num_heads, seq_len, head_dim)
            block_ids: Physical block IDs to fill (list of integers)
            num_tokens: Number of tokens to inject
        """
        # Validate block_ids are integers
        if not all(isinstance(bid, int) for bid in block_ids):
            logger.error(
                f"block_ids contains non-integer values: {[type(bid) for bid in block_ids[:5]]}"
            )
            raise TypeError(
                f"block_ids must be a list of integers, got: {type(block_ids[0])}"
            )
        
        logger.debug(
            f"Injecting KV: cache_shape={kv_cache.shape}, "
            f"keys_shape={keys_continuous.shape}, num_tokens={num_tokens}, "
            f"num_blocks={len(block_ids)}, block_ids={block_ids[:5]}..."
        )
        
        # Determine KV cache format
        if kv_cache.shape[0] == 2:
            # Flash Attention format: [2, num_blocks, block_size, num_heads, head_dim]
            key_cache = kv_cache[0]  # [num_blocks, block_size, num_heads, head_dim]
            value_cache = kv_cache[1]
            
            num_blocks = len(block_ids)
            
            for block_idx, block_id in enumerate(block_ids):
                # Calculate token range for this block
                start_token = block_idx * self._block_size
                end_token = min(start_token + self._block_size, num_tokens)
                block_num_tokens = end_token - start_token
                
                if block_num_tokens <= 0:
                    break
                
                # Extract KV for this block
                # (num_heads, block_num_tokens, head_dim) -> (block_num_tokens, num_heads, head_dim)
                block_keys = keys_continuous[:, start_token:end_token, :].permute(1, 0, 2)
                block_values = values_continuous[:, start_token:end_token, :].permute(1, 0, 2)
                
                # Inject into paged cache at the specified block
                key_cache[block_id, :block_num_tokens, :, :] = block_keys
                value_cache[block_id, :block_num_tokens, :, :] = block_values
            
            logger.debug(
                f"Injected {num_tokens} tokens into {num_blocks} blocks "
                f"(Flash Attention format)"
            )
            
        elif kv_cache.numel() > 0 and kv_cache.shape[1] == 2:
            # FlashInfer format: [num_blocks, 2, block_size, num_heads, head_dim]
            num_blocks = len(block_ids)
            
            for block_idx, block_id in enumerate(block_ids):
                start_token = block_idx * self._block_size
                end_token = min(start_token + self._block_size, num_tokens)
                block_num_tokens = end_token - start_token
                
                if block_num_tokens <= 0:
                    break
                
                # Extract and permute
                block_keys = keys_continuous[:, start_token:end_token, :].permute(1, 0, 2)
                block_values = values_continuous[:, start_token:end_token, :].permute(1, 0, 2)
                
                # Inject
                kv_cache[block_id, 0, :block_num_tokens, :, :] = block_keys
                kv_cache[block_id, 1, :block_num_tokens, :, :] = block_values
            
            logger.debug(
                f"Injected {num_tokens} tokens into {num_blocks} blocks "
                f"(FlashInfer format)"
            )
        else:
            logger.warning(
                f"Unsupported KV cache format: shape={kv_cache.shape}. "
                f"Skipping injection."
            )

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Synchronous loading - nothing to wait for"""
        pass

    def save_kv_layer(
        self, 
        layer_name: str, 
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any
    ) -> None:
        """This connector only loads KV, doesn't save"""
        pass

    def wait_for_save(self):
        """No saving - return immediately"""
        pass

    # ========================================
    # Helper Methods
    # ========================================

    def _find_kv_cache_file(self, request: "Request") -> Optional[str]:
        """
        Find KV cache file for a request.
        
        Strategies:
        1. Use prompt hash to find matching file
        2. Fall back to default file if exists
        """
        if request.prompt_token_ids is None:
            return None
        
        # Strategy 1: Hash-based matching
        prompt_str = ','.join(map(str, request.prompt_token_ids[:100]))
        prompt_hash = hashlib.md5(prompt_str.encode()).hexdigest()[:16]
        
        hash_file = Path(self._kv_cache_dir) / f"kv_cache_{prompt_hash}.pt"
        if hash_file.exists():
            logger.debug(f"Found KV cache by hash: {hash_file}")
            return str(hash_file)
        
        # Strategy 2: Default file
        default_file = Path(self._kv_cache_dir) / "kv_cache.pt"
        if default_file.exists():
            logger.debug(f"Using default KV cache: {default_file}")
            return str(default_file)
        
        return None

    def _count_matched_prefix(
        self, 
        cached_tokens: list, 
        request_tokens: list
    ) -> int:
        """Count how many prefix tokens match"""
        min_len = min(len(cached_tokens), len(request_tokens))
        for i in range(min_len):
            if cached_tokens[i] != request_tokens[i]:
                return i
        return min_len

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: "VllmConfig") -> Optional[str]:
        """This connector works with any KV cache layout"""
        return None

