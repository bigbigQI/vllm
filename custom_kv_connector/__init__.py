"""
Custom KV Connector for vLLM

This package provides a custom KV connector that loads pre-computed KV cache
from local files, enabling faster inference by skipping computation for cached tokens.
"""

__version__ = "0.1.0"

from custom_kv_connector.custom_file_kv_connector import (
    CustomFileKVConnector,
    CustomFileConnectorMetadata,
    LoadRequestMeta,
)

__all__ = [
    "CustomFileKVConnector",
    "CustomFileConnectorMetadata",
    "LoadRequestMeta",
]

