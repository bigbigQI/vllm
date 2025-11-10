"""
Register the CustomFileKVConnector with vLLM.

This should be imported before starting vLLM to make the connector available.
"""

from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.logger import init_logger

logger = init_logger(__name__)


def register_custom_file_connector():
    """Register CustomFileKVConnector with vLLM's connector factory"""
    try:
        KVConnectorFactory.register_connector(
            name="CustomFileKVConnector",
            module_path="custom_kv_connector.custom_file_kv_connector",
            class_name="CustomFileKVConnector"
        )
        logger.info("✓ CustomFileKVConnector registered successfully")
    except Exception as e:
        logger.error(f"Failed to register CustomFileKVConnector: {e}")
        raise


# Auto-register when imported
register_custom_file_connector()

