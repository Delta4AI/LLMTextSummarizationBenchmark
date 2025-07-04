import logging

from text_summarization.llm_apis.base_client import BaseClient

logger = logging.getLogger(__name__)


class AnthropicClient(BaseClient):
    def __init__(self):
        pass

    def summarize(self, text: str, model_name: str, prompt: str, fallback_summary: str = None) -> str:
        logger.error(f"Anthropic API not yet implemented")
        return fallback_summary
