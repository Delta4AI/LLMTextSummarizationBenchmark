import logging
from typing import Any

from text_summarization.llm_apis.base_client import BaseClient

logger = logging.getLogger(__name__)


class AnthropicClient(BaseClient):
    def __init__(self):
        pass

    def summarize(self, text: str, model_name: str, prompt: str,
                  parameters: dict[str, Any] | None = None) -> str:
        raise NotImplementedError("Anthropic API not yet implemented")
