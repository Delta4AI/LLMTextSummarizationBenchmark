import logging
from typing import Any
import time

from text_summarization.llm_apis.base_client import BaseClient
from text_summarization.config import ANTHROPIC_DEFAULT_PARAMS

import anthropic

logger = logging.getLogger(__name__)


class AnthropicClient(BaseClient):
    def __init__(self, api_key: str = None):
        self.client = anthropic.Anthropic(api_key=api_key or self.get_dotenv_param("ANTHROPIC_API_KEY"))

    def summarize(self, text: str, model_name: str, system_prompt_override: str | None = None,
                  parameter_overrides: dict[str, Any] | None = None) -> str:
        logger.info(f"Making Anthropic request with model {model_name} after sleeping for 1.5 seconds")
        time.sleep(1.5)

        response = self.client.messages.create(
            model=model_name,
            system=system_prompt_override if system_prompt_override is not None else self.system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": text
                }
            ],
            **ANTHROPIC_DEFAULT_PARAMS,
            **(parameter_overrides or {})
        )

        if not response or not hasattr(response, "content"):
            raise ValueError("Invalid or no response from Anthropic")

        return response.content[0].text
