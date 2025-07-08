import logging
from typing import Any

from text_summarization.llm_apis.base_client import BaseClient
from text_summarization.config import ANTHROPIC_DEFAULT_PARAMS

import anthropic

logger = logging.getLogger(__name__)


class AnthropicClient(BaseClient):
    def __init__(self, api_key: str = None):
        if api_key is None:
            api_key = self.get_dotenv_param("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key)

    def summarize(self, text: str, model_name: str, system_prompt_override: str | None = None,
                  parameter_overrides: dict[str, Any] | None = None) -> str:
        logger.info(f"Making Anthropic request with model {model_name}")
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

        logger.info(f"Successfully generated summary with Anthropic {model_name}")

        return response.content[0].text
