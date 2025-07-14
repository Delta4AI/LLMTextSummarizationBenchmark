import logging
from typing import Any
import time

from mistralai import Mistral

from text_summarization.llm_apis.base_client import BaseClient
from text_summarization.config import MISTRAL_DEFAULT_PARAMS

logger = logging.getLogger(__name__)


class MistralClient(BaseClient):
    def __init__(self, api_key: str = None):
        self.client = Mistral(api_key=api_key or self.get_dotenv_param("MISTRAL_API_KEY"))

    def summarize(self, text: str, model_name: str, system_prompt_override: str | None = None,
                  parameter_overrides: dict[str, Any] | None = None) -> str:
        logger.info(f"Making MistralAI request with model {model_name} after sleeping for 1.5 seconds")
        time.sleep(1.5)

        try:
            response = self.client.chat.complete(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": text
                    },
                    {
                        "role": "system",
                        "content": system_prompt_override if system_prompt_override is not None else self.system_prompt
                    }
                ],
                **MISTRAL_DEFAULT_PARAMS,
                **(parameter_overrides or {})
            )
            return response.choices[0].message.content
        except Exception as e:
            raise ValueError(f"Invalid or no response from MistralAI: {e}")
