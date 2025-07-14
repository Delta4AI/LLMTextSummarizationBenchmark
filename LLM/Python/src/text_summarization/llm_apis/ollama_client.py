import logging
from typing import Any

import ollama

from text_summarization.llm_apis.base_client import BaseClient
from text_summarization.config import OLLAMA_DEFAULT_PARAMS

logger = logging.getLogger(__name__)


class OllamaClient(BaseClient):
    def __init__(self, host: str = None):
        self.client = ollama.Client(host=host or self.get_dotenv_param("OLLAMA_BASE_URL"))

    def warmup(self, model_name: str, train_corpus: list[str] | None = None):
        try:
            logger.info(f"Warming up Ollama {model_name} model")
            response = self.client.generate(
                model=model_name,
                prompt="What is 2+2?",
                options={"temperature": 0.1, "num_predict": 10},
                context=None
            )
            logger.info(f"Warmup response: {response['response']}")
        except Exception as e:
            logger.error(f"Warmup failed: {e}")

    def summarize(self, text: str, model_name: str, system_prompt_override: str | None = None,
                  parameter_overrides: dict[str, Any] | None = None) -> str:
        try:

            logger.info(f"Making Ollama request with model {model_name}")
            response = self.client.generate(
                model=model_name,
                prompt=f"{self.system_prompt}\n\n{text}",
                options={**OLLAMA_DEFAULT_PARAMS, **(parameter_overrides or {})},
                context=None
            )

            if not response or 'response' not in response:
                raise ValueError("Invalid or no response from Ollama")

            logger.info(f"Successfully generated summary with Ollama {model_name}")
            return response["response"]

        except Exception as e:
            raise ValueError(f"Unexpected error in Ollama summarization: {e}")

    def test_token_size(self, model_name: str, text: str) -> int:
        try:
            resp = self.client.embed(model=model_name, input=text)
        except ollama._types.ResponseError:
            # fall back for models not supporting the embed endpoint
            resp = self.client.generate(model=model_name, prompt=text, options={"num_predict": 0})
        return resp["prompt_eval_count"]

