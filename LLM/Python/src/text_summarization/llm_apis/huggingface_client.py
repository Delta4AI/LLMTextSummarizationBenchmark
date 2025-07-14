import logging
from typing import Any

from text_summarization.llm_apis.base_client import BaseClient
from text_summarization.config import HUGGINGFACE_DEFAULT_PARAMS

from transformers import pipeline, AutoTokenizer

logger = logging.getLogger(__name__)


class HuggingFaceClient(BaseClient):
    def __init__(self):
        self.summarizer = None

    def warmup(self, model_name: str, train_corpus: list[str] | None = None):
        self.summarizer = pipeline(task="summarization", model=model_name)

    def summarize(self, text: str, model_name: str, system_prompt_override: str | None = None,
                  parameter_overrides: dict[str, Any] | None = None) -> str:
        try:
            logger.info(f"Making Huggingface summarizer pipeline request with model {model_name}")
            response = self.summarizer(
                text,
                **HUGGINGFACE_DEFAULT_PARAMS,
                **(system_prompt_override or {})
            )

            return response[0]['summary_text']
        except Exception as e:
            raise ValueError(f"Huggingface summarizer pipeline request failed: {e}")

    def test_token_size(self, model_name: str, text: str) -> int:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokens = tokenizer.encode(text)
        return len(tokens)
