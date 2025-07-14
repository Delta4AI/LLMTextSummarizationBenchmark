import logging
import os
from pathlib import Path
from typing import Any

from transformers import pipeline, AutoTokenizer

from text_summarization.llm_apis.base_client import BaseClient
from text_summarization.config import HUGGINGFACE_DEFAULT_PARAMS
from text_summarization.utilities import get_dotenv_param


logger = logging.getLogger(__name__)


class HuggingFaceClient(BaseClient):
    def __init__(self):
        init_hf_cache_dir()
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


def init_hf_cache_dir():
    hf_home = get_dotenv_param("HF_HOME")
    if hf_home:
        hf_path = Path(hf_home)
        if hf_path.is_dir():
            os.environ["HF_HOME"] = str(hf_path)
            logger.info(f"Set HF_HOME to: {hf_path}")
        else:
            logger.warning(f"HF_HOME directory does not exist: {hf_path}")
            try:
                hf_path.mkdir(parents=True, exist_ok=True)
                os.environ["HF_HOME"] = str(hf_path)
                logger.info(f"Created and set HF_HOME to: {hf_path}")
            except OSError as e:
                logger.error(f"Failed to create HF_HOME directory: {e}")
                logger.warning("Using default HF_HOME: ~/.cache/huggingface")
    else:
        logger.info("HF_HOME not configured, using default: ~/.cache/huggingface")
