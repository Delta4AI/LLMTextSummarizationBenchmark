import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

from text_summarization.config import SYSTEM_PROMPT, MIN_WORDS, MAX_WORDS


class BaseClient(ABC):
    system_prompt = SYSTEM_PROMPT
    min_words = MIN_WORDS
    max_words = MAX_WORDS

    @abstractmethod
    def summarize(self, text: str, model_name: str, system_prompt_override: str | None = None,
                  parameter_overrides: dict[str, Any] | None = None) -> str:
        pass

    def warmup(self, model_name: str, train_corpus: list[str] | None) -> None:
        """Optional warmup method. Override if needed"""
        pass

    @staticmethod
    def get_dotenv_param(param: str) -> str | None:
        env_file = Path(__file__).parents[4] / "Resources" / ".env"
        load_dotenv(env_file)
        return os.getenv(param)
