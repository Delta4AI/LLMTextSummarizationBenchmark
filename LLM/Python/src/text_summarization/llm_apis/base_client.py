from abc import ABC, abstractmethod
from typing import Any

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
        raise NotImplementedError

    def test_token_size(self, model_name: str, text: str) -> int:
        """Optional test token size method. Override if needed"""
        raise NotImplementedError
