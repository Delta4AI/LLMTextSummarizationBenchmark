from abc import ABC, abstractmethod
from typing import Optional

class BaseClient(ABC):
    @abstractmethod
    def summarize(self, text: str, model_name: str, prompt: str | None = None) -> str:
        pass