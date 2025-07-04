from abc import ABC, abstractmethod


class BaseClient(ABC):
    @abstractmethod
    def summarize(self, text: str, model_name: str, prompt: str, fallback_summary: str = None) -> str:
        pass