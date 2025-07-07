from abc import ABC, abstractmethod

class BaseClient(ABC):

    @abstractmethod
    def summarize(self, text: str, model_name: str, prompt: str | None = None) -> str:
        pass

    def warmup(self, model_name: str) -> None:
        """Optional warmup method. Override if needed"""
        pass