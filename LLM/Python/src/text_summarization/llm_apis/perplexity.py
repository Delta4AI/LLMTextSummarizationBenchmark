import logging

from text_summarization.llm_apis.base_client import BaseClient

logger = logging.getLogger(__name__)


class PerplexityClient(BaseClient):
    def __init__(self):
        pass

    def summarize(self, text: str, model_name: str, prompt: str) -> str:
        raise NotImplementedError("Perplexity API not yet implemented")
