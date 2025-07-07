import logging
from doctest import UnexpectedException
from xmlrpc.client import ResponseError

import ollama

from text_summarization.llm_apis.base_client import BaseClient

logger = logging.getLogger(__name__)


def extract_final_answer(response_text: str) -> str:
    """Extract the final answer from a response that may contain thinking tags."""

    if '</think>' in response_text:
        parts = response_text.split('</think>')
        if len(parts) > 1:
            final_answer = parts[-1].strip()
            return final_answer

    return response_text.strip()


class OllamaClient(BaseClient):
    def __init__(self, host='http://comma5.duckdns.org:11434'):
        self.host = host
        self.client = ollama.Client(host=self.host)

    def warmup(self, model_name: str):
        try:
            response = self.client.generate(
                model=model_name,
                prompt="What is 2+2?",
                options={"temperature": 0.1, "num_predict": 10}
            )
            logger.info(f"Warmup response: ${response['response']}")
        except Exception as e:
            logger.error(f"Warmup failed: {e}")

    def summarize(self, text: str, model_name: str, prompt: str | None = None) -> str:
        """
        Ollama API summarization using official ollama Python library.

        Args:
            text: Input text to summarize
            model_name: Ollama model name (e.g., "granite3.1-dense:8b")
            prompt: Custom prompt template with placeholder for text

        Returns:
            Generated summary or fallback_summary on error
        """
        try:
            formatted_prompt = prompt.format(text=text[:3000])

            logger.info(f"Making Ollama request with model {model_name}")
            response = self.client.generate(
                model=model_name,
                prompt=formatted_prompt,
                options={
                    'temperature': 0.3,
                    'top_k': 40,
                    'top_p': 0.9,
                    # 'num_predict': 200,  # TODO: update dynamically from min_words and max_words (depends on model ..)
                    # 'stop': ['\n\n', '---', 'References:', 'Bibliography:']
                }
            )

            if not response or 'response' not in response:
                raise ResponseError("Invalid or no response from Ollama")

            logger.info(f"Successfully generated summary with Ollama {model_name}")
            return response["response"]

        except Exception as e:
            raise UnexpectedException(f"Unexpected error in Ollama summarization: {e}")
