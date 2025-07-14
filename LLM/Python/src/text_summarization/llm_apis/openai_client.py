import logging
from typing import Any

from openai import OpenAI
import tiktoken

from text_summarization.llm_apis.base_client import BaseClient
from text_summarization.config import OPENAI_DEFAULT_PARAMS
from text_summarization.utilities import get_dotenv_param

logger = logging.getLogger(__name__)


class OpenAIClient(BaseClient):
    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key or get_dotenv_param("OPENAI_API_KEY"))

    def summarize(self, text: str, model_name: str, system_prompt_override: str | None = None,
                  parameter_overrides: dict[str, Any] | None = None) -> str:

        logger.info(f"Making OpenAI request with model {model_name}")
        response = self.client.responses.create(
            model=model_name,
            instructions=system_prompt_override if system_prompt_override is not None else self.system_prompt,
            input=text,
            **OPENAI_DEFAULT_PARAMS,
            **(parameter_overrides or {})
        )

        if not response or not hasattr(response, 'output_text'):
            raise ValueError("Invalid or no response from OpenAI")

        return response.output_text

    def test_token_size(self, model_name: str, text: str) -> int:
        try:
            tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
            logger.warning(f"Model {model_name} not found in tiktoken encoding list. "
                           f"Using cl100k_base model as fallback.")
            tokenizer = tiktoken.get_encoding("cl100k_base")

        tokens = tokenizer.encode(text)
        return len(tokens)


if __name__ == "__main__":
    openai_client = OpenAIClient()
    test_title = "foo"
    test_abstract = "bar"
    print(openai_client.summarize(
        text=f"Title: {test_title}\n\nAbstract: \n{test_abstract}", model_name="gpt-4.1"))

