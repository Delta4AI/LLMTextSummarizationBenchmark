import logging
from typing import Any

from text_summarization.llm_apis.base_client import BaseClient
from text_summarization.config import OPENAI_DEFAULT_PARAMS

from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenAIClient(BaseClient):
    def __init__(self, api_key: str = None):
        if api_key is None:
            api_key = self.get_dotenv_param("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

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

        logger.info(f"Successfully generated summary with OpenAI {model_name}")

        return response.output_text

if __name__ == "__main__":
    openai_client = OpenAIClient()
    test_title = "foo"
    test_abstract = "bar"
    print(openai_client.summarize(
        text=f"Title: {test_title}\n\nAbstract: \n{test_abstract}", model_name="gpt-4.1"))

