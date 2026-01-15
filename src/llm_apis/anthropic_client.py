import logging
from typing import Any

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from utilities import get_dotenv_var
from llm_apis.base_client import SummaryClient
from llm_apis.config import ANTHROPIC_DEFAULT_PARAMS
from llm_apis.exceptions import RefusalError, NoContentError

logger = logging.getLogger(__name__)


class AnthropicSummaryClient(SummaryClient):
    def __init__(self, api_key: str = None):
        self.client = anthropic.Anthropic(api_key=api_key or get_dotenv_var("ANTHROPIC_API_KEY"))

    def summarize(self, text: str, model_name: str, system_prompt_override: str | None = None,
                  parameter_overrides: dict[str, Any] | None = None,
                  tokenizer_overrides: dict[str, Any] | None = None) -> tuple[str, int, int]:
        logger.info(f"Making Anthropic request with model {model_name}")

        response = self.client.messages.create(
            model=model_name,
            system=system_prompt_override or self.text_summarization_system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": text
                }
            ],
            **ANTHROPIC_DEFAULT_PARAMS,
            **(parameter_overrides or {})
        )

        if not response:
            raise RuntimeError(f"No response from Anthropic API for model {model_name}")

        if response.stop_reason == "refusal":
            raise RefusalError(f"Refused to respond to request from Anthropic API for model {model_name}. "
                               f"Raw response: {response}")

        if not response.content:
            raise NoContentError(f"No content in response from Anthropic API for model {model_name}. "
                                 f"Raw response: {response}")

        return response.content[0].text, response.usage.input_tokens, response.usage.output_tokens


    def submit_batch(self, model_name: str, system_prompt_override: str | None = None,
                     parameter_overrides: dict[str, Any] | None = None, prompts: list[str] = None,
                     custom_ids: list[str] = None, papers_hash: str = None) -> str:
        logger.info(f"Creating batch with {len(prompts)} prompts...")

        if custom_ids and len(prompts) != len(custom_ids):
            logger.error(f"Number of custom ids ({len(custom_ids)}) does not match number of prompts ({len(prompts)}).")
            logger.error(f"Will enumerate prompt custom ids")
            custom_ids = [str(_) for _ in range(len(prompts))]
        if not custom_ids:
            custom_ids = [str(_) for _ in range(len(prompts))]

        params = ANTHROPIC_DEFAULT_PARAMS.copy()
        if parameter_overrides:
            params.update(parameter_overrides)

        requests = []
        for prompt, custom_id in zip(prompts, custom_ids):
            request = Request(
                custom_id=custom_id,
                params=MessageCreateParamsNonStreaming(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    system=system_prompt_override or self.text_summarization_system_prompt,
                    **params
                )
            )
            requests.append(request)

        logger.info(f"Submitting batch job...")
        created_batch = self.client.messages.batches.create(requests=requests)

        logger.info(f"Batch created with id: {created_batch.id}")
        logger.info("Note the ID and check later with get_batch")

        return created_batch.id


    def get_batch(self, job_id: str) -> dict:
        logger.info(f"Retrieving batch job {job_id}...")
        batch = self.client.messages.batches.retrieve(job_id)
        results, errors = None, None

        logger.info(f"Status: {batch.processing_status}")
        logger.info(f"Total requests: {batch.request_counts.total}")
        logger.info(f"Successful requests: {batch.request_counts.succeeded}")
        logger.info(f"Failed requests: {batch.request_counts.errored}")

        if batch.request_counts.total > 0:
            percent_done = round(
                (batch.request_counts.succeeded + batch.request_counts.errored) / batch.request_counts.total * 100,
                2
            )
            logger.info(f"Percent done: {percent_done}%")

        if batch.processing_status == "ended":
            logger.info("Batch completed. Retrieving results...")

            results_list = []
            errors_list = []

            for result in self.client.messages.batches.results(job_id):
                if result.result.type == "succeeded":
                    results_list.append({
                        "custom_id": result.custom_id,
                        "result": result.result.message
                    })
                elif result.result.type == "errored":
                    errors_list.append({
                        "custom_id": result.custom_id,
                        "error": result.result.error
                    })

            results = results_list if results_list else None
            errors = errors_list if errors_list else None

            logger.info(f"Retrieved {len(results_list)} successful results and {len(errors_list)} errors")

            if batch.ended_at and batch.created_at:
                duration = (batch.ended_at - batch.created_at).total_seconds()
                logger.info(f"Job duration: {duration} seconds")

        elif batch.processing_status == "in_progress":
            logger.info("Batch is still processing. Check again later.")
        elif batch.processing_status == "canceling":
            logger.info("Batch is being canceled...")
        elif batch.processing_status == "canceled":
            logger.warning("Batch was canceled!")

        return {"batch_job": batch, "results": results, "errors": errors}


if __name__ == "__main__":
    anthropic_client = AnthropicSummaryClient()
    resp = anthropic_client.summarize(text="Title: foo\n\nAbstract: bar", model_name="claude-3-5-haiku-20241022")
    print(resp)