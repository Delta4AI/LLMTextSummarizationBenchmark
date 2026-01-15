import json
import logging
from typing import Any

from mistralai import Mistral, ThinkChunk, TextChunk, ChatCompletionChoice
from mistralai.models import SDKError

from utilities import get_dotenv_var, get_project_root
from llm_apis.base_client import SummaryClient
from llm_apis.config import MISTRAL_DEFAULT_PARAMS
from llm_apis.exceptions import UnknownResponse

logger = logging.getLogger(__name__)


class MistralSummaryClient(SummaryClient):
    def __init__(self, api_key: str = None):
        self.client = Mistral(api_key=api_key or get_dotenv_var("MISTRAL_API_KEY"))

    def summarize(self, text: str, model_name: str, system_prompt_override: str | None = None,
                  parameter_overrides: dict[str, Any] | None = None,
                  tokenizer_overrides: dict[str, Any] | None = None) -> tuple[str, int, int]:
        logger.info(f"Making MistralAI request with model {model_name}")

        max_retries = 5

        for attempt in range(max_retries + 1):

            try:
                response = self.client.chat.complete(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": text
                        },
                        {
                            "role": "system",
                            "content": system_prompt_override or self.text_summarization_system_prompt
                        }
                    ],
                    **MISTRAL_DEFAULT_PARAMS,
                    **(parameter_overrides or {})
                )

                return self._handle_response(response)

            except SDKError as e:
                if (e.status_code == 429 or 500 <= e.status_code < 600) and attempt < max_retries:
                    logger.warning(f"MistralAI rate limit hit. Retry attempt {attempt + 1}/{max_retries + 1} ..")
                    continue
                else:
                    logger.error(f"MistralAI request failed: {e}")
                    raise UnknownResponse(e)
            except Exception as e:
                raise UnknownResponse(f"Invalid or no response from MistralAI: {e}")

        raise UnknownResponse("MistralAI request failed after max retries")

    @staticmethod
    def _handle_response(response):
        if len(response.choices) > 1:
            logger.warning(f"MistralAI returned multiple responses: {response.choices}. Only using first one!")

        _input_tokens = response.usage.prompt_tokens
        _output_tokens = response.usage.completion_tokens
        _choices = response.choices

        _choice = _choices[0]
        _response = _choice.message.content

        if isinstance(_response, str):
            return _response, _input_tokens, _output_tokens

        elif isinstance(_response, ChatCompletionChoice):
            return _response.message.content, _input_tokens, _output_tokens

        # Reasoning model response with thinking + answer
        elif (isinstance(_response, list)
              and len(_response) >= 2
              and isinstance(_response[0], ThinkChunk)
              and isinstance(_response[1], TextChunk)):

            _reasoning = " ".join(_.text for _ in _response[0].thinking)
            _answer = _response[1].text
            return f"{_reasoning}\n\n{_answer}", _input_tokens, _output_tokens
        else:
            logger.error(f"Unknown response format: {type(_response)} - {_response}")
            raise UnknownResponse

    def submit_batch(self, model_name: str, system_prompt_override: str | None = None,
                     parameter_overrides: dict[str, Any] | None = None, prompts: list[str] = None,
                     custom_ids: list[str] = None, papers_hash: str = None) -> str:
        logger.info("Creating .jsonl file ..")

        jsonl_lines = []
        if custom_ids and len(prompts) != len(custom_ids):
            logger.error(f"Number of custom ids ({len(custom_ids)}) does not match number of prompts ({len(prompts)}).")
            logger.error(f"Will enumerate prompt custom ids")
            custom_ids = [str(_) for _ in range(len(prompts))]
        if not custom_ids:
            custom_ids = [str(_) for _ in range(len(prompts))]

        for prompt, custom_id in zip(prompts, custom_ids):
            params = MISTRAL_DEFAULT_PARAMS.copy()
            if parameter_overrides:
                params.update(parameter_overrides)

            request = {
                "custom_id": custom_id,
                "body": {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content":  system_prompt_override or self.text_summarization_system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    **params
                }
            }

            jsonl_lines.append(json.dumps(request))

        output_file = get_project_root() / "Output" / "llm_apis" / f"{model_name}_batch.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, mode="w", encoding="utf-8") as f:
            f.write("\n".join(jsonl_lines))

        logger.info(f"Uploading .jsonl file with {len(prompts)} prompts ..")
        with open(output_file, "rb") as f:
            batch_data = self.client.files.upload(
                file={
                    "file_name": output_file.name,
                    "content": f
                },
                purpose="batch"
            )

        logger.info(f"Creating batch job with id {batch_data.id}..")
        created_job = self.client.batch.jobs.create(
            input_files=[batch_data.id],
            model=model_name,
            endpoint="/v1/chat/completions",
            metadata={
                "job_type": "summarization",
                "model": model_name,
                "num_prompts": len(prompts),
                "papers_hash": papers_hash,
            }
        )

        retrieved_job = self.client.batch.jobs.get(job_id=created_job.id)
        logger.info("Job created with id: " + retrieved_job.id)
        logger.info("Note the ID and check later with check_batch")

        return retrieved_job.id

    def get_batch(self, job_id: str) -> dict:
        logger.info(f"Retrieving batch job {job_id}...")
        batch_job = self.client.batch.jobs.get(job_id=job_id)
        results, errors = None, None

        logger.info(f"Status: {batch_job.status}")
        logger.info(f"Total requests: {batch_job.total_requests}")
        logger.info(f"Successful requests: {batch_job.succeeded_requests}")
        logger.info(f"Failed requests: {batch_job.failed_requests}")

        if batch_job.total_requests > 0:
            percent_done = round(
                (batch_job.succeeded_requests + batch_job.failed_requests) / batch_job.total_requests * 100,
                2
            )
            logger.info(f"Percent done: {percent_done}%")

        if batch_job.status == "SUCCESS":
            output_dir = get_project_root() / "Output" / "llm_apis" / "batch_results"
            output_dir.mkdir(parents=True, exist_ok=True)

            if batch_job.output_file:
                out_file = self.client.files.download(file_id=batch_job.output_file)
                out_path = self._save_batch_jsonl(job_id=job_id, response=out_file, is_error_file=False)
                results = self._load_batch_jsonl(job_id=job_id, is_error_file=False)
                logger.info(f"Output file saved to {out_path}")

            if batch_job.error_file:
                err_file = self.client.files.download(file_id=batch_job.error_file)
                err_path = self._save_batch_jsonl(job_id=job_id, response=err_file, is_error_file=True)
                errors = self._load_batch_jsonl(job_id=job_id, is_error_file=True)
                logger.info(f"Error file saved to {err_path}")

            if batch_job.completed_at and batch_job.created_at:
                duration = batch_job.completed_at - batch_job.created_at
                logger.info(f"Job duration: {duration} seconds")

        elif batch_job.status in ["QUEUED", "RUNNING"]:
            logger.info("Job is still processing. Check again later.")
        elif batch_job.status == "FAILED":
            logger.error("Job failed!")

        return {"batch_job": batch_job, "results": results, "errors": errors}



if __name__ == "__main__":
    mistral_client = MistralSummaryClient()
    resp = mistral_client.summarize(text="Title: foo\n\nAbstract: bar", model_name="mistral-small-2506")
    print(resp)