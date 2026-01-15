import logging
from pathlib import Path
from typing import Any, Type
import os

from openai import OpenAI
from openai.types.batch import Batch
from pydantic import BaseModel
import tiktoken

from utilities import get_dotenv_var
from llm_apis.base_client import SummaryClient, StructuredResponseClient, BatchStatus
from llm_apis.config import OPENAI_DEFAULT_PARAMS, OPENAI_GPT_5_DEFAULT_PARAMS


logger = logging.getLogger(__name__)


class OpenAISummaryClient(SummaryClient):
    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key or get_dotenv_var("OPENAI_API_KEY"))

    def summarize(self, text: str, model_name: str, system_prompt_override: str | None = None,
                  parameter_overrides: dict[str, Any] | None = None,
                  tokenizer_overrides: dict[str, Any] | None = None) -> tuple[str, int, int]:

        logger.info(f"Making OpenAI request with model {model_name}")
        default_params = OPENAI_GPT_5_DEFAULT_PARAMS if model_name.startswith("gpt-5") else OPENAI_DEFAULT_PARAMS
        response = self.client.responses.create(
            model=model_name,
            instructions=system_prompt_override or self.text_summarization_system_prompt,
            input=text,
            **default_params,
            **(parameter_overrides or {})
        )

        if not response or not hasattr(response, 'output_text'):
            raise ValueError("Invalid or no response from OpenAI")

        return response.output_text, response.usage.input_tokens, response.usage.output_tokens

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


class OpenAIStructuredClient(StructuredResponseClient):
    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY") or get_dotenv_var("OPENAI_API_KEY"))

    def get_completion(self, model_name: str, messages: list[dict[str, str]], data_model: BaseModel,
                       parameter_overrides: dict[str, Any] | None = None) -> BaseModel:
        return self.client.responses.parse(
            model=model_name,
            input=messages,
            text_format=data_model,
            **parameter_overrides
        ).output_parsed

    def get_batch_status(self, batch_id: str) -> tuple[BatchStatus, str|None, str|None]:
        # https://platform.openai.com/docs/guides/batch
        status_mapping = {
            "completed": BatchStatus.COMPLETED,
            "validating": BatchStatus.PROCESSING,
            "in_progress": BatchStatus.PROCESSING,
            "finalizing": BatchStatus.PROCESSING,
            "failed": BatchStatus.FAILED,
            "expired": BatchStatus.FAILED,
            "cancelling": BatchStatus.CANCELLED,
            "cancelled": BatchStatus.CANCELLED,
        }

        existing_batches = self.client.batches.list()
        existing_batches_data = existing_batches.data

        logger.info(f"Found {len(existing_batches_data)} batches. Checking status for batch {batch_id} ..")

        for batch in existing_batches_data:
            if not batch.metadata or not batch.metadata.get("d4-batch-id") or batch.metadata.get(
                    "d4-batch-id") != batch_id:
                continue

            self._print_batch_status(batch)

            return status_mapping[batch.status], batch.error_file_id, batch.output_file_id

        return BatchStatus.NOT_STARTED, None, None

    def _print_batch_status(self, batch: Batch) -> None:
        logger.info("---------------------------------------------------------")
        logger.info(f"Batch status:     {batch.status}")
        logger.info(f"Model:            {batch.model}")
        logger.info(f"D4-ID:            {batch.metadata.get('d4-batch-id')}")
        logger.info(f"ID:               {batch.id}")
        logger.info(f"Errors:           {batch.errors}")
        logger.info(f"Input tokens:     {batch.usage.input_tokens}")
        logger.info(f"Output tokens:    {batch.usage.output_tokens}")
        logger.info(f"Total tokens:     {batch.usage.total_tokens}")
        logger.info(f"Output file ID:   {batch.output_file_id}")
        logger.info(f"Error file ID:    {batch.error_file_id}")
        logger.info(f"Created at:       {self._format_time(batch.created_at)}")
        logger.info(f"In progress at:   {self._format_time(batch.in_progress_at)}")
        logger.info(f"Finalizing at:    {self._format_time(batch.finalizing_at)}")
        logger.info(f"Completed at:     {self._format_time(batch.completed_at)}")
        logger.info(f"Expires at:       {self._format_time(batch.expires_at)}")
        logger.info(f"Expired at:       {self._format_time(batch.expired_at)}")
        logger.info(f"Failed at:        {self._format_time(batch.failed_at)}")
        logger.info("---------------------------------------------------------")

    def download_file(self, file_id: str) -> str | None:
        if not file_id:
            return None
        response = self.client.files.content(file_id)
        return response.text

    def submit_completion_batch(self, jsonl_file_path: Path, batch_id: str) -> None:
        batch_input_file = self.client.files.create(
            file=open(jsonl_file_path, "rb"),
            purpose="batch"
        )

        batch_input_file_id = batch_input_file.id

        batch = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={
                "d4-batch-id": f"{batch_id}",
            }
        )

        self._print_batch_status(batch=batch)

    @staticmethod
    def make_schema_strict(schema: dict) -> dict:
        """
        Recursively ensure all properties are in required array for OpenAI strict mode.
        Also resolves $ref with default values by inlining the definition.
        """
        defs = schema.get("$defs", {})

        def resolve_refs(obj: dict, defs: dict) -> dict:
            """Recursively resolve $refs that have sibling keys (like default)."""
            if isinstance(obj, dict):
                # If this object has both $ref and other keys (like default), inline the ref
                if "$ref" in obj and len(obj) > 1:
                    ref_path = obj["$ref"]
                    if ref_path.startswith("#/$defs/"):
                        def_name = ref_path.split("/")[-1]
                        if def_name in defs:
                            # Merge the referenced definition with current keys
                            resolved = defs[def_name].copy()
                            for key, value in obj.items():
                                if key != "$ref":
                                    resolved[key] = value
                            return resolve_refs(resolved, defs)

                # Recursively process nested objects
                return {k: resolve_refs(v, defs) if isinstance(v, dict) else v
                        for k, v in obj.items()}
            return obj

        # Resolve all $refs with siblings
        schema = resolve_refs(schema, defs)

        def add_all_to_required(obj: dict) -> dict:
            if isinstance(obj, dict):
                if "properties" in obj:
                    obj["required"] = list(obj["properties"].keys())

                # Recursively process all nested objects
                for key, value in obj.items():
                    if isinstance(value, dict):
                        obj[key] = add_all_to_required(value)
                    elif isinstance(value, list):
                        obj[key] = [add_all_to_required(item) if isinstance(item, dict) else item
                                    for item in value]
            return obj

        schema = add_all_to_required(schema)

        return schema


if __name__ == "__main__":
    summary_client = OpenAISummaryClient()
    resp = summary_client.summarize(text="Title: foo\n\nAbstract: bar", model_name="gpt-4.1")
    print(resp)

    # structured_client = OpenAIStructuredClient()
    # status, error_url, data_url = structured_client.get_batch_status(batch_id="250304-alb-test-1")
    # print(status)


