import logging
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import ollama

from utilities import get_dotenv_var
from llm_apis.base_client import SummaryClient, StructuredResponseClient, BaseModel, BatchStatus
from llm_apis.config import OLLAMA_DEFAULT_PARAMS


logger = logging.getLogger(__name__)


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


class OllamaSummaryClient(SummaryClient):
    def __init__(self, host: str = None):
        self.client = ollama.Client(host=host or get_dotenv_var("OLLAMA_BASE_URL"))

    def warmup(self, model_name: str, train_corpus: list[str] | None = None):
        try:
            logger.info(f"Warming up Ollama {model_name} model")
            response = self.client.generate(
                model=model_name,
                prompt="What is 2+2?",
                options={"temperature": 0.1, "num_predict": 10},
                context=None,
            )
            logger.info(f"Warmup response: {response['response']}")
        except Exception as e:
            logger.error(f"Warmup failed: {e}")

    def summarize(self, text: str, model_name: str, system_prompt_override: str | None = None,
                  parameter_overrides: dict[str, Any] | None = None,
                  tokenizer_overrides: dict[str, Any] | None = None) -> tuple[str, int, int]:
        try:

            logger.info(f"Making Ollama request with model {model_name}")
            response = self.client.generate(
                model=model_name,
                prompt=f"{system_prompt_override or self.text_summarization_system_prompt}\n\n{text}",
                options={**OLLAMA_DEFAULT_PARAMS, **(parameter_overrides or {})},
                context=None,
            )

            if not response or 'response' not in response:
                raise ValueError("Invalid or no response from Ollama")

            logger.info(f"Successfully generated summary with Ollama {model_name}")

            return response["response"], response.prompt_eval_count, response.eval_count

        except Exception as e:
            raise ValueError(f"Unexpected error in Ollama summarization: {e}")

    def cleanup(self, model_name: str) -> None:
        try:
            logger.info(f"Unloading model: {model_name} to free up resources")
            self.client.generate(
                model=model_name,
                prompt="",
                options={"num_predict": 0},
                keep_alive=0
            )
            time.sleep(2)
        except Exception as e:
            logger.error(f"Error unloading model: {e}")

    def test_token_size(self, model_name: str, text: str) -> int:
        try:
            _resp = self.client.embed(model=model_name, input=text)
        except ollama._types.ResponseError:
            # fall back for models not supporting the embed endpoint
            _resp = self.client.generate(model=model_name, prompt=text, options={"num_predict": 0})
        return _resp["prompt_eval_count"]

    def embed_texts(self, model: str, texts:list[str]) -> Sequence[Sequence[float]]:
        return self.client.embed(model=model, input=texts).embeddings

    def get_similarities(self, model_name: str, texts: list[str], ) -> list[Any]:
        texts = list(set(texts))
        embeddings = self.embed_texts(texts=texts, model=model_name)
        text_to_embeddings = dict(zip(texts, embeddings))

        similarities = []
        for text1, text2 in combinations(texts, 2):
            similarity = cosine_similarity(text_to_embeddings[text1], text_to_embeddings[text2])
            similarities.append((text1, text2, similarity))

        return sorted(similarities, key=lambda x: x[2], reverse=True)

    def get_available_models(self) -> list[tuple[str, str, str]]:
        models = self.client.list()
        return [(_["model"], _["details"]["family"], _["details"]["parameter_size"]) for _ in models["models"]]


class OllamaStructuredClient(StructuredResponseClient):
    def __init__(self, host: str = None):
        self.client = ollama.Client(host=host or get_dotenv_var("OLLAMA_BASE_URL"))

    def get_completion(self, model_name: str, messages: list[dict[str, str]], data_model: BaseModel,
                       parameter_overrides: dict[str, Any] | None) -> BaseModel:

        response = self.client.chat(
            model=model_name,
            messages=messages,
            format=data_model.model_json_schema(),
            options=parameter_overrides
        )
        return data_model.model_validate_json(response.message.content)

    def get_batch_status(self, batch_id: str) -> tuple[BatchStatus, str | None, str | None]:
        raise NotImplementedError("OllamaStructuredClient does not support batches. Use sequential get_completion api.")

    def download_file(self, file_id: str) -> str | None:
        raise NotImplementedError("OllamaStructuredClient does not support batches. Use sequential get_completion api.")

    def submit_completion_batch(self, jsonl_file_path: Path, batch_id: str) -> None:
        raise NotImplementedError("OllamaStructuredClient does not support batches. Use sequential get_completion api.")


if __name__ == "__main__":
    client = OllamaSummaryClient()
    client.get_available_models()
    resp = client.summarize(text="Title: foo\n\nAbstract: bar", model_name="gpt-oss:20b")
    print(resp)