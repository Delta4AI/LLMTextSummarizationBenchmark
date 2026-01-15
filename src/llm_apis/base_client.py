from datetime import datetime
from enum import Enum
import hashlib
import json
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import httpx
from utilities import get_project_root
from pydantic import BaseModel

from llm_apis.config import SUMMARY_SYSTEM_PROMPT, SUMMARY_MIN_WORDS, SUMMARY_MAX_WORDS


OUT_DIR = get_project_root() / "Output" / "llm_apis"
CACHE_FN = OUT_DIR / "cache.json"
BATCH_CACHE_FN = OUT_DIR / "batch_cache.json"


class BatchStatus(str, Enum):
    NOT_STARTED = "not started"
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchCache:
    batch_uuid: str | None = None
    status: BatchStatus = BatchStatus.NOT_STARTED
    duration: int = 0
    results: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> 'BatchCache':
        status_value = data.get("status", BatchStatus.NOT_STARTED.value)
        return cls(
            batch_uuid=data.get("batch_uuid", None),
            status=BatchStatus(status_value),
            duration=data.get("duration", 0),
            results=data.get("results", []),
            errors=data.get("errors", []),
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        return data


class BaseClient(ABC):
    _cache_loaded: bool = False
    _cache: dict[str, dict[str, dict[str, str | float]]]
    _batch_cache: dict[str, dict[str, dict[str, None | list[str] | str]]]

    def warmup(self, model_name: str, train_corpus: list[str] | None) -> None | NotImplementedError:
        """Optional warmup method. Override if needed"""
        raise NotImplementedError

    def cleanup(self, model_name: str) -> None | NotImplementedError:
        """Optional cleanup method. Override if needed"""
        raise NotImplementedError

    def _ensure_cache_loaded(self) -> None:
        if self._cache_loaded:
            return

        self._cache = self._load_json_file(file_path=CACHE_FN)
        self._batch_cache = self._load_json_file(file_path=BATCH_CACHE_FN)

        self._cache_loaded = True

    @staticmethod
    def _load_json_file(file_path: Path) -> dict:
        try:
            with file_path.open(mode="r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {}

        return data

    @staticmethod
    def _safe_replace_cache(data: dict, cache_path: Path) -> None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile("w", delete=False, dir=str(OUT_DIR), encoding="utf-8") as tmp:
            json.dump(data, tmp, ensure_ascii=False, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp.name, cache_path)

    def load_cache(self, method_name: str, system_prompt: str, user_query: str) -> str | None:
        """Return cached response or None when the entry is missing."""
        self._ensure_cache_loaded()
        _hash = self._get_hash_from_query(system_prompt, user_query)
        return self._cache.get(method_name, {}).get(_hash)

    def load_batch_cache(self, method_name: str, papers_hash: str) -> BatchCache:
        self._ensure_cache_loaded()
        _cache = self._batch_cache.get(method_name, {}).get(papers_hash, {})
        return BatchCache.from_dict(_cache) if _cache else BatchCache()

    def save_cache(self, method_name: str, system_prompt: str, user_query: str, response: str,
                   execution_time: float, input_tokens: int, output_tokens: int) -> None:
        """Persist a response to disk (atomic)."""
        self._ensure_cache_loaded()
        _hash = self._get_hash_from_query(system_prompt, user_query)
        self._cache.setdefault(method_name, {})[_hash] = {
            "response": response,
            "execution_time": execution_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }

        self._safe_replace_cache(data=self._cache, cache_path=CACHE_FN)

    def save_batch_cache(self, method_name: str, papers_hash: str, batch_cache: BatchCache) -> None:
        self._ensure_cache_loaded()
        self._batch_cache.setdefault(method_name, {})[papers_hash] = batch_cache.to_dict()
        self._safe_replace_cache(data=self._batch_cache, cache_path=BATCH_CACHE_FN)

    def clean_cache(self, method_name: str) -> None:
        """Remove all cached responses for a given method."""
        self._ensure_cache_loaded()
        self._cache.pop(method_name, None)
        self._safe_replace_cache(data=self._cache, cache_path=CACHE_FN)

    @staticmethod
    def _get_hash_from_query(system_prompt: str, user_query: str) -> str:
        _SEP: str = "\u241E"
        payload: bytes = f"{system_prompt}{_SEP}{user_query}".encode("utf-8", "replace")
        digest: str = hashlib.sha256(payload).hexdigest()

        return digest

    @staticmethod
    def _load_batch_jsonl(job_id: str, is_error_file: bool = False) -> list[dict] | None:
        file_path = OUT_DIR / "batch_results" / f"{job_id}_{'error' if is_error_file else 'output'}.jsonl"
        try:
            with open(file_path, mode="r", encoding="utf-8") as f:
                results = []
                for line in f:
                    line = line.strip()
                    if line:
                        results.append(json.loads(line))
                return results
        except FileNotFoundError:
            return None

    @staticmethod
    def _save_batch_jsonl(job_id: str, response: httpx.Response, is_error_file: bool = False) -> Path:
        file_path = OUT_DIR / "batch_results" / f"{job_id}_{'errors' if is_error_file else 'output'}.jsonl"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, mode="w", encoding="utf-8") as f:
            for chunk in response.stream:
                f.write(chunk.decode("utf-8"))
        return file_path


class SummaryClient(BaseClient):
    text_summarization_system_prompt = SUMMARY_SYSTEM_PROMPT
    min_words = SUMMARY_MIN_WORDS
    max_words = SUMMARY_MAX_WORDS

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def summarize(self, text: str, model_name: str, system_prompt_override: str | None = None,
                  parameter_overrides: dict[str, Any] | None = None,
                  tokenizer_overrides: dict[str, Any] | None = None) -> tuple[str, int, int] | RuntimeError:
        pass

    def summarize_batch(
            self, texts: list[str], model_name: str, batch_size: int = 2, system_prompt_override: str | None = None,
            parameter_overrides: dict[str, Any] | None = None,
            tokenizer_overrides: dict[str, Any] | None = None) -> list[Any] | RuntimeError | NotImplementedError:
        """Optional summarization method for batch processing. Override if needed"""
        raise NotImplementedError

    def test_token_size(self, model_name: str, text: str) -> int | NotImplementedError:
        """Optional test token size method. Override if needed"""
        raise NotImplementedError


class StructuredResponseClient(BaseClient):
    @staticmethod
    def format_messages(system_prompt: str = "You are a helpful assistant.",
                        user_query: str = "What is the meaning of life?") -> list[dict[str, str]]:
        """Return a list of messages to be passed to the LLM."""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

    @staticmethod
    def format_completion_request(
            custom_id: str,
            model_name: str = "gpt-5-2025-08-07",
            messages: list[dict[str, str]] = None,
            data_model_schema: dict = None,
            config: dict[str, Any] = None,
            endpoint: str = "/v1/responses",
    ) -> dict[str, Any]:
        _resp = {
            "custom_id": custom_id,
            "method": "POST",
            "url": endpoint,
            "body": {
                "model": model_name,
            }
        }

        if endpoint == "/v1/chat/completions":
            _completion_body = {
                "messages": messages,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "LLMResponse",
                        "strict": True,
                        "schema": data_model_schema
                    }
                }
            }
            _resp["body"].update(_completion_body)

        elif endpoint == "/v1/responses":
            _responses_body = {
                "input": messages,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "LLMResponse",
                        "strict": True,
                        "schema": data_model_schema
                    }
                }
            }
            _resp["body"].update(_responses_body)

        if config:
            _resp["body"].update(config)

        return _resp

    @staticmethod
    def _format_time(timestamp: int) -> str:
        if not timestamp:
            return "None"
        return f"{datetime.fromtimestamp(timestamp)}"

    @abstractmethod
    def get_completion(self, model_name: str, messages: list[dict[str, str]], data_model: BaseModel,
                       parameter_overrides: dict[str, Any] | None) -> BaseModel:
        raise NotImplementedError

    @abstractmethod
    def get_batch_status(self, batch_id: str) -> tuple[BatchStatus, str | None, str | None]:
        raise NotImplementedError

    @abstractmethod
    def download_file(self, file_id: str) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def submit_completion_batch(self, jsonl_file_path: Path, batch_id: str) -> None:
        raise NotImplementedError
