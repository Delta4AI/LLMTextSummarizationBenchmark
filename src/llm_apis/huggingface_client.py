import gc
import logging
import os
from pathlib import Path
from typing import Any

import torch.cuda
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

from utilities import get_dotenv_var
from llm_apis.config import (HUGGINGFACE_DEFAULT_PIPELINE_PARAMS, HUGGINGFACE_DEFAULT_MODEL_PARAMS,
                             HUGGINGFACE_DEFAULT_TOKENIZER_PARAMS)
from llm_apis.base_client import SummaryClient


logger = logging.getLogger(__name__)


class HuggingFacePipelineSummaryClient(SummaryClient):
    def __init__(self):
        init_hf_cache_dir()
        self.summarizer = None

    def warmup(self, model_name: str, train_corpus: list[str] | None = None):
        self.summarizer = pipeline(
            task="summarization",
            model=model_name,
        )

    def summarize(self, text: str, model_name: str, system_prompt_override: str | None = None,
                  parameter_overrides: dict[str, Any] | None = None,
                  tokenizer_overrides: dict[str, Any] | None = None) -> tuple[str, int, int]:
        try:
            logger.info(f"Making Huggingface summarizer pipeline request with model {model_name}")

            params = HUGGINGFACE_DEFAULT_PIPELINE_PARAMS.copy()
            if parameter_overrides:
                params.update(parameter_overrides)
            if system_prompt_override:
                params.update(system_prompt_override)

            response = self.summarizer(text, **params)

            return response[0]['summary_text'], 0, 0
        except Exception as e:
            raise RuntimeError(f"Huggingface summarizer pipeline request failed: {e}")

    def summarize_batch(self, texts: list[str], model_name: str, batch_size: int = 2,
                        system_prompt_override: str | None = None,
                        parameter_overrides: dict[str, Any] | None = None,
                        tokenizer_overrides: dict[str, Any] | None = None) -> tuple[list[str], list[int], list[int]]:
        try:
            logger.info(f"Making batched Huggingface summarizer pipeline request with model {model_name}")

            params = HUGGINGFACE_DEFAULT_PIPELINE_PARAMS.copy()
            if parameter_overrides:
                params.update(parameter_overrides)
            if system_prompt_override:
                params.update(system_prompt_override)

            params["batch_size"] = batch_size

            responses = self.summarizer(texts, **params)

            if len(responses) != len(texts):
                raise RuntimeError(f"Huggingface summarizer pipeline request failed: response length ({len(responses)})"
                                   f"does not match input length ({len(texts)})")

            return responses

        except Exception as e:
            raise RuntimeError(f"Huggingface summarizer pipeline request failed: {e}")

    def cleanup(self, model_name: str) -> None:
        if self.summarizer is not None:
            logger.info("Cleaning up HuggingFace model")
            del self.summarizer
            self.summarizer = None

        gc.collect()

        if torch.cuda.is_available():
            logger.info("Clearing CUDA GPU cache")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def test_token_size(self, model_name: str, text: str) -> int:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokens = tokenizer.encode(text)
        return len(tokens)


class HuggingFaceIndividualSummaryClient(SummaryClient):
    def __init__(self):
        init_hf_cache_dir()
        self.device = None
        self.model = None
        self.tokenizer = None

    def warmup(self, model_name: str, train_corpus: list[str] | None = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def summarize(self, text: str, model_name: str, system_prompt_override: str | None = None,
                  parameter_overrides: dict[str, Any] | None = None,
                  tokenizer_overrides: dict[str, Any] | None = None) -> tuple[str, int, int]:
        raise NotImplementedError

    def _run_individual_summarization_inference(self, prompt: str = None,
                                                parameter_overrides: dict[str, Any] | None = None,
                                                tokenizer_overrides: dict[str, Any] | None = None) -> str:
        tokenizer_params = HUGGINGFACE_DEFAULT_TOKENIZER_PARAMS.copy()
        if tokenizer_overrides:
            tokenizer_params.update(tokenizer_overrides)

        model_inputs = self.tokenizer([prompt], **tokenizer_params).to(self.device)

        model_params = HUGGINGFACE_DEFAULT_MODEL_PARAMS.copy()
        if parameter_overrides:
            model_params.update(parameter_overrides)

        if 'eos_token_id' not in model_params and self.tokenizer.eos_token_id is not None:
            model_params['eos_token_id'] = self.tokenizer.eos_token_id
        if 'pad_token_id' not in model_params and self.tokenizer.pad_token_id is not None:
            model_params['pad_token_id'] = self.tokenizer.pad_token_id

        generated_ids = self.model.generate(
            **model_inputs,
            **model_params,
        )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]

        return self.tokenizer.decode(output_ids, skip_special_tokens=True)

    def cleanup(self, model_name: str) -> None:
        if self.model is not None:
            logger.info("Cleaning up HuggingFace tokenizer and model")
            del self.model
            self.model = None
            del self.tokenizer
            self.tokenizer = None

        gc.collect()

        if torch.cuda.is_available():
            logger.info("Clearing CUDA GPU cache")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class HuggingFaceCompletionModelSummaryClient(HuggingFaceIndividualSummaryClient):
    def __init__(self):
        super().__init__()

    def summarize(self, text: str, model_name: str, system_prompt_override: str | None = None,
                  parameter_overrides: dict[str, Any] | None = None,
                  tokenizer_overrides: dict[str, Any] | None = None) -> tuple[str, int, int]:
        try:
            logger.info(f"Making Huggingface completion model request with {model_name}")

            # biogpt is a completion model, so we need to format the prompt appropriately
            prompt = f"{self.text_summarization_system_prompt}\n\nText to summarize:\n{text}\n\nSummary:"

            summary = self._run_individual_summarization_inference(prompt, parameter_overrides, tokenizer_overrides)

            return summary, 0, 0
        except Exception as e:
            raise RuntimeError(f"Huggingface completion model request failed: {e}")


class HuggingFaceChatModelSummaryClient(HuggingFaceIndividualSummaryClient):
    def __init__(self):
        super().__init__()

    def summarize(self, text: str, model_name: str, system_prompt_override: str | None = None,
                  parameter_overrides: dict[str, Any] | None = None,
                  tokenizer_overrides: dict[str, Any] | None = None) -> tuple[str, int, int]:
        try:
            logger.info(f"Making Huggingface chat summarizer request with model {model_name}")
            messages = [
                {"role": "system", "content": self.text_summarization_system_prompt},
                {"role": "user", "content": text}
            ]

            try:
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except ValueError:
                prompt = f"{self.text_summarization_system_prompt}\n\n{text}"

            summary = self._run_individual_summarization_inference(prompt, parameter_overrides, tokenizer_overrides)

            return summary, 0, 0
        except Exception as e:
            raise RuntimeError(f"Huggingface chat summarizer request failed: {e}")


class HuggingFaceConversationalModelSummaryClient(HuggingFaceIndividualSummaryClient):
    def __init__(self):
        super().__init__()

    def summarize(self, text: str, model_name: str, system_prompt_override: str | None = None,
                  parameter_overrides: dict[str, Any] | None = None,
                  tokenizer_overrides: dict[str, Any] | None = None) -> tuple[str, int, int]:
        try:
            logger.info(f"Making Huggingface conversational summarizer request with model {model_name}")
            messages = [
                {"role": "user", "content": self.text_summarization_system_prompt},
                {"role": "assistant", "content": "I understand. I will summarize the text you provide."},
                {"role": "user", "content": text}
            ]

            try:
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except ValueError:
                prompt = f"{self.text_summarization_system_prompt}\n\n{text}"

            summary = self._run_individual_summarization_inference(prompt, parameter_overrides, tokenizer_overrides)

            return summary, 0, 0
        except Exception as e:
            raise RuntimeError(f"Huggingface conversational summarizer request failed: {e}")


def init_hf_cache_dir():
    hf_home = get_dotenv_var("HF_HOME")
    if hf_home:
        hf_path = Path(hf_home)
        if hf_path.is_dir():
            os.environ["HF_HOME"] = str(hf_path)
            logger.info(f"Set HF_HOME to: {hf_path}")
        else:
            logger.warning(f"HF_HOME directory does not exist: {hf_path}")
            try:
                hf_path.mkdir(parents=True, exist_ok=True)
                os.environ["HF_HOME"] = str(hf_path)
                logger.info(f"Created and set HF_HOME to: {hf_path}")
            except OSError as e:
                logger.error(f"Failed to create HF_HOME directory: {e}")
                logger.warning("Using default HF_HOME: ~/.cache/huggingface")
    else:
        logger.info("HF_HOME not configured, using default: ~/.cache/huggingface")


if __name__ == "__main__":
    hf_client = HuggingFacePipelineSummaryClient()
    hf_client.warmup(model_name="facebook/bart-large-cnn")
    resp = hf_client.summarize(text="Title: foo\n\nAbstract: bar", model_name="facebook/bart-base")
    print(resp)