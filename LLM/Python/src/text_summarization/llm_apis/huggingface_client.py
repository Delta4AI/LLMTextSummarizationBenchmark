import logging
import warnings
from typing import Any

from text_summarization.llm_apis.base_client import BaseClient

logger = logging.getLogger(__name__)

try:
    from transformers import BartTokenizer, BartForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.error("Transformers not installed. HuggingFace models will be unavailable.")
    TRANSFORMERS_AVAILABLE = False

# Suppress transformers warnings
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')

LOOKUP = {
    "huggingface_bart-large-cnn": (BartTokenizer, BartForConditionalGeneration),
    "huggingface_t5-base": (T5Tokenizer, T5ForConditionalGeneration),
}


class HuggingFaceClient(BaseClient):
    def __init__(self):
        self.models = {}
        if TRANSFORMERS_AVAILABLE:
            self._load_models()

        self.tokenizer = None
        self.model = None

    def _load_models(self):
        """Load pre-trained models on initialization."""
        try:
            self.models['bart_tokenizer'] = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
            self.models['bart_model'] = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
            logger.info("BART model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load BART model: {e}")

        try:
            self.models['t5_tokenizer'] = T5Tokenizer.from_pretrained('t5-base')
            self.models['t5_model'] = T5ForConditionalGeneration.from_pretrained('t5-base')
            logger.info("T5 model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load T5 model: {e}")

    def summarize(self, text: str, model_name: str, system_prompt_override: str | None = None,
                  parameter_overrides: dict[str, Any] | None = None) -> str:
        """
        HuggingFace model summarization.

        Args:
            text: Input text to summarize
            model_name: Model name ("bart-large-cnn" or "t5-base")
            system_prompt_override: Not used for HuggingFace models (they don't need prompts)
            parameter_overrides: Additional parameters for Ollama API (e.g., {"temperature": 0.3})

        Returns:
            Generated summary

        Raises:
            Exception: If the model is not available or inference fails
        """
        # TODO: harmonize class, use warmup for initializing tokenizer and model
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available")

        if model_name == "bart-large-cnn":
            return self._bart_summarize(text)
        elif model_name == "t5-base":
            return self._t5_summarize(text)
        else:
            raise FileNotFoundError(f"Unsupported HuggingFace model: {model_name}")

    def _bart_summarize(self, text: str) -> str:
        """BART-based abstractive summarization."""
        if 'bart_tokenizer' not in self.models or 'bart_model' not in self.models:
            raise FileNotFoundError("BART model not available")

        tokenizer = self.models['bart_tokenizer']
        model = self.models['bart_model']

        inputs = tokenizer.encode(text, return_tensors='pt', max_length=1024, truncation=True)

        with torch.no_grad():
            summary_ids = model.generate(
                inputs,
                max_length=50,  # Adjust based on your target length
                min_length=15,  # Adjust based on your target length
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        logger.info("Successfully generated summary with BART")
        return summary

    def _t5_summarize(self, text: str) -> str:
        """T5-based abstractive summarization."""
        if 't5_tokenizer' not in self.models or 't5_model' not in self.models:
            raise FileNotFoundError("T5 model not available")

        tokenizer = self.models['t5_tokenizer']
        model = self.models['t5_model']

        # T5 requires task prefix
        input_text = f"summarize: {text}"
        inputs = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)

        with torch.no_grad():
            summary_ids = model.generate(
                inputs,
                max_length=50,  # Adjust based on your target length
                min_length=15,  # Adjust based on your target length
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        logger.info("Successfully generated summary with T5")
        return summary