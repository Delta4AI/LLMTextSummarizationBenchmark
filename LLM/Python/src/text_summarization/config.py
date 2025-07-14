from collections import namedtuple
from typing import NamedTuple, Any

MIN_WORDS: int = 15
MAX_WORDS: int = 50
OUTPUT_DIR: str = "benchmark_results"

GOLD_STANDARD_DATA: list[str] = [
    "Resources/text_summarization_goldstandard_data_AKI_CKD.json",
    "Resources/text_summarization_goldstandard_data_test.json"
]

# the system prompt used for ollama, openai etc. model calls
SYSTEM_PROMPT = f"""Summarize the provided publication (consisting of a title and abstract) in {MIN_WORDS}-{MAX_WORDS} words, preserving all key findings and central conclusions.
                            
- Carefully read the title and abstract to identify the most significant findings, results, or contributions.
- Retain relevant context or nuance so the key findings and main message are clear.
- The summary must be within {MIN_WORDS}-{MAX_WORDS} words. Do not exceed or go under this limit.
- Exclude unnecessary background, introductory explanations, or restatements of methods unless directly related to the main conclusion.
- Write concisely and objectively, focusing on substance rather than style."""

TOKEN_SIZE_SAMPLE_TEXT = """Many words map to one token, but some don't: indivisible.
Unicode characters like emojis may be split into many tokens containing the underlying bytes: 🤚🏾
Sequences of characters commonly found next to each other may be grouped together: 1234567890"""

HUGGINGFACE_DEFAULT_PARAMS = {
    "max_new_tokens": int(MAX_WORDS * 1.40),  # calculated with text_summarization.llm_apis.get_token_sizes()
    "min_new_tokens": int(MIN_WORDS * 1.40),
    "do_sample": False,
    "num_beams": 4,
    "early_stopping": True,
    "length_penalty": 2.0,
    "top_p": 1,
    "temperature": 1,
}

OPENAI_DEFAULT_PARAMS = {
    "temperature": 1,
    "top_p": 1,
    "max_tokens": int(MAX_WORDS * 1.45),  # https://platform.openai.com/tokenizer
}

ANTHROPIC_DEFAULT_PARAMS = {
    "temperature": 1,
    "top_p": 1,
    "top_k": 40,
    "max_tokens": 1024,
}

MISTRAL_DEFAULT_PARAMS = {
    "temperature": 1,
    "top_p": 1,
    "max_tokens": 1024,
    "stream": False,
}

OLLAMA_DEFAULT_PARAMS = {
    "temperature": 1,
    "top_p": 1,
    "top_k": 40,
    # "num_predict": 200,  # TODO: might influence the returned length .. update dynamically from min_words and max_words, calculate in warum?
}