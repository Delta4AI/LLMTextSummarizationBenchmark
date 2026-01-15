SUMMARY_MIN_WORDS: int = 15
SUMMARY_MAX_WORDS: int = 100


# the system prompt used for ollama, openai etc. model calls
SUMMARY_SYSTEM_PROMPT = f"""Summarize the provided publication (title and abstract) in {SUMMARY_MIN_WORDS}-{SUMMARY_MAX_WORDS} words.

Key requirements:
- Identify main findings, results, or contributions
- Preserve essential context and nuance
- Exclude background, methods unless crucial to conclusions
- Write concisely and objectively
- Avoid repetition and unnecessary qualifiers

If no substantial findings exist, respond: 'INSUFFICIENT_FINDINGS'
"""

TOKEN_SIZE_SAMPLE_TEXT = """Many words map to one token, but some don't: indivisible.
Unicode characters like emojis may be split into many tokens containing the underlying bytes: 🤚🏾
Sequences of characters commonly found next to each other may be grouped together: 1234567890"""

# specify platform-specific default parameters below

# https://huggingface.co/docs/transformers/v4.53.3/en/main_classes/pipelines#transformers.SummarizationPipeline
HUGGINGFACE_DEFAULT_PIPELINE_PARAMS = {
    "max_new_tokens": int(SUMMARY_MAX_WORDS * 1.40),  # calculated with text_summarization.llm_apis.get_token_sizes()
    "min_new_tokens": int(SUMMARY_MIN_WORDS * 1.40),
    "do_sample": False,
    "num_beams": 4,
    "early_stopping": True,
    "truncation": True,
    "length_penalty": 1.0,
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 3
    # "top_p": 1,
    # "temperature": 0.2,
}

HUGGINGFACE_DEFAULT_MODEL_PARAMS = {
    "max_new_tokens": int(SUMMARY_MAX_WORDS * 1.40),
    "min_new_tokens": int(SUMMARY_MIN_WORDS * 1.40),
    "do_sample": True,
    "temperature": 0.2
}

HUGGINGFACE_DEFAULT_TOKENIZER_PARAMS = {
    "return_tensors": "pt"
}

# https://platform.openai.com/docs/api-reference/responses/create
OPENAI_DEFAULT_PARAMS = {
    "temperature": 0.2,
    # "top_p": 1,
    # "max_completion_tokens": int(MAX_WORDS * 1.45),  # https://platform.openai.com/tokenizer
}

# https://platform.openai.com/docs/guides/latest-model#verbosity
OPENAI_GPT_5_DEFAULT_PARAMS = {
    "text": {
        "verbosity": "low"
    },
    "reasoning": {
        "effort": "minimal"
    }
}

# https://docs.anthropic.com/en/api/messages#body-temperature
ANTHROPIC_DEFAULT_PARAMS = {
    "temperature": 0.2,
    # "top_p": 1,
    # "top_k": 40,
    "max_tokens": 8192,
    "stream": False,
}

# https://docs.mistral.ai/api/#tag/chat/operation/chat_completion_v1_chat_completions_post
MISTRAL_DEFAULT_PARAMS = {
    "temperature": 0.2,
    # "top_p": 1,
    # "max_tokens": 8192,
    "stream": False,
}

# https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
OLLAMA_DEFAULT_PARAMS = {
    "temperature": 0.2,
    # "top_p": 1,
    # "top_k": 40,
    # "num_predict": 200,
}