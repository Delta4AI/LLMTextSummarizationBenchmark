import re

import numpy as np


def extract_response(response_text: str) -> str:
    # Empty input check
    if not response_text or not response_text.strip():
        return ""

    # Trim whitespaces
    text = response_text.strip()

    # Remove internal tags (think blocks, XML-like tags, <n> placeholders)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'</?think>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^<[^>]*>\s*', '', text)
    text = re.sub(r'</?n[^>]*>', ' ', text, flags=re.IGNORECASE)

    # Remove generic AI / abstract summary prefixes in parentheses
    text = re.sub(
        r'^\s*\((?:Summary|Abstract)\s*(?:written\s*by\s*AI\s*model|by\s*AAI\s*Researcher|adapted\s*from(?:\s*the)?\s*(?:provided\s*)?abstract|modified\s*from\s*original)?\)\s*',
        '',
        text,
        flags=re.IGNORECASE
    )

    # Detect insufficient summaries (normalize to "INSUFFICIENT_FINDINGS")
    if re.search(
        r'(?:(?<=^)|(?<=\)))\s*INSUFFICIENT(?:[_\s-]*FINDINGS)?\b[:.]?',
        text
    ):
        return "INSUFFICIENT_FINDINGS"

    # Detect dictionary-like responses containing "summary"
    dict_like_match = re.search(
        r"""['"{\s]*summary['"\s:]+['"](.+?)['"]\s*[},]*$""",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if dict_like_match:
        return _clean_text(dict_like_match.group(1))

    # Look for explicit markers
    summary_patterns = [
        r'(?:Summary|Answer|Result):\s*(.+?)(?:\n|$)',
        r'(?:TL;DR|TLDR):\s*(.+?)(?:\n|$)',
    ]

    for pattern in summary_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return _clean_text(match.group(1))

    lines = [line.strip() for line in text.split('\n') if line.strip()]
    good_lines = []

    for line in lines:
        if re.match(r'^(?:let me|i will|i need to|okay|now|first|the user|based on)', line, re.IGNORECASE):
            continue
        if len(line.split()) < 5:
            continue
        good_lines.append(line)

    # Pick best line
    if good_lines:
        complete_sentences = [line for line in good_lines if line.endswith(('.', '!', '?'))]
        if complete_sentences:
            return _clean_text(max(complete_sentences, key=len))
        else:
            return _clean_text(good_lines[0])

    # Fallback
    for line in lines:
        if len(line) > 20:
            return _clean_text(line)

    # Handle symbol-only outputs (like "**", "..." or "___")
    if re.fullmatch(r'[\s\*\_\-\.\,\'\"`~!@#$%^&*()+=/\\|<>?\[\]{}:;]+', text):
        return "INSUFFICIENT_FINDINGS"
    
    # Handle BioGPT garbage or prompt echoes
    if re.search(r'<\s*(?:AbstractText|ns0:|math|mrow|mi|mn|msub)', text, flags=re.IGNORECASE):
        return "INSUFFICIENT_FINDINGS"
    if re.match(r'^\s*Summarize\s+the\s+provided\s+publication', text, flags=re.IGNORECASE):
        return "INSUFFICIENT_FINDINGS"
    if len(re.findall(r'<[^>]+>', text)) > len(text.split()) * 0.2:
        return "INSUFFICIENT_FINDINGS"

    # Handle URL-only outputs
    if re.fullmatch(r'\(?\s*(?:source|abstract\s*from)[:\s-]*https?://\S+\)?', text, flags=re.IGNORECASE):
        return "INSUFFICIENT_FINDINGS"
    if re.fullmatch(r'https?://\S+', text.strip(), flags=re.IGNORECASE):
        return "INSUFFICIENT_FINDINGS"
    
    if re.match(r'^\s*\(Summary\s+by[^)]*\)\s*$', text.strip(), flags=re.IGNORECASE):
        return "INSUFFICIENT_FINDINGS"

    return _clean_text(text)


def _clean_text(text: str) -> str:
    # Remove Markdown bold/italic markers
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)

    # Unescape common escaped quotes and slashes (like: \"word\" -> "word")
    text = text.replace(r'\"', '"').replace(r"\'", "'").replace(r"\\", "\\")

    # Remove prefixes like "Summary:" or "Answer:"
    text = re.sub(r'^(?:Summary|Answer|Result):\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^["\'](.+)["\']$', r'\1', text)
    text = ' '.join(text.split())

    # Remove spaces before punctuation (like: "word ," -> "word,")
    text = re.sub(r'\s+([,.;:!?])', r'\1', text)

    # Remove spaces after opening brackets, before closing ones and around slashes
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    text = re.sub(r'\[\s+', '[', text)
    text = re.sub(r'\s+\]', ']', text)
    text = re.sub(r'\s+/', '/', text)
    text = re.sub(r'/\s+', '/', text)

    return text.strip()


def get_min_max_mean_std(values: list[float]) -> dict[str, float]:
    """Get min, max, mean, and std from a list of values."""
    return {
        "min": float(np.min(values)) if values else 0.0,
        "max": float(np.max(values)) if values else 0.0,
        "mean": float(np.mean(values)) if values else 0.0,
        "std": float(np.std(values)) if values else 0.0
    }
