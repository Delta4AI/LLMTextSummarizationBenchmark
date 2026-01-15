import re

import numpy as np

INSUFFICIENT_FINDINGS = "INSUFFICIENT_FINDINGS"


def extract_response(response_text: str) -> str:
    text = None

    try:
        text = response_text.strip()
    except AttributeError:
        if isinstance(response_text, list) and isinstance(
                response_text[0], dict) and "thinking" in response_text[0].keys():
            text = response_text[1]["text"].strip()

    if not text or re.search(r'(?:(?<=^)|(?<=\)))\s*INSUFFICIENT(?:[_\s-]*FINDINGS)?\b[:.]?', text):
        return INSUFFICIENT_FINDINGS

    # Remove internal tags (think blocks, XML-like tags, <n> placeholders)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'</?think>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^<[^>]*>\s*', '', text)
    text = re.sub(r'</?n[^>]*>', ' ', text, flags=re.IGNORECASE)

    text = _strip_markdown(text)

    # Remove generic AI / abstract summary prefixes in parentheses
    text = re.sub(
        r'^\s*\((?:Summary|Abstract)\s*(?:written\s*by\s*AI\s*model|by\s*AAI\s*Researcher|adapted\s*from(?:\s*the)?\s*(?:provided\s*)?abstract|modified\s*from\s*original)?\)\s*',
        '',
        text,
        flags=re.IGNORECASE
    )

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
        r'(?:Summary|Answer|Result):\s*\n?(.+)',
        r'(?:TL;DR|TLDR):\s*\n?(.+)',
    ]

    for pattern in summary_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
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
        return INSUFFICIENT_FINDINGS
    
    # Handle BioGPT garbage or prompt echoes
    if re.search(r'<\s*(?:AbstractText|ns0:|math|mrow|mi|mn|msub)', text, flags=re.IGNORECASE):
        return INSUFFICIENT_FINDINGS
    if re.match(r'^\s*Summarize\s+the\s+provided\s+publication', text, flags=re.IGNORECASE):
        return INSUFFICIENT_FINDINGS
    if len(re.findall(r'<[^>]+>', text)) > len(text.split()) * 0.2:
        return INSUFFICIENT_FINDINGS

    # Handle URL-only outputs
    if re.fullmatch(r'\(?\s*(?:source|abstract\s*from)[:\s-]*https?://\S+\)?', text, flags=re.IGNORECASE):
        return INSUFFICIENT_FINDINGS
    if re.fullmatch(r'https?://\S+', text.strip(), flags=re.IGNORECASE):
        return INSUFFICIENT_FINDINGS
    
    if re.match(r'^\s*\(Summary\s+by[^)]*\)\s*$', text.strip(), flags=re.IGNORECASE):
        return INSUFFICIENT_FINDINGS

    return _clean_text(text)


def _strip_markdown(text: str) -> str:
    """Strip markdown formatting to normalize text for pattern matching."""
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)  # Headings
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # Bold
    text = re.sub(r'__(.+?)__', r'\1', text)  # Bold
    text = re.sub(r'\*(.+?)\*', r'\1', text)  # Italic
    text = re.sub(r'_(.+?)_', r'\1', text)  # Italic
    text = re.sub(r'`([^`]+)`', r'\1', text)  # Inline code
    return text


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

    # Strip (n words)
    text = re.sub(r'\s*\(\d+\s*words?\)\s*$', '', text, flags=re.IGNORECASE)

    return text.strip()


def normalize_llm_response(response_body):
    choices = response_body.get("choices", [])

    if not choices:
        raise ValueError("No choices in response")

    first_choice = choices[0]
    message = first_choice.get("message", {})
    content = message.get("content", "")

    # mistral with separate thinking/answer (2 choices)
    if len(choices) == 2:
        thinking = choices[0]["message"]["content"][0]["thinking"][0]["text"]
        answer = choices[0]["message"]["content"][1]["text"]
        return {
            "raw_response": thinking,
            "answer": answer
        }

    # structured content with text field
    if isinstance(content, list) and len(content) >= 2:
        last_item = content[-1]
        if isinstance(last_item, dict) and "text" in last_item:
            return {
                "raw_response": content,
                "answer": last_item["text"]
            }

    # standard single string content (OpenAI, Anthropic, etc.)
    if isinstance(content, str):
        return {
            "raw_response": content,
            "answer": content
        }

    # fallback
    if content:
        return {
            "raw_response": str(content),
            "answer": str(content)
        }

    raise NotImplementedError(
        f"Unsupported response format: {len(choices)} choices, "
        f"content type: {type(content)}"
    )


def find_truncated(raw_responses: list[str], threshold: float = 0.4):
    """Find summaries that lost significant content."""
    truncated = []

    for i, raw in enumerate(raw_responses):
        result = extract_response(raw)

        if result == INSUFFICIENT_FINDINGS:
            continue

        # Compare meaningful content length (stripped of whitespace/formatting)
        raw_words = len(raw.split())
        result_words = len(result.split())

        if raw_words > 0:
            ratio = result_words / raw_words

            if ratio < threshold:
                truncated.append({
                    'index': i,
                    'raw_words': raw_words,
                    'result_words': result_words,
                    'ratio': ratio,
                    'raw': raw,
                    'result': result,
                })

    # Sort by worst truncation
    truncated.sort(key=lambda x: x['ratio'])

    print(
        f"Found {len(truncated)} / {len(raw_responses)} potentially truncated ({len(truncated) / len(raw_responses) * 100:.1f}%)\n")

    for item in truncated[:10]:
        print(f"#{item['index']}: {item['raw_words']} words -> {item['result_words']} words ({item['ratio']:.0%})")
        print(f"  IN:  {item['raw'][:150]}...")
        print(f"  OUT: {item['result'][:150]}...")
        print()

    return truncated


def get_min_max_mean_std(values: list[float]) -> dict[str, float]:
    """Get min, max, mean, and std from a list of values."""
    return {
        "min": float(np.min(values)) if values else 0.0,
        "max": float(np.max(values)) if values else 0.0,
        "mean": float(np.mean(values)) if values else 0.0,
        "std": float(np.std(values)) if values else 0.0
    }



if __name__ == "__main__":
    foo = extract_response("""**Summary:**
SynPull—a novel method integrating single-molecule pull-down, super-resolution microscopy, and computational analysis—enables characterization of neurodegeneration-linked protein aggregates in synaptosomes. Key findings include:
- **Tau (AT8-positive)** aggregates dominate Alzheimer’s disease (AD) synaptosomes, with no size differences between healthy and diseased samples.
- **Alpha-synuclein and beta-amyloid** aggregates, though less abundant in synapses, are **larger** than extra-synaptic counterparts.

The method advances understanding of synaptic dysfunction mechanisms in neurodegeneration.""")
    print(foo)

