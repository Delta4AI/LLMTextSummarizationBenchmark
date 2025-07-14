import re
from pathlib import Path
import os
import functools

from dotenv import load_dotenv
import numpy as np


def extract_response(response_text: str) -> str:
    if not response_text or not response_text.strip():
        return ""

    text = response_text.strip()

    # Remove thinking blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'</?think>', '', text, flags=re.IGNORECASE)

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

    return _clean_text(text)


def _clean_text(text: str) -> str:
    text = re.sub(r'^(?:Summary|Answer|Result):\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^["\'](.+)["\']$', r'\1', text)
    text = ' '.join(text.split())
    return text.strip()


def get_min_max_mean_std(values: list[float]) -> dict[str, float]:
    """Get min, max, mean, and std from a list of values."""
    return {
        "min": float(np.min(values)) if values else 0.0,
        "max": float(np.max(values)) if values else 0.0,
        "mean": float(np.mean(values)) if values else 0.0,
        "std": float(np.std(values)) if values else 0.0
    }


@functools.lru_cache(maxsize=1)
def _find_project_structure():
    """Find project root and load .env file once and cache the result."""
    current_path = Path(__file__).resolve()
    project_root = None
    env_file_path = None

    # Search upwards for project root (pyproject.toml) and .env files
    for parent in current_path.parents:
        if project_root is None and (parent / "pyproject.toml").exists():
            project_root = parent

        # Check for Resources/.env
        if env_file_path is None:
            env_file = parent / "Resources" / ".env"
            if env_file.exists():
                load_dotenv(env_file)
                env_file_path = str(env_file)

    # Fallback: try to find any .env file in parent directories
    if env_file_path is None:
        for parent in current_path.parents:
            env_file = parent / ".env"
            if env_file.exists():
                load_dotenv(env_file)
                env_file_path = str(env_file)
                break

    return {
        'project_root': project_root or Path.cwd(),
        'env_file': env_file_path
    }


def get_project_root() -> Path:
    """Get the project root directory."""
    return _find_project_structure()['project_root']


def get_dotenv_param(param: str) -> str | None:
    """Get parameter from .env file (loads once and caches)."""
    _find_project_structure()
    return os.getenv(param)
