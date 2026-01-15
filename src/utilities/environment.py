import functools
from pathlib import Path
from typing import Optional
import inspect
import os

from dotenv import load_dotenv


@functools.lru_cache(maxsize=12)
def _find_project_structure(start_path: str) -> dict[str, str|Path|None]:
    """Find the project root directory and .env file based on the caller's filename.

    Args:
        start_path: Starting path for the search

    Returns:
        dict[str, str|Path|None]: Dictionary containing 'project_root' and 'env_file' keys
    """

    def search_upwards(path: Path) -> tuple[Optional[Path], Optional[str]]:
        """Recursively search upwards for project files."""
        if path.parent == path:  # Reached filesystem root
            return None, None

        _project_root = path if (path / "pyproject.toml").exists() else None

        for env_file in [path / "Resources" / ".env", path / ".env"]:
            if env_file.exists():
                return _project_root, str(env_file)

        # Recurse to parent directory
        parent_root, parent_env = search_upwards(path.parent)
        return _project_root or parent_root, parent_env

    current_path = Path(start_path).resolve()

    if current_path.is_file():
        search_start = current_path.parent
    else:
        search_start = current_path

    project_root, env_file_path = search_upwards(search_start)

    if env_file_path:
        load_dotenv(env_file_path, override=False)

    return {
        'project_root': project_root or Path.cwd(),
        'env_file': env_file_path
    }


def get_project_root(caller_file: Optional[str] = None) -> Path:
    """Get the project root directory."""
    if caller_file is None:
        # Get the caller's filename
        frame = inspect.currentframe()
        try:
            caller_file = frame.f_back.f_code.co_filename
        finally:
            del frame

    return _find_project_structure(caller_file)['project_root']


def get_dotenv_var(var: str, caller_file: Optional[str] = None) -> Optional[str]:
    """Get environment variable (loads from .env file if needed)."""
    if caller_file is None:
        # Get the caller's filename
        frame = inspect.currentframe()
        try:
            caller_file = frame.f_back.f_code.co_filename
        finally:
            del frame

    _find_project_structure(caller_file)

    return os.getenv(var)