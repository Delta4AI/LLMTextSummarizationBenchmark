from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Paper:
    """Data class for scientific papers with gold-standard summaries."""
    title: str
    abstract: str
    id: str
    summaries: list[str]  # Gold-standard reference summaries
    formatted_text: str = field(init=False)
    full_text: str = field(init=False)
    raw_response: str | None = None
    extracted_response: str | None = None
    execution_time: float | None = None
    input_tokens: int | None = None  # N/A for huggingface pipeline
    output_tokens: int | None = None  # N/A for huggingface pipeline
    scores: defaultdict[str, list] = field(default_factory=lambda: defaultdict(list))

    def __post_init__(self):
        self.formatted_text = f"Title: {self.title}\n\nAbstract: \n{self.abstract}"
        self.full_text = f"{self.title}\n\n{self.abstract}"
