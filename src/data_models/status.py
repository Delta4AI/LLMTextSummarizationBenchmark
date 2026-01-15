from enum import Enum


class RunStatus(Enum):
    NOT_STARTED = "not started"
    SKIPPED = "skipped"
    OK = "ok"
    FAILED = "failed"
    NO_RESULTS = "no results"
