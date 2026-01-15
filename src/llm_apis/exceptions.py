class RefusalError(RuntimeError):
    """Raised when the LLM refuses to generate a summary."""
    pass

class NoContentError(RuntimeError):
    """Raised when the LLM returns no content."""
    pass

class UnknownResponse(RuntimeError):
    """Raised when the LLM returns an unknown response."""
    pass