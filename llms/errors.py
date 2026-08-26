"""Provider-independent errors of the model layer."""


class LLMError(Exception):
    """Model call failed; the message is safe to show to the user."""


class LLMTimeoutError(LLMError):
    """Model did not answer within LLM_TIMEOUT_SECONDS."""
