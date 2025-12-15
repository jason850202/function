class TransformerError(Exception):
    """Base transformer error."""


class SpecError(TransformerError):
    """Raised when an invalid specification is provided."""


class MergeCollisionError(TransformerError):
    """Raised when merge collisions cannot be resolved."""
