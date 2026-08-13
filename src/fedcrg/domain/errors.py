"""Domain-specific exceptions."""


class FedCRGError(Exception):
    """Base exception for FedCRG failures."""


class ConfigurationError(FedCRGError):
    """Raised when configuration cannot be resolved or validated."""


class DataIntegrityError(FedCRGError):
    """Raised when dataset or split integrity is violated."""


class ImmutableRunError(FedCRGError):
    """Raised when application code tries to mutate a completed run."""
