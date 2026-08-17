class BotError(Exception):
    """Base exception for the application."""
    pass

class ConfigurationError(BotError):
    """Raised for configuration-related errors."""
    pass

class ExchangeError(BotError):
    """Raised for exchange-related errors (API, connection, etc.)."""
    pass

class CexError(ExchangeError):
    """Raised for CEX-specific errors."""
    pass

class DexError(ExchangeError):
    """Raised for DEX-specific errors."""
    pass

class RiskManagementError(BotError):
    """Raised when a risk limit is breached."""
    pass

class HedgingError(BotError):
    """Raised when an attempt to hedge a failed trade also fails."""
    def __init__(self, message, original_summary, hedge_summary):
        super().__init__(message)
        self.original_summary = original_summary
        self.hedge_summary = hedge_summary

class InsufficientFundsError(BotError):
    """Raised when there are not enough funds to perform a trade."""
    pass
