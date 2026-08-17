import sys
import orjson
from loguru import logger

from ..core.config import ObservabilityConfig

def setup_logging(config: ObservabilityConfig):
    """
    Configure loguru for structured logging.
    """
    logger.remove() # remove the default handler

    log_level = config.log_level.upper()
    redact_keys = config.redact_keys

    def serialize(record):
        """Serialise a log record to JSON, redacting sensitive fields."""
        subset = {
            "timestamp": record["time"].isoformat(),
            "level": record["level"].name,
            "message": record["message"],
            "name": record["name"],
            "function": record["function"],
            "line": record["line"],
        }
        # attach extra fields, redacting sensitive ones
        if record["extra"]:
            for key, value in record["extra"].items():
                if any(redact_key in key.lower() for redact_key in redact_keys):
                    subset[key] = "[REDACTED]"
                else:
                    subset[key] = value
        
        # handle exceptions
        if record["exception"]:
            subset["exception"] = str(record["exception"])

        return orjson.dumps(subset).decode('utf-8')

    def sink(message):
        """Write the serialised record to stdout."""
        serialized = serialize(logger.patch(lambda record: record.update(message=message)).__dict__)
        print(serialized, file=sys.stdout)

    # logger.add(sink, level=log_level, format="{message}")
    # using the plain format for readability during development
    logger.add(sys.stdout, level=log_level)

    # logger.info(f"Logging configured, level: {log_level}")
    return logger
