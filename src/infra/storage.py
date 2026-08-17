import orjson
from typing import Dict, Any
from loguru import logger

STATE_FILE_PATH = "data/state.json"

def save_state(state: Dict[str, Any]):
    """Persist the bot's state to a JSON file."""
    try:
        with open(STATE_FILE_PATH, 'wb') as f:
            f.write(orjson.dumps(state, option=orjson.OPT_INDENT_2))
        logger.debug(f"State saved to {STATE_FILE_PATH}")
    except Exception as e:
        logger.error(f"Failed to save state: {e}")

def load_state() -> Dict[str, Any]:
    """Load the bot's state from a JSON file."""
    try:
        with open(STATE_FILE_PATH, 'rb') as f:
            state = orjson.loads(f.read())
            # logger.info(f"Loaded previous state from {STATE_FILE_PATH}.")
            return state
    except FileNotFoundError:
        logger.warning(f"State file {STATE_FILE_PATH} not found; starting from an empty state.")
        return {}
    except Exception as e:
        logger.error(f"Failed to load state: {e}")
        return {}
