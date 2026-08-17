import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from loguru import logger

from src.core.config import RiskConfig
from src.core.types import Opportunity, ExecutionSummary

STATE_FILE = Path("data/risk_state.json")

def get_current_date_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

class RiskManager:
    def __init__(self, config: RiskConfig):
        self.config = config
        self.positions: dict = {}
        self.daily_pnl: Decimal = Decimal("0")
        self._load_state()
        logger.info(f"Risk manager initialised. PnL today: {self.daily_pnl:.4f}")

    def _load_state(self):
        if not STATE_FILE.exists():
            logger.warning("No risk state file found; starting from a clean state.")
            return
        try:
            with STATE_FILE.open("r", encoding="utf-8") as f:
                state = json.load(f)

            if state.get("date") == get_current_date_str():
                self.daily_pnl = Decimal(str(state.get("daily_pnl", "0")))
                logger.info("Loaded today's risk state successfully.")
            else:
                logger.info("New UTC day detected; resetting daily PnL.")
                self.daily_pnl = Decimal("0")

            self.positions = state.get("positions", {})

        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Error reading the risk state file: {e}. Starting from a clean state.")
            self.daily_pnl = Decimal("0")
            self.positions = {}

    def _save_state(self):
        current_state = {
            "date": get_current_date_str(),
            "daily_pnl": float(self.daily_pnl),
            "positions": self.positions,
        }
        try:
            STATE_FILE.parent.mkdir(exist_ok=True)
            with STATE_FILE.open("w", encoding="utf-8") as f:
                json.dump(current_state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving risk state: {e}")

    def is_trade_allowed(self, opp: Opportunity) -> bool:
        notional_value = opp.cex_price * opp.size
        if notional_value > self.config.max_notional_per_leg_quote:
            logger.warning(f"Trade rejected: exceeds the per-leg notional limit ({notional_value:.2f} > {self.config.max_notional_per_leg_quote:.2f}).")
            return False
        return True

    def update_state(self, summary: ExecutionSummary):
        if not summary.legs:
            return

        if summary.pnl_quote is not None:
            self.daily_pnl += summary.pnl_quote
            logger.info(f"Daily PnL updated: {self.daily_pnl:.4f}")

        self._save_state()
