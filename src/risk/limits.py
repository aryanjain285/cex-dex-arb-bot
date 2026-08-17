"""Risk state and pre-trade gating.

Two design commitments, both the result of audit findings:

- **Money is persisted as a decimal string, never a float.** Consistent with
  `infra.evaluation_store`, so the on-disk financial record is bit-identical
  to the value the process computed and can be reconciled by exact equality.

- **Corrupt state fails closed.** The previous behaviour caught a parse error,
  reset `daily_pnl` to zero, logged, and kept trading -- which meant a
  well-timed crash during a write silently restored the full daily loss
  budget. A loss record that resets itself is worse than no loss record, so
  unreadable state now refuses to start.
"""
import json
import os
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

from loguru import logger

from src.infra import metrics
from src.core.config import RiskConfig
from src.core.types import ExecutionSummary, Opportunity

STATE_FILE = Path("data/risk_state.json")

ZERO = Decimal("0")


class RiskStateError(RuntimeError):
    """Raised when persisted risk state cannot be trusted.

    Deliberately fatal: an operator must inspect and re-arm rather than have
    the system quietly resume with a fresh loss allowance.
    """


def get_current_date_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


_UNSET = object()


class RiskManager:
    def __init__(self, config: RiskConfig, state_path=_UNSET):
        """`state_path=None` keeps all state in memory and persists nothing.

        The path used to be a module constant, so every RiskManager in the
        process shared one file -- and a backtest therefore read and wrote the
        LIVE bot's daily loss budget. Replaying a losing day consumed the live
        allowance and could halt the live bot for a loss that happened only in a
        simulation; replaying a winning day inflated the allowance, which is
        worse, because the daily loss limit is the last line of defence before
        capital is gone.

        Simulations pass None. Omitting the argument keeps the shared live file,
        so a halt still survives a restart.
        """
        self.config = config
        self.state_path = STATE_FILE if state_path is _UNSET else state_path
        self.positions: dict = {}
        self.daily_pnl: Decimal = ZERO
        self.halted: bool = False
        self.halt_reason: str = ""
        self._load_state()
        logger.info(
            f"Risk manager initialised. PnL today: {self.daily_pnl:.4f} "
            f"(state: {self.state_path if self.state_path else 'in memory only'})"
        )

    # ------------------------------------------------------------------

    def _load_state(self) -> None:
        if self.state_path is None:
            # A simulation starts clean by construction. Inheriting the live
            # day's PnL or halt would make a replay's result depend on what the
            # live bot happened to do earlier today.
            logger.info("Risk state is in-memory only; starting clean.")
            return

        if not self.state_path.exists():
            logger.warning("No risk state file found; starting from a clean state.")
            return

        try:
            raw = self.state_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RiskStateError(f"Cannot read {self.state_path}: {exc}") from exc

        try:
            state = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RiskStateError(
                f"{self.state_path} is corrupt ({exc}). Refusing to start: resetting "
                f"the daily loss budget on a corrupt file would defeat the loss "
                f"limit. Inspect the file, restore it, or delete it deliberately."
            ) from exc

        try:
            stored_pnl = Decimal(str(state.get("daily_pnl", "0")))
        except (InvalidOperation, TypeError) as exc:
            raise RiskStateError(
                f"{self.state_path} holds an unparseable daily_pnl "
                f"({state.get('daily_pnl')!r}). Refusing to start."
            ) from exc

        if state.get("date") == get_current_date_str():
            self.daily_pnl = stored_pnl
            self.halted = bool(state.get("halted", False))
            self.halt_reason = str(state.get("halt_reason", ""))
            logger.info("Loaded today's risk state successfully.")
            if self.halted:
                logger.error(
                    f"Risk state is HALTED: {self.halt_reason}. "
                    f"Manual re-arm required before trading resumes."
                )
        else:
            logger.info("New UTC day detected; resetting daily PnL.")
            self.daily_pnl = ZERO

        self.positions = state.get("positions", {})

    def _save_state(self) -> None:
        """Persist atomically: write a sibling temp file, then os.replace.

        A plain open("w") truncates before writing, so a crash mid-write left
        a partial file -- which the loader then treated as a reason to zero the
        loss budget. os.replace is atomic on POSIX and Windows.
        """
        payload = {
            "date": get_current_date_str(),
            # str(Decimal) is exact and losslessly reversible; float is not.
            "daily_pnl": str(self.daily_pnl),
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "positions": self.positions,
        }
        try:
            metrics.risk_halted.set(1 if self.halted else 0)
            metrics.daily_pnl_quote.set(float(self.daily_pnl))
        except Exception:  # pragma: no cover - telemetry is never fatal
            pass

        if self.state_path is None:
            # Metrics above are still emitted: an in-memory run must remain
            # observable even though it leaves nothing on disk.
            return

        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.state_path)
        except Exception as exc:
            logger.error(f"Error saving risk state: {exc}")

    # ------------------------------------------------------------------

    def halt(self, reason: str) -> None:
        """Stop trading until an operator clears the state file.

        Persisted immediately, so a restart cannot clear a halt.
        """
        if not self.halted:
            logger.error(f"RISK HALT: {reason}")
        self.halted = True
        self.halt_reason = reason
        self._save_state()

    def is_trade_allowed(self, opp: Opportunity) -> bool:
        if self.halted:
            logger.warning(f"Trade rejected: risk manager is halted ({self.halt_reason}).")
            return False

        # Notional is the capital committed on the BUY leg, whichever venue
        # that is. Using cex_price unconditionally sized the wrong leg for
        # DEX_to_CEX, and diverged most exactly when the spread was widest.
        buy_price = opp.dex_price if opp.direction == "DEX_to_CEX" else opp.cex_price
        notional_value = buy_price * opp.size

        if notional_value > Decimal(str(self.config.max_notional_per_leg_quote)):
            logger.warning(
                f"Trade rejected: exceeds the per-leg notional limit "
                f"({notional_value:.2f} > {self.config.max_notional_per_leg_quote:.2f})."
            )
            return False

        limit = self.config.max_daily_loss_quote
        if limit is not None and self.daily_pnl <= -Decimal(str(limit)):
            self.halt(
                f"daily loss limit breached: {self.daily_pnl:.2f} <= -{limit:.2f}"
            )
            return False

        return True

    def update_state(self, summary: ExecutionSummary) -> None:
        if not summary.legs:
            return

        if summary.pnl_quote is not None:
            # Accumulates losses as negatives. This must be REALISED PnL once
            # execution is wired -- an expected value that already cleared a
            # non-negative floor can never record a loss.
            self.daily_pnl += summary.pnl_quote
            logger.info(f"Daily PnL updated: {self.daily_pnl:.4f}")

        limit = self.config.max_daily_loss_quote
        if limit is not None and self.daily_pnl <= -Decimal(str(limit)):
            self.halt(
                f"daily loss limit breached: {self.daily_pnl:.2f} <= -{limit:.2f}"
            )
            return

        self._save_state()
