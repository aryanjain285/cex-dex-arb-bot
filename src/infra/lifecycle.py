"""Cooperative shutdown for a long-running, money-touching process.

No signal handling existed anywhere in this codebase. Only `KeyboardInterrupt`
(SIGINT) was caught, while both documented deployment paths stop with
**SIGTERM** -- `systemd` with `Restart=always`, and `docker stop`. Python's
default SIGTERM disposition terminates the interpreter without unwinding, so
the `finally: await shutdown()` never ran. On every stop, deploy, restart and
OOM-kill the listenKey was never deleted, sockets were dropped without close
frames, and un-persisted state was lost.

The pieces here are deliberately small and synchronous in effect: a flag the
main loop checks, a handler installer that degrades on platforms without
signal support, and a bounded drain. A shutdown that waits indefinitely is a
hang; one that waits not at all abandons in-flight orders.
"""
from __future__ import annotations

import asyncio
import signal
from typing import Iterable, List, Optional, Sequence

from loguru import logger

__all__ = ["ShutdownSignal", "install_signal_handlers", "drain"]


class ShutdownSignal:
    """A one-way latch the main loop polls between iterations.

    Deliberately not an `asyncio.Event`: the loop must be able to check it
    synchronously at the top of each pass without an await point, and the
    reason must survive for the shutdown log.
    """

    def __init__(self) -> None:
        self._requested = False
        self._reason: Optional[str] = None

    @property
    def requested(self) -> bool:
        return self._requested

    @property
    def reason(self) -> Optional[str]:
        return self._reason

    def request(self, reason: str) -> bool:
        """Latch a shutdown request.

        Returns True if this call caused the transition, False if a shutdown
        was already pending. Idempotence matters: a supervisor sending a second
        SIGTERM must not trigger a second shutdown sequence, and the first
        reason is the one that explains what happened.
        """
        if self._requested:
            logger.info(f"Shutdown already requested; ignoring duplicate ({reason}).")
            return False
        self._requested = True
        self._reason = reason
        logger.warning(f"Shutdown requested: {reason}. Finishing the current cycle.")
        return True


def install_signal_handlers(loop, shutdown: ShutdownSignal) -> List[signal.Signals]:
    """Route termination signals into `shutdown`.

    Returns the signals successfully hooked. Windows has no
    `loop.add_signal_handler`, so installation degrades to an empty list rather
    than raising -- the same code path must work in development and in
    production, and a development machine that cannot install handlers should
    not be a different program.
    """
    installed: List[signal.Signals] = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, shutdown.request, f"signal {sig.name}")
            installed.append(sig)
        except (NotImplementedError, AttributeError, ValueError, RuntimeError) as exc:
            logger.warning(
                f"Cannot install a {getattr(sig, 'name', sig)} handler on this "
                f"platform ({type(exc).__name__}); relying on KeyboardInterrupt."
            )
    if installed:
        logger.info(
            f"Shutdown handlers installed for: "
            f"{', '.join(s.name for s in installed)}."
        )
    return installed


async def drain(tasks: Sequence[asyncio.Task], timeout: float) -> bool:
    """Wait for in-flight tasks, then cancel whatever is left.

    Returns True if everything finished within `timeout`. A task that overruns
    is cancelled and awaited, so it is never left orphaned holding a socket or
    a half-written file.
    """
    pending = [t for t in tasks if t is not None and not t.done()]
    if not pending:
        return True

    logger.info(f"Draining {len(pending)} in-flight task(s), up to {timeout}s.")
    done, still_pending = await asyncio.wait(pending, timeout=timeout)

    if not still_pending:
        logger.info("Drain complete; all in-flight work finished.")
        return True

    logger.warning(
        f"Drain timed out with {len(still_pending)} task(s) outstanding; "
        f"cancelling them."
    )
    for task in still_pending:
        task.cancel()
    # Await the cancellations so cleanup actually runs before we tear down the
    # session underneath them.
    await asyncio.gather(*still_pending, return_exceptions=True)
    return False
