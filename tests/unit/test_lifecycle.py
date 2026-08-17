"""Process lifecycle: shutdown must actually run, and metrics must be honest.

Two audits found that no signal handling exists anywhere in `src/`. Only
`KeyboardInterrupt` (SIGINT) is caught, while both documented deployment paths
stop with **SIGTERM** -- `systemd` with `Restart=always`, and `docker stop`.
Default SIGTERM terminates the interpreter without unwinding, so the
`finally: await self.shutdown()` never runs: on every stop, deploy and restart
the listenKey is never deleted, sockets are dropped without close frames, and
un-persisted state is lost.

Separately, five Prometheus series are defined and never emitted, which makes
four of the five documented alert rules unfireable -- an absent series returns
no data, so the alert never fires and an operator reads silence as health.
"""
import asyncio
import signal

import pytest

from src.infra import metrics


# --------------------------------------------------------------------------
# graceful shutdown
# --------------------------------------------------------------------------

async def test_shutdown_requested_stops_the_loop_cooperatively():
    """A shutdown request must break the loop at the next iteration rather
    than killing the process mid-await."""
    from src.infra.lifecycle import ShutdownSignal

    sig = ShutdownSignal()
    assert not sig.requested

    iterations = 0
    while not sig.requested:
        iterations += 1
        if iterations == 3:
            sig.request("test")
        if iterations > 100:
            pytest.fail("shutdown never took effect")

    assert iterations == 3
    assert sig.reason == "test"


async def test_shutdown_signal_is_idempotent():
    """Two SIGTERMs in quick succession must not double-run shutdown."""
    from src.infra.lifecycle import ShutdownSignal

    sig = ShutdownSignal()
    assert sig.request("first") is True
    assert sig.request("second") is False, "already requested"
    assert sig.reason == "first", "the first reason is the real one"


async def test_signal_handlers_are_installed_for_sigterm_and_sigint():
    """SIGTERM is the one that matters: it is what systemd and docker send.

    Tested against a recording loop rather than the real one. Windows has no
    `add_signal_handler`, so a test that required real installation could only
    ever run on Linux -- and this codebase already shipped a Linux-fatal bug
    that was invisible here for exactly that reason. This asserts the logic
    (which signals we ask for, and that firing one latches the shutdown)
    independently of whether the host supports it.
    """
    from src.infra.lifecycle import ShutdownSignal, install_signal_handlers

    class RecordingLoop:
        def __init__(self):
            self.handlers = {}

        def add_signal_handler(self, sig, callback, *args):
            self.handlers[sig] = (callback, args)

    sig = ShutdownSignal()
    loop = RecordingLoop()
    installed = install_signal_handlers(loop, sig)

    assert signal.SIGTERM in installed, "SIGTERM handler must be installed"
    assert signal.SIGINT in installed
    assert set(loop.handlers) == {signal.SIGTERM, signal.SIGINT}

    # Firing SIGTERM must latch the shutdown with a legible reason.
    callback, args = loop.handlers[signal.SIGTERM]
    callback(*args)
    assert sig.requested
    assert "SIGTERM" in sig.reason


async def test_signal_handlers_work_on_the_real_loop_where_supported():
    """Complements the test above by exercising the real installation path
    wherever the platform allows it."""
    from src.infra.lifecycle import ShutdownSignal, install_signal_handlers

    loop = asyncio.get_running_loop()
    sig = ShutdownSignal()
    installed = install_signal_handlers(loop, sig)
    try:
        if not installed:
            pytest.skip("platform does not support asyncio signal handlers")
        assert signal.SIGTERM in installed
    finally:
        for s_ in installed:
            loop.remove_signal_handler(s_)


async def test_install_is_safe_where_signals_are_unsupported():
    """Windows lacks add_signal_handler; installation must degrade, not crash,
    so the same code path works in development and production."""
    from src.infra.lifecycle import ShutdownSignal, install_signal_handlers

    class NoSignals:
        def add_signal_handler(self, *a, **k):
            raise NotImplementedError

    installed = install_signal_handlers(NoSignals(), ShutdownSignal())
    assert installed == [], "unsupported platform yields no handlers, not an error"


async def test_drain_waits_for_in_flight_work_then_gives_up():
    """Shutdown must bound how long it waits: an indefinite drain is a hang,
    and no drain at all abandons in-flight orders."""
    from src.infra.lifecycle import drain

    slow = asyncio.Event()

    async def never_finishes():
        await slow.wait()

    task = asyncio.create_task(never_finishes())
    completed = await drain([task], timeout=0.05)
    assert completed is False, "must report that the drain timed out"
    assert task.cancelled() or task.done(), "the task must be cancelled, not orphaned"


async def test_drain_returns_true_when_work_finishes_in_time():
    from src.infra.lifecycle import drain

    async def quick():
        await asyncio.sleep(0.01)

    task = asyncio.create_task(quick())
    assert await drain([task], timeout=1.0) is True


# --------------------------------------------------------------------------
# honest metrics
# --------------------------------------------------------------------------

def test_no_metric_is_defined_without_being_emitted():
    """A defined-but-never-emitted series makes its alert rule unfireable.

    Every metric in the module must either be emitted somewhere in src/, or be
    deleted. Four of five documented alert rules referenced series that were
    never created.
    """
    import pathlib
    import re

    source_dir = pathlib.Path("src")
    all_source = "\n".join(
        p.read_text(encoding="utf-8")
        for p in source_dir.rglob("*.py")
        if p.name != "metrics.py"
    )

    metric_names = re.findall(
        r"^(\w+)\s*=\s*(?:Counter|Gauge|Histogram)\(",
        (source_dir / "infra" / "metrics.py").read_text(encoding="utf-8"),
        flags=re.M,
    )
    assert metric_names, "expected to find metric definitions"

    never_emitted = [
        name for name in metric_names
        if f"metrics.{name}" not in all_source and f"{name}.labels" not in all_source
    ]
    assert not never_emitted, (
        f"metrics defined but never emitted (delete them or emit them): "
        f"{never_emitted}"
    )


def test_cycle_observability_metrics_exist():
    """An operator must be able to distinguish 'quiet market' from 'wedged'."""
    assert hasattr(metrics, "evaluations_total")
    assert hasattr(metrics, "cycle_duration_seconds")
    assert hasattr(metrics, "book_age_seconds")
    assert hasattr(metrics, "feed_age_seconds")


def test_rejection_reasons_are_counted_not_just_logged():
    """The most common decision -- rejected below the floor -- previously
    produced no metric at any level, so a cycle that evaluated 100 directions
    and rejected all of them looked identical to a cycle that did nothing."""
    assert hasattr(metrics, "evaluations_total")
    labels = metrics.evaluations_total._labelnames
    assert "reason" in labels and "outcome" in labels
