"""One process, one IP, one budget.

Each client creating its own WeightGovernor is only half a fix. The exchange's
limit is per IP, so three clients with three private budgets of 3000 can spend
9000 against a 6000 ceiling and get the IP banned while every one of them
believes it stayed inside its allowance.

A process-wide governor is the correct scope precisely because the constraint is
per IP and a process has one outbound address. It is explicit and resettable
rather than an implicit import-time global, so tests can control it.
"""
import pytest

from src.exchange.rate_limit import (
    WeightGovernor, get_shared_governor, reset_shared_governor,
)


@pytest.fixture(autouse=True)
def clean_shared():
    reset_shared_governor()
    yield
    reset_shared_governor()


def test_repeated_calls_return_the_same_instance():
    first = get_shared_governor(max_weight_per_minute=6000, safety_fraction=0.5)
    second = get_shared_governor(max_weight_per_minute=6000, safety_fraction=0.5)

    assert first is second


async def test_the_budget_is_actually_shared():
    """The property that matters: weight spent through one reference is visible
    through the other."""
    a = get_shared_governor(max_weight_per_minute=100, safety_fraction=1.0)
    b = get_shared_governor(max_weight_per_minute=100, safety_fraction=1.0)

    await a.acquire(30)

    assert b.used_weight() == 30


def test_a_later_conflicting_configuration_does_not_silently_win():
    """Two components disagreeing about the ceiling is a real possibility, and
    the safe resolution is the stricter one -- never the more permissive.

    Silently keeping the first would be defensible; silently adopting a larger
    later value would not, because it would raise the ceiling after the earlier
    caller had already sized its behaviour to the smaller one.
    """
    strict = get_shared_governor(max_weight_per_minute=1000, safety_fraction=0.5)
    assert strict.ceiling == 500

    same = get_shared_governor(max_weight_per_minute=6000, safety_fraction=0.9)

    assert same is strict
    assert same.ceiling == 500, "a later, larger ceiling was adopted"


def test_a_later_stricter_configuration_is_applied():
    lenient = get_shared_governor(max_weight_per_minute=6000, safety_fraction=1.0)
    assert lenient.ceiling == 6000

    tightened = get_shared_governor(max_weight_per_minute=1000, safety_fraction=0.5)

    assert tightened is lenient, "still one instance"
    assert tightened.ceiling == 500, "the stricter ceiling must be adopted"


async def test_tightening_does_not_discard_weight_already_spent():
    """Otherwise a component that starts late could hand the process a fresh
    budget it has not earned."""
    gov = get_shared_governor(max_weight_per_minute=1000, safety_fraction=1.0)
    await gov.acquire(400)

    get_shared_governor(max_weight_per_minute=1000, safety_fraction=0.5)

    assert gov.used_weight() == 400


def test_reset_is_available_for_tests_only_and_works():
    first = get_shared_governor()
    reset_shared_governor()
    second = get_shared_governor()

    assert first is not second


def test_the_shared_governor_is_a_real_governor():
    gov = get_shared_governor()
    assert isinstance(gov, WeightGovernor)
