"""The uvloop installation helper must not recurse.

Regression test for a defect introduced by a careless bulk find-and-replace:
every `uvloop.install()` call site was rewritten to
`install_event_loop_policy()`, including the one *inside* that function's own
body, turning it into unbounded self-recursion.

It was invisible on Windows because `uvloop` is unimportable there, so the
`if uvloop is not None` guard short-circuited before recursing -- while on
Linux, the only platform this is deployed to, every CLI command would die
with RecursionError. A test that only ever runs on the development machine
would not have caught it, so this one injects a stub uvloop to force the
guard open.
"""
import sys
import types

import pytest


def test_install_event_loop_policy_does_not_recurse(monkeypatch):
    from src.cli import main

    installed = []
    stub = types.SimpleNamespace(install=lambda: installed.append(True))
    monkeypatch.setattr(main, "uvloop", stub)

    # A low limit turns any accidental recursion into a fast, clear failure.
    original_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(200)
    try:
        main.install_event_loop_policy()
    finally:
        sys.setrecursionlimit(original_limit)

    assert installed == [True], "uvloop.install() should be called exactly once"


def test_install_event_loop_policy_is_a_noop_without_uvloop(monkeypatch):
    from src.cli import main

    monkeypatch.setattr(main, "uvloop", None)
    main.install_event_loop_policy()  # must not raise
