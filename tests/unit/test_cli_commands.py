"""Every CLI command must at least be constructible.

A dependency resolution produced click 8.4 against the pinned typer 0.12.3, and
the combination raised at command-registration time:

    TypeError: Secondary flag is not valid for non-boolean flag.

Every single CLI invocation died -- `--help` included -- while the entire unit
test suite passed, because no test had ever gone through the typer app. The
tests exercised the functions the commands call, never the commands.

These tests close that gap: they walk the app's registered commands and ask each
one for its help text, which is enough to force typer to build the click
parameter objects. That is exactly where the breakage was.

They are also the local mirror of the CI smoke step. CI catches this on Linux,
but a developer should not have to push to discover that the CLI cannot start.
"""
import typer
from typer.testing import CliRunner

from src.cli.main import app

runner = CliRunner()

# Registration is what breaks, so the list is derived from the app rather than
# hardcoded -- a new command is covered the moment it is added.
COMMAND_NAMES = sorted(
    (cmd.name or cmd.callback.__name__.replace("_", "-"))
    for cmd in app.registered_commands
)


def test_the_app_registers_the_expected_commands():
    """A guard on the guard: if this list ever empties, the loop below would
    pass while testing nothing."""
    assert len(COMMAND_NAMES) >= 9, (
        f"expected the full command set, found {COMMAND_NAMES}"
    )


def test_top_level_help_works():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, _describe(result)


def test_every_command_help_works():
    """One test over all commands, so a failure names the command that broke."""
    broken = {}
    for name in COMMAND_NAMES:
        result = runner.invoke(app, [name, "--help"])
        if result.exit_code != 0:
            broken[name] = _describe(result)

    assert not broken, "commands that cannot even print help: " + "; ".join(
        f"{k}: {v}" for k, v in broken.items()
    )


def test_a_boolean_toggle_option_keeps_both_flags():
    """The specific shape that broke: a `--x/--no-x` pair.

    `spike-run` declares `--evaluate/--no-evaluate`. Under the broken dependency
    combination, building this parameter raised before any argument was parsed.

    Asserting on the click parameter object rather than on help text is
    deliberate: rendered help is wrapped to the runner's terminal width, so a
    text assertion would pass or fail depending on the width rather than on
    whether the flag exists.
    """
    group = typer.main.get_command(app)
    command = group.commands["spike-run"]

    option = next(
        (p for p in command.params if "--evaluate" in getattr(p, "opts", [])), None
    )
    assert option is not None, (
        f"--evaluate is gone; spike-run params: "
        f"{[getattr(p, 'opts', None) for p in command.params]}"
    )
    assert option.is_bool_flag, "the toggle must remain a boolean flag"
    assert "--no-evaluate" in option.secondary_opts, (
        "the negative half of the toggle must survive parameter construction"
    )
    assert option.default is True, "scanning should evaluate by default"


def _describe(result) -> str:
    exc = result.exception
    detail = f"{type(exc).__name__}: {exc}" if exc else "no exception"
    return f"exit={result.exit_code} {detail}\n{result.output}"
