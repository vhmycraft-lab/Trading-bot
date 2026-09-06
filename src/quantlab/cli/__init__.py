"""``quantlab`` command line (Typer).

The CLI is the outermost layer: it parses arguments, builds the container and
translates :class:`~quantlab.core.errors.QuantLabError` into the exit codes of
master spec section 18.2.

Global options accepted before any subcommand::

    quantlab --config configs/x.yaml --set optimize.n_trials=50 config show
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from quantlab import __version__
from quantlab.adapters.store.sqlite import (
    create_db_engine,
    current_revision,
    missing_tables,
    upgrade_to_head,
)
from quantlab.cli import backtest as backtest_module
from quantlab.cli import data as data_module
from quantlab.cli import doctor as doctor_module
from quantlab.cli import evolve as evolve_module
from quantlab.cli import optimize as optimize_module
from quantlab.cli import report as report_module
from quantlab.cli import validate as validate_module
from quantlab.cli._exit import QuantLabGroup
from quantlab.container import PROFILES, Container, Profile, build_container
from quantlab.core.config import AppConfig, load_config
from quantlab.core.errors import ConfigError
from quantlab.core.logging import configure_logging, get_logger

__all__ = ["CliState", "QuantLabGroup", "app", "main"]

console = Console()
log = get_logger("quantlab.cli")


@dataclass
class CliState:
    """Global options, plus lazily-loaded config and container."""

    config_paths: list[Path] = field(default_factory=list)
    overrides: list[str] = field(default_factory=list)
    profile: Profile = "research"
    _config: AppConfig | None = None
    _container: Container | None = None

    @property
    def config(self) -> AppConfig:
        if self._config is None:
            self._config = load_config(self.config_paths, self.overrides)
            self._configure_logging(self._config)
            for override in self.overrides:
                log.info("config_override", override=override)
        return self._config

    @property
    def container(self) -> Container:
        if self._container is None:
            self._container = build_container(self.config, profile=self.profile)
        return self._container

    @staticmethod
    def _configure_logging(config: AppConfig) -> None:
        configure_logging(
            level=config.logging.level,
            json_output=config.logging.json_output,
            redact_keys=config.logging.redact_keys,
            log_dir=Path(config.project.artifacts_dir) / "logs",
            log_date=dt.datetime.now(dt.UTC).date(),
        )


app = typer.Typer(
    cls=QuantLabGroup,
    name="quantlab",
    help="AI-assisted quantitative trading research and paper trading. Never places real orders.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
config_app = typer.Typer(help="Inspect the resolved configuration.", no_args_is_help=True)
db_app = typer.Typer(help="Manage the experiment database.", no_args_is_help=True)
app.add_typer(config_app, name="config")
app.add_typer(db_app, name="db")
app.add_typer(data_module.app, name="data")
app.add_typer(backtest_module.app, name="backtest")
app.add_typer(report_module.app, name="report")
app.add_typer(optimize_module.app, name="optimize")
app.add_typer(evolve_module.app, name="evolve")
app.add_typer(validate_module.app, name="validate")


def _version_callback(value: bool) -> None:
    if value:
        console.print(__version__)
        raise typer.Exit(0)


@app.callback()
def main(
    ctx: typer.Context,
    config: Annotated[
        list[Path] | None,
        typer.Option("--config", "-c", help="YAML file merged over configs/default.yaml."),
    ] = None,
    set_: Annotated[
        list[str] | None,
        typer.Option("--set", "-s", metavar="SECTION.KEY=VALUE", help="Override one value."),
    ] = None,
    profile: Annotated[
        str,
        typer.Option("--profile", help=f"Container profile: {', '.join(PROFILES)}."),
    ] = "research",
    _version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="Print the version and exit.",
        ),
    ] = False,
) -> None:
    """Set up global state shared by every subcommand."""
    if profile not in PROFILES:
        raise ConfigError("unknown container profile", profile=profile, allowed=list(PROFILES))
    ctx.obj = CliState(
        config_paths=list(config or []),
        overrides=list(set_ or []),
        profile=profile,  # type: ignore[arg-type]  # validated in build_container
    )


@app.command()
def version() -> None:
    """Print the QuantLab version."""
    console.print(__version__)


@app.command("doctor")
def doctor_command(ctx: typer.Context) -> None:
    """Check the interpreter, configuration, directories, secrets and database."""
    state: CliState = ctx.obj
    checks = doctor_module.run_checks(state.config, state.container.secrets)
    doctor_module.render(checks, console)
    raise typer.Exit(doctor_module.worst_exit_code(checks))


@config_app.command("show")
def config_show(
    ctx: typer.Context,
    indent: Annotated[
        int, typer.Option(help="Pretty-print indentation; 0 for canonical JSON.")
    ] = 2,
) -> None:
    """Print the fully-resolved configuration as JSON."""
    state: CliState = ctx.obj
    if indent <= 0:
        console.print_json(state.config.resolved_json())
    else:
        console.print_json(json.dumps(state.config.resolved_dict(), indent=indent))


@config_app.command("hash")
def config_hash(ctx: typer.Context) -> None:
    """Print the config_hash stored with every experiment."""
    state: CliState = ctx.obj
    console.print(state.config.config_hash)


@db_app.command("upgrade")
def db_upgrade(ctx: typer.Context) -> None:
    """Apply all pending migrations to the configured database."""
    state: CliState = ctx.obj
    engine = state.container.db_engine
    revision = upgrade_to_head(engine)
    console.print(f"database at [bold]{engine.url.database}[/] upgraded to revision {revision}")


@db_app.command("info")
def db_info(ctx: typer.Context) -> None:
    """Show the database path, size, revision and any missing tables."""
    state: CliState = ctx.obj
    db_path = Path(state.config.project.db_path)
    console.print(f"path      : {db_path}")
    if not db_path.exists():
        console.print("state     : not created yet; run `quantlab db upgrade`")
        return
    console.print(f"size      : {db_path.stat().st_size} bytes")
    engine = create_db_engine(db_path)
    try:
        console.print(f"revision  : {current_revision(engine) or 'unmanaged'}")
        absent = missing_tables(engine)
    finally:
        engine.dispose()
    console.print(f"tables    : {'all present' if not absent else 'missing ' + ', '.join(absent)}")
