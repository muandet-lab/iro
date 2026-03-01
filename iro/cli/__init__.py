"""Typer CLI for IRO."""

from __future__ import annotations

import typer

from .train import register_commands

app = typer.Typer(help="IRO command line interface")


@app.callback()
def main() -> None:
    """IRO CLI root command."""


register_commands(app)

__all__ = ["app"]
