"""Entry point for `python -m bot ...`.

Parsing and dispatch live in bot/cli.py; the subcommand
implementations live in bot/commands.py. This module exists so the
package has a __main__ to execute.
"""
from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
