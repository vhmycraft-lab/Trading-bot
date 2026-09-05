"""``python -m quantlab`` — the same entry point as the installed console script.

Exists so that a subprocess can invoke the CLI without depending on the console
script being on ``PATH``: the integration tests of section 20 reproduce a run *in
a fresh process*, and a test that shelled out to a script the environment may not
have installed would be testing the environment rather than the platform.
"""

from __future__ import annotations

from quantlab.cli import app

if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    app()
