"""Paper trading (master spec section 16).

Paper, and only paper. INV-1 is a property of the whole package: nothing here
imports a broker adapter, holds a credential, or has a code path that transmits
an order — see ``ports/broker.py`` for why the port itself makes that
unwritable rather than merely absent.

``paper`` may import ``core`` and ``ports`` only (INV-8).
"""

from __future__ import annotations

from quantlab.paper.broker import PaperBroker

__all__ = ["PaperBroker"]
