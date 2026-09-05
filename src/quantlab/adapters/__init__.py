"""Adapters: concrete implementations of the ports.

Import rule (INV-8): adapters may import ``core`` and ``ports``.  Nothing may
import ``quantlab.adapters`` except :mod:`quantlab.container`, :mod:`quantlab.cli`
and the tests — ``tests/unit/test_architecture.py`` enforces that.
"""

from __future__ import annotations
