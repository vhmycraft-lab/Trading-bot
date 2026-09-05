"""Ports: the structural interfaces the rest of the system is written against.

Import rule (INV-8): ``ports`` imports only ``core``.  Every port is a
:class:`typing.Protocol`, so an adapter satisfies it structurally and can be
swapped by adding one branch in :mod:`quantlab.container`.
"""

from __future__ import annotations
