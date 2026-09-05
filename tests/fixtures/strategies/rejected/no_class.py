"""Declares no strategy class at all."""

from quantlab.core.strategy import ParamSpec

DEFAULT_N = ParamSpec(kind="int", default=20, low=5, high=100)
