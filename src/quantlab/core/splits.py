"""Train / validation / test partitioning and walk-forward windows (spec sections 5, 15).

The split is the backbone of every honest claim this platform makes:

* **train** is the only segment a parameter search may touch
* an **embargo** of ``embargo_bars`` sits between segments so that a strategy
  holding a position at the end of one cannot leak information into the next
* **validation** is for out-of-sample checking, never for optimisation
* **test** is the lockbox: it is not loaded at all outside the ``lockbox``
  profile (INV-5), and its end is frozen the first time it is used so every
  lockbox evaluation of a split covers the same period

Changing any boundary here produces a different :attr:`SplitPolicy.split_id`,
so old verdicts stay bound to the split they were computed on and cannot be
silently re-interpreted under new boundaries.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import yaml

from quantlab.core.errors import ConfigError, LockboxViolation
from quantlab.core.hashing import canonical_json, short_id
from quantlab.core.types import format_ts, timeframe_ms

__all__ = [
    "SEGMENT_NAMES",
    "Segment",
    "SegmentName",
    "SplitPolicy",
    "WalkForwardWindow",
    "describe_windows",
    "load_split_policy",
    "parse_split_policy",
    "segment_bar_counts",
    "walk_forward_windows",
]

SegmentName = Literal["train", "val", "test"]

#: The three partitions, in chronological order.
SEGMENT_NAMES: Final[tuple[SegmentName, ...]] = ("train", "val", "test")


@dataclass(frozen=True, slots=True)
class Segment:
    """A closed range of bar open times, ``[start_ts, end_ts]``.

    ``end_ts`` is ``None`` only for the test segment before it is frozen.
    """

    name: str
    start_ts: int
    end_ts: int | None

    def contains(self, ts: int) -> bool:
        if ts < self.start_ts:
            return False
        return self.end_ts is None or ts <= self.end_ts

    def n_bars(self, bar_ms: int) -> int | None:
        """Number of bars in the segment, or ``None`` if the end is open."""
        if self.end_ts is None:
            return None
        return (self.end_ts - self.start_ts) // bar_ms + 1

    def describe(self, bar_ms: int) -> str:
        count = self.n_bars(bar_ms)
        return (
            f"{self.name:<5} {format_ts(self.start_ts)} .. {format_ts(self.end_ts)}"
            f"  {'open-ended' if count is None else f'{count} bars'}"
        )


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    """One in-sample / out-of-sample pair (spec section 15)."""

    k: int
    is_start_ts: int
    is_end_ts: int
    oos_start_ts: int
    oos_end_ts: int

    @property
    def is_segment(self) -> str:
        return f"wf_is:{self.k}"

    @property
    def oos_segment(self) -> str:
        return f"wf_oos:{self.k}"

    def describe(self) -> str:
        return (
            f"window {self.k:>2}  IS {format_ts(self.is_start_ts)} .. {format_ts(self.is_end_ts)}"
            f"  OOS {format_ts(self.oos_start_ts)} .. {format_ts(self.oos_end_ts)}"
        )


@dataclass(frozen=True, slots=True)
class SplitPolicy:
    """A resolved, hashable train/validation/test partition.

    Instances are immutable.  Freezing the test end produces a *new* policy with
    a new :attr:`split_id`, which is exactly the intent: the frozen and unfrozen
    partitions are different objects of study.
    """

    symbol: str
    timeframe: str
    train_start_ts: int
    train_end_ts: int
    val_start_ts: int
    val_end_ts: int
    test_start_ts: int
    test_end_ts: int | None
    embargo_bars: int
    dataset_id: str = ""
    #: Canonical JSON of the policy file, kept so ``split_id`` is reproducible.
    source_json: str = ""

    # -- identity ----------------------------------------------------------
    @property
    def split_id(self) -> str:
        """``sha256(canonical policy + dataset_id)[:16]`` (spec section 6)."""
        payload = self.source_json or canonical_json(self.to_dict())
        return short_id(f"{payload}{self.dataset_id}")

    @property
    def bar_ms(self) -> int:
        return timeframe_ms(self.timeframe)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-compatible view, used for hashing and for storage."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "train": {"start": self.train_start_ts, "end": self.train_end_ts},
            "validation": {"start": self.val_start_ts, "end": self.val_end_ts},
            "test": {"start": self.test_start_ts, "end": self.test_end_ts},
            "embargo_bars": self.embargo_bars,
        }

    # -- segments ----------------------------------------------------------
    @property
    def train(self) -> Segment:
        return Segment("train", self.train_start_ts, self.train_end_ts)

    @property
    def val(self) -> Segment:
        return Segment("val", self.val_start_ts, self.val_end_ts)

    @property
    def test(self) -> Segment:
        return Segment("test", self.test_start_ts, self.test_end_ts)

    def segment(self, name: str) -> Segment:
        """Return one segment by name.

        Raises:
            ConfigError: if ``name`` is not train, val or test.
        """
        mapping = {"train": self.train, "val": self.val, "test": self.test}
        try:
            return mapping[name]
        except KeyError:
            raise ConfigError(
                "unknown segment", segment=name, allowed=list(SEGMENT_NAMES)
            ) from None

    def segments(self) -> tuple[Segment, ...]:
        return (self.train, self.val, self.test)

    # -- the lockbox boundary (INV-5) --------------------------------------
    def intersects_test(self, start_ts: int | None, end_ts: int | None) -> bool:
        """True if ``[start_ts, end_ts]`` reaches into the test partition.

        An open-ended request (``end_ts is None``) always reaches it: the test
        partition is open at the right until it is frozen.
        """
        if start_ts is not None and end_ts is not None and int(start_ts) > int(end_ts):
            return False  # an empty range reaches nothing
        if end_ts is None:
            return True
        return int(end_ts) >= self.test_start_ts

    def assert_no_test_data(self, start_ts: int | None, end_ts: int | None) -> None:
        """Raise if the requested range would expose test-partition bars.

        Raises:
            LockboxViolation: never caught except at the CLI top level.
        """
        if self.intersects_test(start_ts, end_ts):
            raise LockboxViolation(
                "requested range reaches into the locked test partition",
                requested_start=format_ts(start_ts),
                requested_end=format_ts(end_ts),
                test_start=format_ts(self.test_start_ts),
                split_id=self.split_id,
            )

    def freeze_test_end(self, end_ts: int) -> SplitPolicy:
        """Return a copy with the test end pinned to ``end_ts``.

        Raises:
            ConfigError: if the end is already frozen to a different value, or
                lies before the start of the test partition.
        """
        if self.test_end_ts is not None and self.test_end_ts != end_ts:
            raise ConfigError(
                "the test end is already frozen and must not be moved",
                frozen_at=format_ts(self.test_end_ts),
                requested=format_ts(end_ts),
            )
        if end_ts < self.test_start_ts:
            raise ConfigError(
                "the test end must not precede the test start",
                start=format_ts(self.test_start_ts),
                requested=format_ts(end_ts),
            )
        return replace(self, test_end_ts=int(end_ts))

    def describe(self) -> str:
        """Multi-line summary printed by ``quantlab data splits``."""
        bar_ms = self.bar_ms
        lines = [
            f"split_id   : {self.split_id}",
            f"symbol     : {self.symbol}  timeframe: {self.timeframe}",
            f"dataset_id : {self.dataset_id or '(unbound)'}",
            f"embargo    : {self.embargo_bars} bars",
            "segments   :",
        ]
        lines.extend(f"  {segment.describe(bar_ms)}" for segment in self.segments())
        return "\n".join(lines)


def _require(mapping: Mapping[str, Any], key: str, *, where: str) -> Any:
    if key not in mapping:
        raise ConfigError("split policy is missing a key", key=key, section=where)
    return mapping[key]


def _range(mapping: Mapping[str, Any], key: str) -> tuple[int, int | None]:
    from quantlab.core.types import to_ms  # local import keeps the module import graph flat

    section = _require(mapping, key, where="policy")
    if not isinstance(section, Mapping):
        raise ConfigError("split range must be a mapping with start and end", section=key)
    start = _require(section, "start", where=key)
    end = section.get("end")
    if start is None:
        raise ConfigError("split range start must not be null", section=key)
    return to_ms(start), (None if end is None else to_ms(end))


def parse_split_policy(
    document: Mapping[str, Any],
    *,
    dataset_id: str = "",
    source_json: str | None = None,
) -> SplitPolicy:
    """Build a :class:`SplitPolicy` from a parsed policy document.

    Raises:
        ConfigError: on a missing key, a non-chronological boundary, or an
            embargo that the configured start dates do not actually honour.
    """
    symbol = str(_require(document, "symbol", where="policy"))
    timeframe = str(_require(document, "timeframe", where="policy"))
    bar_ms = timeframe_ms(timeframe)

    train_start, train_end = _range(document, "train")
    val_start, val_end = _range(document, "validation")
    test_start, test_end = _range(document, "test")
    embargo_bars = int(document.get("embargo_bars", 0))

    if train_end is None or val_end is None:
        raise ConfigError("only the test segment may have an open end")
    if embargo_bars < 0:
        raise ConfigError("embargo_bars must not be negative", embargo_bars=embargo_bars)

    for name, start, end in (
        ("train", train_start, train_end),
        ("validation", val_start, val_end),
    ):
        if end < start:
            raise ConfigError(
                "segment end precedes its start",
                segment=name,
                start=format_ts(start),
                end=format_ts(end),
            )
    if test_end is not None and test_end < test_start:
        raise ConfigError("segment end precedes its start", segment="test")

    for ts, label in (
        (train_start, "train.start"),
        (train_end, "train.end"),
        (val_start, "validation.start"),
        (val_end, "validation.end"),
        (test_start, "test.start"),
    ):
        if ts % bar_ms != 0:
            raise ConfigError(
                "split boundary is not aligned to the timeframe grid",
                boundary=label,
                ts=format_ts(ts),
                timeframe=timeframe,
            )

    # The first bar after a segment is end + 1 bar; `embargo_bars` of them are
    # discarded, so the next segment may start one bar after those.
    for earlier_end, later_start, pair in (
        (train_end, val_start, "train -> validation"),
        (val_end, test_start, "validation -> test"),
    ):
        earliest = earlier_end + bar_ms * (embargo_bars + 1)
        if later_start < earliest:
            raise ConfigError(
                "segments are closer than the embargo allows",
                boundary=pair,
                embargo_bars=embargo_bars,
                earliest_allowed=format_ts(earliest),
                configured=format_ts(later_start),
            )

    return SplitPolicy(
        symbol=symbol,
        timeframe=timeframe,
        train_start_ts=train_start,
        train_end_ts=train_end,
        val_start_ts=val_start,
        val_end_ts=val_end,
        test_start_ts=test_start,
        test_end_ts=test_end,
        embargo_bars=embargo_bars,
        dataset_id=dataset_id,
        source_json=source_json if source_json is not None else canonical_json(dict(document)),
    )


def load_split_policy(path: str | Path, *, dataset_id: str = "") -> SplitPolicy:
    """Load and validate a split policy YAML file.

    Raises:
        ConfigError: if the file is missing, is not a YAML mapping, or fails
            any of the checks in :func:`parse_split_policy`.
    """
    file = Path(path)
    if not file.is_file():
        raise ConfigError("split policy file not found", path=str(file))
    try:
        document = yaml.safe_load(file.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError("split policy is not valid YAML", path=str(file)) from exc
    if not isinstance(document, Mapping):
        raise ConfigError("split policy must be a mapping", path=str(file))
    return parse_split_policy(document, dataset_id=dataset_id)


def _usable_timeline(policy: SplitPolicy) -> tuple[np.ndarray, int]:
    """Bar open times of train + validation, with the embargo removed.

    Returns the timeline and the index at which validation begins.  Because the
    embargoed bars are simply absent, no walk-forward window can contain one.
    """
    bar_ms = policy.bar_ms
    train = np.arange(policy.train_start_ts, policy.train_end_ts + bar_ms, bar_ms, dtype="int64")
    val = np.arange(policy.val_start_ts, policy.val_end_ts + bar_ms, bar_ms, dtype="int64")
    return np.concatenate([train, val]), int(train.size)


def walk_forward_windows(
    policy: SplitPolicy,
    *,
    is_bars: int,
    oos_bars: int,
    step_bars: int,
    scheme: Literal["rolling", "anchored"] = "rolling",
) -> tuple[WalkForwardWindow, ...]:
    """Generate walk-forward windows over train + validation (spec section 15.1).

    Windows advance by ``step_bars`` over the concatenated train and validation
    bars.  The embargo is excluded from that timeline, so no window can straddle
    it — the property spec section 15.1 asks for — and the last window ends at
    the end of validation.

    ``anchored`` keeps every in-sample window starting at ``train_start``;
    ``rolling`` moves it forward with the out-of-sample window.

    Raises:
        ConfigError: on a non-positive size, or if the timeline is too short to
            hold even one window.
    """
    if min(is_bars, oos_bars, step_bars) <= 0:
        raise ConfigError(
            "walk-forward sizes must be positive",
            is_bars=is_bars,
            oos_bars=oos_bars,
            step_bars=step_bars,
        )

    timeline, _val_index = _usable_timeline(policy)
    total = int(timeline.size)
    span = is_bars + oos_bars
    if total < span:
        raise ConfigError(
            "train + validation is too short for one walk-forward window",
            available_bars=total,
            required_bars=span,
        )

    windows: list[WalkForwardWindow] = []
    for k, start in enumerate(range(0, total - span + 1, step_bars)):
        is_start_index = 0 if scheme == "anchored" else start
        is_end_index = start + is_bars - 1
        oos_start_index = start + is_bars
        oos_end_index = start + span - 1
        windows.append(
            WalkForwardWindow(
                k=k,
                is_start_ts=int(timeline[is_start_index]),
                is_end_ts=int(timeline[is_end_index]),
                oos_start_ts=int(timeline[oos_start_index]),
                oos_end_ts=int(timeline[oos_end_index]),
            )
        )
    return tuple(windows)


def segment_bar_counts(policy: SplitPolicy) -> Mapping[str, int | None]:
    """Bars per segment, for reporting."""
    bar_ms = policy.bar_ms
    return {segment.name: segment.n_bars(bar_ms) for segment in policy.segments()}


def describe_windows(windows: Sequence[WalkForwardWindow]) -> str:
    """Render walk-forward windows for the CLI."""
    return "\n".join(window.describe() for window in windows)
