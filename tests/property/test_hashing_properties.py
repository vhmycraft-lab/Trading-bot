"""Property tests for canonical JSON (master spec section 19)."""

from __future__ import annotations

import json

from hypothesis import given
from hypothesis import strategies as st

from quantlab.core.hashing import ID_LENGTH, canonical_json, short_id

_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(10**12), max_value=10**12),
    st.floats(allow_nan=False, allow_infinity=False, width=64),
    st.text(max_size=40),
)
_json_values = st.recursive(
    _scalars,
    lambda children: st.one_of(
        st.lists(children, max_size=6),
        st.dictionaries(st.text(max_size=12), children, max_size=6),
    ),
    max_leaves=25,
)
_json_objects = st.dictionaries(st.text(max_size=12), _json_values, max_size=8)


@given(_json_objects)
def test_canonical_json_is_idempotent(payload: dict[str, object]) -> None:
    once = canonical_json(payload)
    assert canonical_json(json.loads(once)) == once


@given(_json_objects)
def test_canonical_json_is_key_order_independent(payload: dict[str, object]) -> None:
    reordered = dict(reversed(list(payload.items())))
    assert canonical_json(reordered) == canonical_json(payload)


@given(_json_objects)
def test_ids_are_stable_and_the_right_length(payload: dict[str, object]) -> None:
    identifier = short_id(canonical_json(payload))
    assert len(identifier) == ID_LENGTH
    assert identifier == short_id(canonical_json(payload))
    assert set(identifier) <= set("0123456789abcdef")


@given(st.text(max_size=64), st.text(max_size=64))
def test_distinct_inputs_rarely_collide(a: str, b: str) -> None:
    if a != b:
        assert short_id(a) != short_id(b)
