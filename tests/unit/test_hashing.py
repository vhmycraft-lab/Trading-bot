"""Canonical JSON and identity hashing (master spec section 1.3, 7.2, 11.2)."""

from __future__ import annotations

import json

from quantlab.core.hashing import (
    ID_LENGTH,
    canonical_json,
    dataset_id,
    file_sha256,
    run_id,
    sha256_hex,
    short_id,
    split_id,
    strategy_id,
)


def test_canonical_json_is_key_order_independent() -> None:
    a = {"b": 1, "a": {"z": 2, "y": [3, 4]}}
    b = {"a": {"y": [3, 4], "z": 2}, "b": 1}
    assert canonical_json(a) == canonical_json(b)


def test_canonical_json_uses_compact_separators() -> None:
    assert canonical_json({"a": 1, "b": [1, 2]}) == '{"a":1,"b":[1,2]}'


def test_canonical_json_keeps_unicode_unescaped() -> None:
    assert canonical_json({"k": "café"}) == '{"k":"café"}'


def test_canonical_json_is_idempotent_through_a_round_trip() -> None:
    payload = {"z": [1, {"b": None, "a": True}], "a": 0.5}
    once = canonical_json(payload)
    assert canonical_json(json.loads(once)) == once


def test_sha256_hex_matches_known_vector() -> None:
    assert sha256_hex("abc") == ("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


def test_sha256_accepts_bytes_and_str_equivalently() -> None:
    assert sha256_hex("abc") == sha256_hex(b"abc")


def test_short_id_length() -> None:
    assert len(short_id("anything")) == ID_LENGTH == 16


def test_file_sha256_matches_string_hash(tmp_path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"quantlab")
    assert file_sha256(path) == sha256_hex(b"quantlab")


def test_dataset_id_is_order_independent_but_content_sensitive() -> None:
    files_a = [("year=2018/bars.parquet", "bb"), ("year=2017/bars.parquet", "aa")]
    files_b = [("year=2017/bars.parquet", "aa"), ("year=2018/bars.parquet", "bb")]
    assert dataset_id(files_a, "BTC/USDT", "1h") == dataset_id(files_b, "BTC/USDT", "1h")

    changed = [("year=2017/bars.parquet", "aa"), ("year=2018/bars.parquet", "cc")]
    assert dataset_id(files_b, "BTC/USDT", "1h") != dataset_id(changed, "BTC/USDT", "1h")


def test_dataset_id_depends_on_symbol_and_timeframe() -> None:
    files = [("year=2017/bars.parquet", "aa")]
    assert dataset_id(files, "BTC/USDT", "1h") != dataset_id(files, "ETH/USDT", "1h")
    assert dataset_id(files, "BTC/USDT", "1h") != dataset_id(files, "BTC/USDT", "1m")
    assert len(dataset_id(files, "BTC/USDT", "1h")) == ID_LENGTH


def test_split_id_changes_with_policy_or_dataset() -> None:
    base = split_id("train: 2017", "ds0")
    assert base != split_id("train: 2018", "ds0")
    assert base != split_id("train: 2017", "ds1")
    assert len(base) == ID_LENGTH


def test_strategy_id_is_the_hash_of_the_code() -> None:
    code = "class S: ...\n"
    assert strategy_id(code) == sha256_hex(code)[:ID_LENGTH]
    assert strategy_id(code) == strategy_id(code.encode())


def _run_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "strategy": "s0",
        "params": {"fast": 10, "slow": 50},
        "dataset": "d0",
        "split": "p0",
        "segment": "train",
        "backtest_config_hash": "c0",
        "seed": 42,
        "engine_name": "simple_bar",
        "engine_version": "1",
    }
    kwargs.update(overrides)
    return kwargs


def test_run_id_is_deterministic_and_param_order_independent() -> None:
    first = run_id(**_run_kwargs())  # type: ignore[arg-type]
    second = run_id(**_run_kwargs(params={"slow": 50, "fast": 10}))  # type: ignore[arg-type]
    assert first == second == run_id(**_run_kwargs())  # type: ignore[arg-type]
    assert len(first) == ID_LENGTH


def test_run_id_changes_with_every_input() -> None:
    baseline = run_id(**_run_kwargs())  # type: ignore[arg-type]
    variations = [
        _run_kwargs(strategy="s1"),
        _run_kwargs(params={"fast": 11, "slow": 50}),
        _run_kwargs(dataset="d1"),
        _run_kwargs(split="p1"),
        _run_kwargs(segment="val"),
        _run_kwargs(backtest_config_hash="c1"),
        _run_kwargs(seed=43),
        _run_kwargs(engine_name="other"),
        _run_kwargs(engine_version="2"),
    ]
    ids = {run_id(**kwargs) for kwargs in variations}  # type: ignore[arg-type]
    assert baseline not in ids
    assert len(ids) == len(variations), "each input must change the run id"
