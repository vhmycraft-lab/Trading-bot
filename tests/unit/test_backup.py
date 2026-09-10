"""Backup and restore (spec T43).

What a backup of this platform has to preserve is not "the files" — it is the
ability to reproduce a number. So the tests here are mostly about the two ways
that fails silently: something needed is missing from the archive, or something
in it changed on the way and nothing said so.

The third concern is the one that would not show up as a failure at all: a
secret riding along inside an archive. An archive travels — to another disk,
another machine, a bucket — and a credential in one is a credential everywhere
it has ever been. That exclusion is asserted from both sides, because a list
that quietly shrinks looks exactly like a list that was always short.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from quantlab.cli.backup import (
    BACKUP_VERSION,
    EXCLUDED,
    INCLUDED,
    MANIFEST_NAME,
    create_backup,
    read_manifest,
    restore_backup,
    verify_restore,
)
from quantlab.core.errors import ConfigError, QuantLabError


def a_tree(root: Path) -> Path:
    """A miniature of the real layout: database, dataset, artifact, config."""
    (root / "data" / "binance" / "BTCUSDT" / "1h").mkdir(parents=True)
    (root / "artifacts" / "run0").mkdir(parents=True)
    (root / "configs").mkdir()

    (root / "quantlab.db").write_bytes(b"SQLite format 3\x00" + b"rows" * 64)
    (root / "data" / "binance" / "BTCUSDT" / "1h" / "bars.parquet").write_bytes(b"PAR1data")
    (root / "data" / "binance" / "BTCUSDT" / "1h" / "manifest.json").write_text('{"files": []}')
    (root / "artifacts" / "run0" / "equity.parquet").write_bytes(b"PAR1equity")
    (root / "configs" / "default.yaml").write_text("project:\n  db_path: ./quantlab.db\n")
    return root


# ---------------------------------------------------------------------------
# what goes in
# ---------------------------------------------------------------------------
def test_a_backup_holds_everything_a_reproduction_needs(tmp_path: Path) -> None:
    """Section 22's criterion is "fresh clone + restore => reproduce passes", so
    the database alone is not enough: a run cites a dataset, and without the
    bars it can be read back but not re-executed."""
    root = a_tree(tmp_path / "src")
    manifest = create_backup(root, tmp_path / "b.tar.gz")

    names = set(manifest.files)
    assert "quantlab.db" in names
    assert "data/binance/BTCUSDT/1h/bars.parquet" in names
    assert "data/binance/BTCUSDT/1h/manifest.json" in names
    assert "artifacts/run0/equity.parquet" in names
    assert "configs/default.yaml" in names


def test_the_included_list_is_what_the_archive_actually_contains(tmp_path: Path) -> None:
    """Guards the test above against a list that documents more than it does."""
    root = a_tree(tmp_path / "src")
    manifest = create_backup(root, tmp_path / "b.tar.gz")
    assert set(manifest.included) <= set(INCLUDED)
    for name in manifest.included:
        assert any(f == name or f.startswith(f"{name}/") for f in manifest.files), name


def test_a_missing_directory_is_not_an_error(tmp_path: Path) -> None:
    """A tree that has never run has no `artifacts/`. Refusing to back that up
    would make the first backup the hardest one to take."""
    root = tmp_path / "src"
    root.mkdir()
    (root / "quantlab.db").write_bytes(b"x" * 32)
    manifest = create_backup(root, tmp_path / "b.tar.gz")
    assert manifest.included == ["quantlab.db"]


# ---------------------------------------------------------------------------
# what stays out
# ---------------------------------------------------------------------------
def test_a_dotenv_is_never_archived(tmp_path: Path) -> None:
    """The exclusion that matters. An archive travels, and a credential inside
    one is a credential in every place the archive has ever been."""
    root = a_tree(tmp_path / "src")
    (root / ".env").write_text("GLM_API_KEY=" + "x" * 40 + "\n")

    manifest = create_backup(root, tmp_path / "b.tar.gz")
    assert not any(".env" in name for name in manifest.files)

    with tarfile.open(tmp_path / "b.tar.gz") as archive:
        assert not any(".env" in member.name for member in archive.getmembers())


def test_a_dotenv_nested_anywhere_is_still_excluded(tmp_path: Path) -> None:
    """Matched per path component, not as a prefix: a `.env` that ended up under
    `configs/` is the same secret in a less obvious place."""
    root = a_tree(tmp_path / "src")
    (root / "configs" / ".env").write_text("GLM_API_KEY=" + "y" * 40 + "\n")
    manifest = create_backup(root, tmp_path / "b.tar.gz")
    assert not any(".env" in name for name in manifest.files)


def test_the_exclusions_are_recorded_with_a_reason(tmp_path: Path) -> None:
    """A restore that finds no `.env` should be able to tell "left out on
    purpose" from "lost". The manifest says which."""
    root = a_tree(tmp_path / "src")
    manifest = create_backup(root, tmp_path / "b.tar.gz")
    assert ".env" in manifest.excluded
    assert "Keychain" in manifest.excluded[".env"]
    assert all(len(reason) > 20 for reason in manifest.excluded.values())


def test_the_exclusion_list_still_covers_the_dangerous_names() -> None:
    """Pinned so the list cannot quietly shrink. A shorter list looks identical
    to a list that was always short, and the difference is a leaked key."""
    assert ".env" in EXCLUDED
    assert ".git" in EXCLUDED
    assert ".venv" in EXCLUDED


# ---------------------------------------------------------------------------
# round trip
# ---------------------------------------------------------------------------
def test_a_restore_reproduces_every_byte(tmp_path: Path) -> None:
    root = a_tree(tmp_path / "src")
    archive = tmp_path / "b.tar.gz"
    manifest = create_backup(root, archive)

    target = tmp_path / "restored"
    target.mkdir()
    restore_backup(archive, target)

    assert verify_restore(manifest, target) == []
    assert (target / "quantlab.db").read_bytes() == (root / "quantlab.db").read_bytes()


def test_a_tampered_file_is_reported_after_restore(tmp_path: Path) -> None:
    """The failure this command exists to make impossible to have silently: a
    tree that looks complete and reproduces different numbers."""
    root = a_tree(tmp_path / "src")
    archive = tmp_path / "b.tar.gz"
    manifest = create_backup(root, archive)

    target = tmp_path / "restored"
    target.mkdir()
    restore_backup(archive, target)
    (target / "quantlab.db").write_bytes(b"SQLite format 3\x00" + b"different")

    problems = verify_restore(manifest, target)
    assert any("quantlab.db" in problem for problem in problems), problems


def test_a_file_deleted_after_restore_is_reported(tmp_path: Path) -> None:
    root = a_tree(tmp_path / "src")
    archive = tmp_path / "b.tar.gz"
    manifest = create_backup(root, archive)
    target = tmp_path / "restored"
    target.mkdir()
    restore_backup(archive, target)
    (target / "configs" / "default.yaml").unlink()
    assert any("missing" in p for p in verify_restore(manifest, target))


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------
def test_a_restore_refuses_to_overwrite_by_default(tmp_path: Path) -> None:
    """A backup is reached for at exactly the moment the live tree is confusing.
    Silently replacing `quantlab.db` would destroy the evidence somebody was in
    the middle of reading."""
    root = a_tree(tmp_path / "src")
    archive = tmp_path / "b.tar.gz"
    create_backup(root, archive)

    with pytest.raises(ConfigError, match="already exist"):
        restore_backup(archive, root)


def test_force_allows_the_overwrite(tmp_path: Path) -> None:
    """The refusal has to be escapable, or it gets worked around with `rm -rf`."""
    root = a_tree(tmp_path / "src")
    archive = tmp_path / "b.tar.gz"
    create_backup(root, archive)
    restore_backup(archive, root, force=True)


def test_backing_up_over_an_existing_archive_is_refused(tmp_path: Path) -> None:
    """Backups are cheap; a clobbered one is not recoverable."""
    root = a_tree(tmp_path / "src")
    archive = tmp_path / "b.tar.gz"
    create_backup(root, archive)
    with pytest.raises(ConfigError, match="refusing to overwrite"):
        create_backup(root, archive)


def test_an_archive_with_no_manifest_is_refused(tmp_path: Path) -> None:
    """A tarball from somewhere else cannot be checked, and restoring something
    uncheckable is how an unverifiable tree gets built."""
    archive = tmp_path / "stranger.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        payload = tmp_path / "x.txt"
        payload.write_text("hello")
        handle.add(payload, arcname="x.txt")

    with pytest.raises(QuantLabError, match="no BACKUP_MANIFEST"):
        read_manifest(archive)


def test_an_archive_from_a_future_format_is_refused(tmp_path: Path) -> None:
    """Refusing beats guessing, as with the paper state file."""
    root = a_tree(tmp_path / "src")
    archive = tmp_path / "b.tar.gz"
    create_backup(root, archive)

    bumped = tmp_path / "bumped.tar.gz"
    with tarfile.open(archive) as src, tarfile.open(bumped, "w:gz") as dst:
        for member in src.getmembers():
            handle = src.extractfile(member)
            if member.name == MANIFEST_NAME and handle is not None:
                payload = json.loads(handle.read())
                payload["version"] = BACKUP_VERSION + 1
                data = json.dumps(payload).encode()
                member.size = len(data)
                import io

                dst.addfile(member, io.BytesIO(data))
            elif handle is not None:
                dst.addfile(member, handle)

    with pytest.raises(QuantLabError, match="refusing to restore"):
        read_manifest(bumped)


def test_a_path_that_escapes_the_target_is_refused(tmp_path: Path) -> None:
    """An archive is usually one's own, but "usually" is not a security property
    and a tarball can arrive from anywhere."""
    root = a_tree(tmp_path / "src")
    archive = tmp_path / "b.tar.gz"
    create_backup(root, archive)

    evil = tmp_path / "evil.tar.gz"
    with tarfile.open(archive) as src, tarfile.open(evil, "w:gz") as dst:
        for member in src.getmembers():
            handle = src.extractfile(member)
            if handle is None:
                continue
            if member.name == "quantlab.db":
                member.name = "../escaped.db"
            dst.addfile(member, handle)

    target = tmp_path / "restored"
    target.mkdir()
    with pytest.raises(QuantLabError, match="escapes the restore"):
        restore_backup(evil, target)
    assert not (tmp_path / "escaped.db").exists()
