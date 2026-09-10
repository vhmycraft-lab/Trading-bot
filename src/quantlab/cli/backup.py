"""``quantlab backup`` / ``quantlab restore`` (master spec T43).

What a backup of this platform has to preserve is not "the files" — it is the
ability to *reproduce a number*. Section 22's acceptance criterion says so
directly: a fresh clone plus a restore must let ``quantlab report reproduce``
re-run a golden run and get the same metrics.

That fixes what goes in, and it is more than the database:

* ``quantlab.db`` — every run, verdict, candidate and lineage row. The
  append-only record of what was searched and what was decided.
* ``data/`` — the Parquet datasets **and their manifests**. A run cites a
  ``dataset_id``; without the bars behind it the run cannot be re-executed, only
  read back.
* ``artifacts/`` — equity curves, trade ledgers, tearsheets, prompt and response
  files. Not needed to *re-run*, needed to compare against.
* ``configs/`` — a run's ``config_hash`` is computed from the resolved
  configuration, so a restore under different defaults reproduces a different
  run identity.

What is deliberately **excluded** is anything that would restore a secret.
``.env`` is not backed up and never will be: a backup archive travels — to
another disk, another machine, a cloud bucket — and a credential inside one is a
credential in every place the archive has ever been. Secrets live in the
Keychain and are re-entered on the new machine. The manifest records that the
exclusion happened, so a restore that finds no ``.env`` knows it was by design
rather than by loss.

Restore refuses to overwrite by default. A backup is most often reached for at
exactly the moment when the live tree is confusing, and a restore that silently
replaced ``quantlab.db`` would destroy the evidence someone was in the middle of
reading.
"""

from __future__ import annotations

import io
import json
import tarfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Final

import typer
from rich.console import Console

from quantlab.core.errors import ConfigError, QuantLabError
from quantlab.core.hashing import file_sha256

__all__ = ["EXCLUDED", "MANIFEST_NAME", "BackupManifest", "app", "create_backup", "restore_backup"]

app = typer.Typer(help="Back up and restore everything needed to reproduce a run.")
console = Console()

#: Name of the manifest inside the archive.
MANIFEST_NAME: Final[str] = "BACKUP_MANIFEST.json"

#: Format version. A restore refuses an archive it may misread rather than
#: guessing, for the same reason the paper state file does.
BACKUP_VERSION: Final[int] = 1

#: What a backup contains, in the order it is written.
INCLUDED: Final[tuple[str, ...]] = ("quantlab.db", "data", "artifacts", "configs")

#: What a backup never contains, and why. Checked by a test, because the cost of
#: this list quietly shrinking is a credential in an archive somebody emails.
EXCLUDED: Final[dict[str, str]] = {
    ".env": "secrets live in the Keychain and are re-entered, never shipped in an archive",
    ".venv": "rebuilt by `make sync` from uv.lock, and machine-specific",
    "__pycache__": "derived, and stale copies confuse an editable install",
    ".git": "the repository is restored by cloning it, not by unpacking a tarball",
}


@dataclass(frozen=True)
class BackupManifest:
    """What the archive holds, and what it deliberately does not."""

    version: int
    created_at: str
    quantlab_version: str
    included: list[str]
    excluded: dict[str, str]
    #: ``path -> sha256`` for every regular file in the archive. This is what
    #: makes a restore checkable rather than hopeful.
    files: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _is_excluded(relative: Path) -> bool:
    return any(part in EXCLUDED for part in relative.parts)


def _files_under(root: Path, source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        return []
    return [
        path
        for path in sorted(source.rglob("*"))
        if path.is_file() and not _is_excluded(path.relative_to(root))
    ]


def create_backup(root: Path, destination: Path, *, version: str = "unknown") -> BackupManifest:
    """Write a backup of ``root`` to ``destination``.

    Every file is hashed as it goes in. The hashes are what let ``restore``
    report a truncated or edited archive instead of unpacking it and leaving
    somebody to discover the problem later, in the middle of a comparison they
    thought was meaningful.
    """
    root = root.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise ConfigError(
            f"{destination} already exists; refusing to overwrite a backup. "
            "Backups are cheap and a clobbered one is not recoverable — "
            "choose another name.",
            path=str(destination),
        )

    hashes: dict[str, str] = {}
    present: list[str] = []
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(destination, "w:gz") as archive:
        for name in INCLUDED:
            source = root / name
            if not source.exists():
                continue
            present.append(name)
            for path in _files_under(root, source):
                relative = path.relative_to(root)
                hashes[str(relative)] = file_sha256(path)
                archive.add(path, arcname=str(relative))

        manifest = BackupManifest(
            version=BACKUP_VERSION,
            created_at=datetime.now(UTC).isoformat(),
            quantlab_version=version,
            included=present,
            excluded=dict(EXCLUDED),
            files=hashes,
        )
        payload = manifest.to_json().encode("utf-8")
        info = tarfile.TarInfo(MANIFEST_NAME)
        info.size = len(payload)
        info.mtime = 0
        archive.addfile(info, io.BytesIO(payload))

    return manifest


def read_manifest(archive_path: Path) -> BackupManifest:
    """The manifest inside ``archive_path``."""
    with tarfile.open(archive_path, "r:gz") as archive:
        try:
            member = archive.getmember(MANIFEST_NAME)
        except KeyError as exc:
            raise QuantLabError(
                f"{archive_path} has no {MANIFEST_NAME}; it was not written by "
                "`quantlab backup` and its contents cannot be checked",
                path=str(archive_path),
            ) from exc
        handle = archive.extractfile(member)
        if handle is None:  # pragma: no cover - a regular file always extracts
            raise QuantLabError("the backup manifest could not be read")
        payload = json.loads(handle.read().decode("utf-8"))

    if int(payload.get("version", 0)) != BACKUP_VERSION:
        raise QuantLabError(
            f"{archive_path} is backup format v{payload.get('version')}, this build "
            f"writes v{BACKUP_VERSION}; refusing to restore from a layout it may misread",
            path=str(archive_path),
        )
    return BackupManifest(**payload)


def restore_backup(archive_path: Path, target: Path, *, force: bool = False) -> list[str]:
    """Unpack ``archive_path`` into ``target``; return the files restored.

    Refuses to overwrite unless ``force``. A backup is most often reached for at
    the moment the live tree is confusing, and silently replacing ``quantlab.db``
    would destroy the evidence somebody was reading.
    """
    manifest = read_manifest(archive_path)
    target = target.resolve()

    clashes = [name for name in manifest.files if (target / name).exists()]
    if clashes and not force:
        raise ConfigError(
            f"{len(clashes)} file(s) already exist under {target}, including "
            f"{clashes[0]}. Restoring would overwrite them. Move them aside, "
            "restore into an empty directory, or pass --force if you are certain.",
            path=str(target),
            n_clashes=len(clashes),
        )

    with tarfile.open(archive_path, "r:gz") as archive:
        members = [m for m in archive.getmembers() if m.name != MANIFEST_NAME]
        for member in members:
            # Refuse anything that would escape the target directory. A backup
            # is usually one's own, but "usually" is not a security property and
            # an archive can arrive from anywhere.
            resolved = (target / member.name).resolve()
            if not resolved.is_relative_to(target):
                raise QuantLabError(
                    f"{archive_path} contains a path that escapes the restore "
                    f"directory ({member.name}); refusing to unpack it",
                    member=member.name,
                )
        # `filter="data"` is Python 3.12+'s hardened extraction: it refuses
        # absolute paths, `..` traversal, symlinks, links out of the tree, and
        # device files, and drops ownership and permission bits. The explicit
        # loop above stays because it produces a *named* refusal a person can
        # act on, where the filter raises something generic — but the filter is
        # the wider net and covers shapes the loop does not check for.
        archive.extractall(target, members=members, filter="data")

    return sorted(manifest.files)


def verify_restore(manifest: BackupManifest, target: Path) -> list[str]:
    """Every file whose hash no longer matches the manifest.

    Run after restoring. An archive that truncated in transit, or a file edited
    after unpacking, produces a tree that looks complete and reproduces
    different numbers — which is the failure this whole command exists to make
    impossible to have silently.
    """
    problems: list[str] = []
    for name, expected in sorted(manifest.files.items()):
        path = target / name
        if not path.is_file():
            problems.append(f"{name}: missing after restore")
        elif file_sha256(path) != expected:
            problems.append(f"{name}: hash does not match the manifest")
    return problems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@app.command("create")
def create(
    ctx: typer.Context,
    destination: Annotated[Path, typer.Argument(help="Archive to write (.tar.gz).")],
) -> None:
    """Back up the database, datasets, artifacts and configuration."""
    from quantlab import __version__

    state = ctx.obj
    root = Path.cwd()
    manifest = create_backup(root, destination, version=__version__)
    size_mb = destination.stat().st_size / 1e6
    console.print(f"[green]wrote[/] {destination} ({size_mb:.1f} MB)")
    console.print(f"  {len(manifest.files)} file(s) from {', '.join(manifest.included)}")
    console.print(f"  [dim]excluded: {', '.join(sorted(manifest.excluded))}[/dim]")
    del state


@app.command("restore")
def restore(
    ctx: typer.Context,
    archive: Annotated[Path, typer.Argument(help="Archive to restore.")],
    target: Annotated[Path, typer.Option("--into", help="Where to restore.")] = Path(),
    force: Annotated[bool, typer.Option("--force", help="Overwrite existing files.")] = False,
) -> None:
    """Restore a backup, then verify every file against its recorded hash."""
    manifest = read_manifest(archive)
    restored = restore_backup(archive, target, force=force)
    problems = verify_restore(manifest, target.resolve())

    console.print(f"[green]restored[/] {len(restored)} file(s) into {target}")
    if problems:
        console.print(f"[red]{len(problems)} file(s) do not match the manifest:[/]")
        for problem in problems[:10]:
            console.print(f"  {problem}")
        raise typer.Exit(1)
    console.print("  [green]every file matches its recorded hash[/]")
    console.print(
        "  [dim]secrets were not in the archive; set them up again with "
        "`quantlab doctor` to check[/dim]"
    )
    del ctx


@app.command("verify")
def verify(
    ctx: typer.Context,
    archive: Annotated[Path, typer.Argument(help="Archive to inspect.")],
) -> None:
    """Read an archive's manifest without unpacking it."""
    manifest = read_manifest(archive)
    console.print(f"backup v{manifest.version} written {manifest.created_at}")
    console.print(f"  quantlab {manifest.quantlab_version}")
    console.print(f"  {len(manifest.files)} file(s): {', '.join(manifest.included)}")
    for name, why in sorted(manifest.excluded.items()):
        console.print(f"  [dim]excluded {name} — {why}[/dim]")
    del ctx
