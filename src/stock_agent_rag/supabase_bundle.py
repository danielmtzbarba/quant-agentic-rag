from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class MigrationSource:
    name: str
    repo_root: Path
    config_path: Path
    migrations_dir: Path


@dataclass(frozen=True)
class BundledMigration:
    filename: str
    source_name: str
    source_path: Path


@dataclass(frozen=True)
class SupabasePushRequest:
    bundle_dir: Path
    project_ref: str | None = None
    db_url: str | None = None
    password: str | None = None
    skip_pooler: bool = False
    dry_run: bool = False
    include_all: bool = False


def build_supabase_bundle(
    *,
    core_repo: Path,
    rag_repo: Path,
    output_dir: Path,
) -> list[BundledMigration]:
    sources = [
        _resolve_source(name="core", repo_root=core_repo),
        _resolve_source(name="rag", repo_root=rag_repo),
    ]
    _prepare_output_dir(output_dir)

    bundled: list[BundledMigration] = []
    seen_filenames: dict[str, BundledMigration] = {}
    output_supabase_dir = output_dir / "supabase"
    output_migrations_dir = output_supabase_dir / "migrations"
    output_migrations_dir.mkdir(parents=True, exist_ok=True)

    _copy_config(primary_source=sources[0], output_supabase_dir=output_supabase_dir)

    for source in sources:
        for path in sorted(source.migrations_dir.glob("*.sql")):
            bundled_item = BundledMigration(
                filename=path.name,
                source_name=source.name,
                source_path=path,
            )
            existing = seen_filenames.get(path.name)
            if existing is not None:
                existing_text = existing.source_path.read_text(encoding="utf-8")
                current_text = path.read_text(encoding="utf-8")
                if existing_text == current_text:
                    logger.warning(
                        "skipping duplicate migration with identical contents",
                        extra={"filename": path.name, "source": source.name},
                    )
                    continue
                raise ValueError(
                    f"duplicate migration filename detected: {path.name} "
                    f"from {existing.source_name} and {source.name}"
                )
            shutil.copy2(path, output_migrations_dir / path.name)
            seen_filenames[path.name] = bundled_item
            bundled.append(bundled_item)

    bundled.sort(key=lambda item: item.filename)
    _write_manifest(output_dir=output_dir, bundled=bundled, sources=sources)
    return bundled


def _resolve_source(*, name: str, repo_root: Path) -> MigrationSource:
    resolved_root = repo_root.expanduser().resolve()
    supabase_dir = resolved_root / "supabase"
    config_path = supabase_dir / "config.toml"
    migrations_dir = supabase_dir / "migrations"
    if not config_path.exists():
        raise FileNotFoundError(f"missing Supabase config: {config_path}")
    if not migrations_dir.exists():
        raise FileNotFoundError(f"missing Supabase migrations directory: {migrations_dir}")
    return MigrationSource(
        name=name,
        repo_root=resolved_root,
        config_path=config_path,
        migrations_dir=migrations_dir,
    )


def _prepare_output_dir(output_dir: Path) -> None:
    resolved = output_dir.expanduser().resolve()
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def _copy_config(*, primary_source: MigrationSource, output_supabase_dir: Path) -> None:
    output_supabase_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(primary_source.config_path, output_supabase_dir / "config.toml")


def _write_manifest(
    *,
    output_dir: Path,
    bundled: list[BundledMigration],
    sources: list[MigrationSource],
) -> None:
    manifest = {
        "sources": [
            {
                "name": source.name,
                "repo_root": str(source.repo_root),
                "config_path": str(source.config_path),
                "migrations_dir": str(source.migrations_dir),
            }
            for source in sources
        ],
        "migrations": [
            {
                "filename": item.filename,
                "source_name": item.source_name,
                "source_path": str(item.source_path),
            }
            for item in bundled
        ],
    }
    (output_dir / "bundle-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def run_supabase_push(request: SupabasePushRequest) -> None:
    bundle_dir = request.bundle_dir.expanduser().resolve()
    if request.db_url:
        _run_supabase_command(
            ["supabase", "db", "push", "--db-url", request.db_url, *_push_flags(request)],
            cwd=bundle_dir,
            password=request.password,
        )
        return

    if not request.project_ref:
        raise ValueError("project_ref or db_url is required to push migrations.")

    link_command = ["supabase", "link", "--project-ref", request.project_ref]
    if request.skip_pooler:
        link_command.append("--skip-pooler")
    _run_supabase_command(link_command, cwd=bundle_dir, password=request.password)
    _run_supabase_command(
        ["supabase", "db", "push", *_push_flags(request)],
        cwd=bundle_dir,
        password=request.password,
    )


def _push_flags(request: SupabasePushRequest) -> list[str]:
    flags: list[str] = []
    if request.dry_run:
        flags.append("--dry-run")
    if request.include_all:
        flags.append("--include-all")
    return flags


def _run_supabase_command(
    command: list[str],
    *,
    cwd: Path,
    password: str | None,
) -> None:
    env = os.environ.copy()
    if password:
        env["SUPABASE_DB_PASSWORD"] = password
    subprocess.run(command, cwd=str(cwd), env=env, check=True)
