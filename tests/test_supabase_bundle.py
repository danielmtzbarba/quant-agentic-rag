from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import stock_agent_rag.supabase_bundle as bundle_module
from stock_agent_rag.supabase_bundle import (
    SupabasePushRequest,
    build_supabase_bundle,
    run_supabase_push,
)


def _make_repo(root: Path, *, project_id: str, migrations: dict[str, str]) -> Path:
    supabase_dir = root / "supabase"
    migrations_dir = supabase_dir / "migrations"
    migrations_dir.mkdir(parents=True, exist_ok=True)
    (supabase_dir / "config.toml").write_text(
        f'project_id = "{project_id}"\n',
        encoding="utf-8",
    )
    for filename, content in migrations.items():
        (migrations_dir / filename).write_text(content, encoding="utf-8")
    return root


def test_build_supabase_bundle_merges_core_and_rag_migrations(tmp_path: Path) -> None:
    core_repo = _make_repo(
        tmp_path / "core-repo",
        project_id="core-project",
        migrations={"20260417000000_init_core_schema.sql": "create schema if not exists core;"},
    )
    rag_repo = _make_repo(
        tmp_path / "rag-repo",
        project_id="rag-project",
        migrations={"20260417070000_init_rag_schema.sql": "create schema if not exists rag;"},
    )
    output_dir = tmp_path / "bundle"

    bundled = build_supabase_bundle(
        core_repo=core_repo,
        rag_repo=rag_repo,
        output_dir=output_dir,
    )

    assert [item.filename for item in bundled] == [
        "20260417000000_init_core_schema.sql",
        "20260417070000_init_rag_schema.sql",
    ]
    assert (output_dir / "supabase" / "config.toml").read_text(encoding="utf-8").strip() == (
        'project_id = "core-project"'
    )
    assert (
        output_dir / "supabase" / "migrations" / "20260417070000_init_rag_schema.sql"
    ).exists()

    manifest = json.loads((output_dir / "bundle-manifest.json").read_text(encoding="utf-8"))
    assert {item["source_name"] for item in manifest["migrations"]} == {"core", "rag"}


def test_build_supabase_bundle_rejects_conflicting_duplicate_filenames(tmp_path: Path) -> None:
    filename = "20260417000000_shared.sql"
    core_repo = _make_repo(
        tmp_path / "core-repo",
        project_id="core-project",
        migrations={filename: "create schema if not exists core;"},
    )
    rag_repo = _make_repo(
        tmp_path / "rag-repo",
        project_id="rag-project",
        migrations={filename: "create schema if not exists rag;"},
    )

    with pytest.raises(ValueError, match="duplicate migration filename"):
        build_supabase_bundle(
            core_repo=core_repo,
            rag_repo=rag_repo,
            output_dir=tmp_path / "bundle",
        )


def test_run_supabase_push_links_then_pushes_for_project_ref(tmp_path: Path) -> None:
    calls: list[tuple[list[str], str, str | None]] = []

    def fake_run(command, *, cwd, env, check):
        assert check is True
        calls.append((list(command), cwd, env.get("SUPABASE_DB_PASSWORD")))
        return subprocess.CompletedProcess(command, 0)

    original = bundle_module.subprocess.run
    bundle_module.subprocess.run = fake_run
    try:
        run_supabase_push(
            SupabasePushRequest(
                bundle_dir=tmp_path,
                project_ref="proj-ref-123",
                password="secret",
                skip_pooler=True,
                dry_run=True,
                include_all=True,
            )
        )
    finally:
        bundle_module.subprocess.run = original

    assert calls == [
        (
            ["supabase", "link", "--project-ref", "proj-ref-123", "--skip-pooler"],
            str(tmp_path.resolve()),
            "secret",
        ),
        (
            ["supabase", "db", "push", "--dry-run", "--include-all"],
            str(tmp_path.resolve()),
            "secret",
        ),
    ]


def test_run_supabase_push_uses_db_url_without_link(tmp_path: Path) -> None:
    calls: list[tuple[list[str], str]] = []

    def fake_run(command, *, cwd, env, check):
        assert check is True
        assert "SUPABASE_DB_PASSWORD" not in env
        calls.append((list(command), cwd))
        return subprocess.CompletedProcess(command, 0)

    original = bundle_module.subprocess.run
    bundle_module.subprocess.run = fake_run
    try:
        run_supabase_push(
            SupabasePushRequest(
                bundle_dir=tmp_path,
                db_url="postgresql://user:pass@host:5432/db",
                dry_run=True,
            )
        )
    finally:
        bundle_module.subprocess.run = original

    assert calls == [
        (
            [
                "supabase",
                "db",
                "push",
                "--db-url",
                "postgresql://user:pass@host:5432/db",
                "--dry-run",
            ],
            str(tmp_path.resolve()),
        )
    ]


def test_run_supabase_push_requires_project_ref_or_db_url(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="project_ref or db_url is required"):
        run_supabase_push(SupabasePushRequest(bundle_dir=tmp_path))
