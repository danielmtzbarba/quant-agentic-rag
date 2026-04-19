from __future__ import annotations

from stock_agent_rag.cli import DEFAULT_SHARED_SUPABASE_BUNDLE_DIR, build_parser


def test_push_shared_supabase_parser_defaults_output_dir() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "push-shared-supabase",
            "--core-repo",
            "/tmp/core-repo",
            "--db-url",
            "postgresql://example",
        ]
    )

    assert args.command == "push-shared-supabase"
    assert args.output_dir == str(DEFAULT_SHARED_SUPABASE_BUNDLE_DIR)
    assert args.db_url == "postgresql://example"
