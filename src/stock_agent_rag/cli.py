from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import uvicorn

from .api import create_app
from .config import get_settings
from .db import initialize_database
from .evaluation import DEFAULT_GOLDEN_SET_PATH, evaluate_release_gates, load_golden_set
from .indexing import ChunkIndexingService
from .ingestion import (
    AlphaVantageNewsIngestionService,
    AlphaVantageTranscriptIngestionService,
    SecFilingIngestionService,
)
from .logging import setup_logging
from .schemas import ResearchRequest
from .service import get_research_service
from .supabase_bundle import (
    SupabasePushRequest,
    build_supabase_bundle,
    run_supabase_push,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHARED_SUPABASE_BUNDLE_DIR = Path("/tmp/quant-supabase-bundle")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stock agent RAG service")
    subcommands = parser.add_subparsers(dest="command", required=True)

    serve_parser = subcommands.add_parser("serve", help="Run the FastAPI service.")
    serve_parser.add_argument("--host", help="Override host from settings.")
    serve_parser.add_argument("--port", type=int, help="Override port from settings.")

    research_parser = subcommands.add_parser("research", help="Run a one-off research workflow.")
    research_parser.add_argument("--ticker", required=True, help="Ticker symbol to analyze.")
    research_parser.add_argument(
        "--question",
        default="Generate an evidence-backed investment thesis.",
        help="Research question for the workflow.",
    )

    release_gates_parser = subcommands.add_parser(
        "release-gates",
        help="Evaluate golden-set run outputs against release-gate thresholds.",
    )
    release_gates_parser.add_argument(
        "--results",
        required=True,
        help="JSON file containing a list of workflow results or an object with `results`.",
    )
    release_gates_parser.add_argument(
        "--golden-set",
        default=str(DEFAULT_GOLDEN_SET_PATH),
        help="Path to the golden-set manifest JSON.",
    )
    release_gates_parser.add_argument(
        "--retrieval-k",
        type=int,
        default=5,
        help="Top-k cutoff used for retrieval precision@k and recall@k.",
    )

    sec_parser = subcommands.add_parser("ingest-sec", help="Download and normalize SEC filings.")
    sec_parser.add_argument("--ticker", required=True, help="Ticker symbol to ingest.")
    sec_parser.add_argument(
        "--form-type",
        default="10-K",
        help="SEC form type such as 10-K or 10-Q.",
    )
    sec_parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="How many recent filings to ingest.",
    )

    transcript_parser = subcommands.add_parser(
        "ingest-transcript",
        help="Download and normalize an Alpha Vantage earnings transcript.",
    )
    transcript_parser.add_argument("--ticker", required=True, help="Ticker symbol to ingest.")
    transcript_parser.add_argument("--year", required=True, type=int, help="Transcript year.")
    transcript_parser.add_argument(
        "--quarter",
        required=True,
        type=int,
        choices=[1, 2, 3, 4],
        help="Fiscal quarter for the transcript.",
    )
    transcript_parser.add_argument(
        "--force",
        action="store_true",
        help="Force redownload even if a cached raw payload exists.",
    )

    news_parser = subcommands.add_parser(
        "ingest-news",
        help="Download and normalize an Alpha Vantage news batch.",
    )
    news_parser.add_argument("--ticker", required=True, help="Ticker symbol to ingest.")
    news_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of recent news articles to ingest.",
    )
    news_parser.add_argument(
        "--force",
        action="store_true",
        help="Force redownload even if a cached raw payload exists.",
    )

    index_parser = subcommands.add_parser(
        "index-chunks",
        help="Embed registry chunks so semantic retrieval can search them.",
    )
    index_parser.add_argument("--ticker", help="Optional ticker filter.")
    index_parser.add_argument(
        "--limit",
        type=int,
        help="Optional cap on how many missing embeddings to generate.",
    )
    index_parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild embeddings even if an index record already exists.",
    )

    bundle_parser = subcommands.add_parser(
        "bundle-supabase",
        help="Assemble a merged Supabase workspace from the core and rag repositories.",
    )
    bundle_parser.add_argument(
        "--core-repo",
        required=True,
        help=(
            "Absolute or relative path to the core repository that owns "
            "the main Supabase project."
        ),
    )
    bundle_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the merged Supabase workspace should be generated.",
    )
    bundle_parser.add_argument(
        "--project-ref",
        help="Optional Supabase project ref. If provided with --push, the bundle is linked first.",
    )
    bundle_parser.add_argument(
        "--db-url",
        help="Optional direct database URL to use for supabase db push instead of linking.",
    )
    bundle_parser.add_argument(
        "--password",
        help="Optional database password passed through via SUPABASE_DB_PASSWORD.",
    )
    bundle_parser.add_argument(
        "--skip-pooler",
        action="store_true",
        help="Pass --skip-pooler when linking the generated Supabase workspace.",
    )
    bundle_parser.add_argument(
        "--push",
        action="store_true",
        help="After bundling, run supabase link/db push or db push --db-url automatically.",
    )
    bundle_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="When used with --push, print the migrations that would be applied.",
    )
    bundle_parser.add_argument(
        "--include-all",
        action="store_true",
        help="When used with --push, include all local migrations missing from remote history.",
    )

    push_shared_parser = subcommands.add_parser(
        "push-shared-supabase",
        help="Bundle shared Supabase migrations and push them in one step.",
    )
    push_shared_parser.add_argument(
        "--core-repo",
        required=True,
        help=(
            "Absolute or relative path to the core repository that owns "
            "the main Supabase project."
        ),
    )
    push_shared_parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_SHARED_SUPABASE_BUNDLE_DIR),
        help=(
            "Directory where the merged Supabase workspace should be generated. "
            f"Defaults to {DEFAULT_SHARED_SUPABASE_BUNDLE_DIR}."
        ),
    )
    push_shared_parser.add_argument(
        "--project-ref",
        help="Optional Supabase project ref. If provided, the bundle is linked first.",
    )
    push_shared_parser.add_argument(
        "--db-url",
        help="Optional direct database URL to use for supabase db push instead of linking.",
    )
    push_shared_parser.add_argument(
        "--password",
        help="Optional database password passed through via SUPABASE_DB_PASSWORD.",
    )
    push_shared_parser.add_argument(
        "--skip-pooler",
        action="store_true",
        help="Pass --skip-pooler when linking the generated Supabase workspace.",
    )
    push_shared_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the migrations that would be applied without applying them.",
    )
    push_shared_parser.add_argument(
        "--include-all",
        action="store_true",
        help="Include all local migrations missing from remote history.",
    )

    subcommands.add_parser("db-init", help="Create database tables for the registry layer.")
    return parser


def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level, settings.resolved_log_format)
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "serve":
        uvicorn.run(
            create_app,
            factory=True,
            host=args.host or settings.app_host,
            port=args.port or settings.app_port,
            log_config=None,
        )
        return

    if args.command == "research":
        service = get_research_service()
        result = service.run(ResearchRequest(ticker=args.ticker, question=args.question))
        print("\n=== PLAN ===\n")
        print(result.plan)
        print("\n=== REPORT ===\n")
        print(result.report)
        print("\n=== VERIFICATION ===\n")
        print(result.verification_summary)
        return

    if args.command == "release-gates":
        payload = json.loads(Path(args.results).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            results = payload.get("results", [])
        else:
            results = payload
        if not isinstance(results, list):
            parser.error("--results must contain a JSON list or an object with a `results` list.")
        evaluation = evaluate_release_gates(
            results=[item for item in results if isinstance(item, dict)],
            manifest=load_golden_set(args.golden_set),
            retrieval_k=args.retrieval_k,
        )
        print("\n=== RELEASE GATES ===\n")
        print(json.dumps(evaluation, indent=2, sort_keys=True))
        return

    if args.command == "ingest-sec":
        service = SecFilingIngestionService(settings=settings)
        summary = service.ingest(ticker=args.ticker, form_type=args.form_type, limit=args.limit)
        print("\n=== SEC INGESTION ===\n")
        print(json.dumps(asdict(summary), indent=2, sort_keys=True))
        return

    if args.command == "ingest-transcript":
        service = AlphaVantageTranscriptIngestionService(settings=settings)
        summary = service.ingest(
            ticker=args.ticker,
            year=args.year,
            quarter=args.quarter,
            force=args.force,
        )
        print("\n=== TRANSCRIPT INGESTION ===\n")
        print(json.dumps(asdict(summary), indent=2, sort_keys=True))
        return

    if args.command == "ingest-news":
        service = AlphaVantageNewsIngestionService(settings=settings)
        summary = service.ingest(
            ticker=args.ticker,
            limit=args.limit,
            force=args.force,
        )
        print("\n=== NEWS INGESTION ===\n")
        print(json.dumps(asdict(summary), indent=2, sort_keys=True))
        return

    if args.command == "db-init":
        initialize_database(settings)
        print("Database tables initialized.")
        return

    if args.command == "index-chunks":
        initialize_database(settings)
        from .db import get_db_session

        session = get_db_session()
        try:
            service = ChunkIndexingService(session, settings=settings)
            summary = service.index_chunks(
                ticker=args.ticker,
                limit=args.limit,
                force=args.force,
            )
        finally:
            session.close()
        print("\n=== CHUNK INDEXING ===\n")
        print(json.dumps(asdict(summary), indent=2, sort_keys=True))
        return

    if args.command == "bundle-supabase":
        if args.push and not (args.project_ref or args.db_url):
            parser.error(
                "bundle-supabase --push requires either --project-ref or --db-url."
            )
        _run_bundle_supabase(
            core_repo=Path(args.core_repo),
            output_dir=Path(args.output_dir),
            project_ref=args.project_ref,
            db_url=args.db_url,
            password=args.password,
            skip_pooler=args.skip_pooler,
            push=args.push,
            dry_run=args.dry_run,
            include_all=args.include_all,
        )
        return

    if args.command == "push-shared-supabase":
        if not (args.project_ref or args.db_url):
            parser.error(
                "push-shared-supabase requires either --project-ref or --db-url."
            )
        _run_bundle_supabase(
            core_repo=Path(args.core_repo),
            output_dir=Path(args.output_dir),
            project_ref=args.project_ref,
            db_url=args.db_url,
            password=args.password,
            skip_pooler=args.skip_pooler,
            push=True,
            dry_run=args.dry_run,
            include_all=args.include_all,
        )
        return


def _run_bundle_supabase(
    *,
    core_repo: Path,
    output_dir: Path,
    project_ref: str | None,
    db_url: str | None,
    password: str | None,
    skip_pooler: bool,
    push: bool,
    dry_run: bool,
    include_all: bool,
) -> None:
    bundled = build_supabase_bundle(
        core_repo=core_repo,
        rag_repo=PROJECT_ROOT,
        output_dir=output_dir,
    )
    bundle_path = output_dir.resolve()
    if push:
        run_supabase_push(
            SupabasePushRequest(
                bundle_dir=bundle_path,
                project_ref=project_ref,
                db_url=db_url,
                password=password,
                skip_pooler=skip_pooler,
                dry_run=dry_run,
                include_all=include_all,
            )
        )
    print("\n=== SUPABASE BUNDLE ===\n")
    print(f"Generated bundle in: {bundle_path}")
    print(f"Migrations bundled: {len(bundled)}")
    if push:
        print("Supabase push completed from the generated bundle.")
    else:
        print("Run Supabase CLI from the generated workspace, for example:")
        if project_ref:
            print(f"  cd {bundle_path} && supabase link --project-ref {project_ref}")
            print(f"  cd {bundle_path} && supabase db push")
        elif db_url:
            print(f"  cd {bundle_path} && supabase db push --db-url 'postgresql://...'")
        else:
            print(f"  cd {bundle_path} && supabase link --project-ref YOUR_PROJECT_REF")
            print(f"  cd {bundle_path} && supabase db push")
