from __future__ import annotations

import argparse
from pathlib import Path

from .aozora import build_aozora, selected_source_paths
from .common import write_json
from .discover import (
    AOZORA_METADATA_URL,
    AOZORA_REPOSITORY_URL,
    WIKIMEDIA_ROOT,
    discover_sources,
)
from .manifest import build_manifest, verify_remote_assets
from .wikipedia import build_wikipedia


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="japanese-corpus",
        description="Build a redistributable Japanese corpus from open sources.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser("discover", help="Discover current source versions")
    discover.add_argument("--output", type=Path, required=True)
    discover.add_argument("--wikipedia-root", default=WIKIMEDIA_ROOT)
    discover.add_argument("--aozora-repository", default=AOZORA_REPOSITORY_URL)
    discover.add_argument("--aozora-metadata", default=AOZORA_METADATA_URL)

    wikipedia = subparsers.add_parser(
        "build-wikipedia", help="Build one Wikipedia corpus shard"
    )
    wikipedia.add_argument("input", type=Path)
    wikipedia.add_argument("output", type=Path)
    wikipedia.add_argument("--stats", type=Path, required=True)
    wikipedia.add_argument("--dump-date", required=True)
    wikipedia.add_argument("--limit", type=int, default=0)

    aozora = subparsers.add_parser("build-aozora", help="Build the Aozora Bunko corpus")
    aozora.add_argument("--metadata-zip", type=Path, required=True)
    aozora.add_argument("--source-root", type=Path, required=True)
    aozora.add_argument("--output", type=Path, required=True)
    aozora.add_argument("--stats", type=Path, required=True)
    aozora.add_argument("--source-commit", required=True)
    aozora.add_argument("--limit", type=int, default=0)

    paths = subparsers.add_parser(
        "list-aozora-paths", help="List public-domain source ZIP paths"
    )
    paths.add_argument("--metadata-zip", type=Path, required=True)
    paths.add_argument("--output", type=Path, required=True)

    manifest = subparsers.add_parser(
        "build-manifest", help="Build a release manifest and checksums"
    )
    manifest.add_argument("--stats-dir", type=Path, required=True)
    manifest.add_argument("--discovery", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--checksums", type=Path, required=True)
    manifest.add_argument("--version", required=True)
    manifest.add_argument("--pipeline-commit", required=True)
    manifest.add_argument("--built-at")

    verify = subparsers.add_parser(
        "verify-assets", help="Verify uploaded Release asset names and sizes"
    )
    verify.add_argument("--stats-dir", type=Path, required=True)
    verify.add_argument("--assets-json", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "discover":
        write_json(
            args.output,
            discover_sources(
                wikipedia_root=args.wikipedia_root,
                aozora_repository=args.aozora_repository,
                aozora_metadata=args.aozora_metadata,
            ),
        )
    elif args.command == "build-wikipedia":
        build_wikipedia(
            input_path=args.input,
            output_path=args.output,
            stats_path=args.stats,
            dump_date=args.dump_date,
            limit=args.limit,
        )
    elif args.command == "build-aozora":
        build_aozora(
            metadata_zip=args.metadata_zip,
            source_root=args.source_root,
            output_path=args.output,
            stats_path=args.stats,
            source_commit=args.source_commit,
            limit=args.limit,
        )
    elif args.command == "list-aozora-paths":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(f"{path}\n" for path in selected_source_paths(args.metadata_zip)),
            encoding="utf-8",
            newline="\n",
        )
    elif args.command == "build-manifest":
        build_manifest(
            stats_directory=args.stats_dir,
            discovery_path=args.discovery,
            output_path=args.output,
            checksums_path=args.checksums,
            version=args.version,
            pipeline_commit=args.pipeline_commit,
            built_at=args.built_at,
        )
    elif args.command == "verify-assets":
        verify_remote_assets(args.stats_dir, args.assets_json)
    else:
        raise AssertionError(f"Unhandled command {args.command}")
