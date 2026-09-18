"""Command line entry point for pgvb. Only `version` is implemented until later phases."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pgvb import __version__

_PROFILES_DIR = Path(__file__).resolve().parents[2] / "profiles"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pgvb")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("version", help="print the installed pgvb version")
    app_parser = subparsers.add_parser("app", help="launch the desktop app")
    app_parser.add_argument("--profile", default="example", help="profile name (no .json)")
    app_parser.add_argument("--camera", type=int, default=None, help="camera index override")
    subparsers.add_parser("demo", help="run a demo pipeline (added in later phases)")

    args = parser.parse_args(argv)

    if args.command == "version":
        print(__version__)
        return 0
    if args.command == "app":
        from pgvb.gui.app import main as app_main

        profile_path = _PROFILES_DIR / f"{args.profile}.json"
        return app_main(profile_path, args.camera)
    if args.command == "demo":
        print("pgvb demo is not implemented yet", file=sys.stderr)
        return 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
