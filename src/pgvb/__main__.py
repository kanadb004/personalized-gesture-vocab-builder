"""Command line entry point for pgvb. Only `version` is implemented until later phases."""

from __future__ import annotations

import argparse
import sys

from pgvb import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pgvb")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("version", help="print the installed pgvb version")
    subparsers.add_parser("app", help="launch the desktop app (added in Phase 7)")
    subparsers.add_parser("demo", help="run a demo pipeline (added in later phases)")

    args = parser.parse_args(argv)

    if args.command == "version":
        print(__version__)
        return 0
    if args.command == "app":
        print("pgvb app is not implemented until Phase 7", file=sys.stderr)
        return 1
    if args.command == "demo":
        print("pgvb demo is not implemented yet", file=sys.stderr)
        return 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
