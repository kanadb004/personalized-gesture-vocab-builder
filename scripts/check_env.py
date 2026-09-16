#!/usr/bin/env python
"""Imports the pinned runtime dependencies and prints their versions. Exits non-zero on
any import failure or if numpy 2 is installed (tensorflow 2.16 requires numpy<2)."""

from __future__ import annotations

import sys


def main() -> int:
    failures: list[str] = []

    def check(name: str, import_name: str | None = None) -> None:
        module_name = import_name or name
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"{name}: {version}")
        except Exception as exc:  # noqa: BLE001 - report any import failure
            failures.append(f"{name}: FAILED ({exc})")
            print(f"{name}: FAILED ({exc})", file=sys.stderr)

    check("numpy")
    check("cv2", "cv2")
    check("mediapipe")
    check("pyttsx3")
    check("yaml")

    try:
        import tkinter

        print(f"tkinter: {tkinter.TkVersion}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"tkinter: FAILED ({exc})")
        print(f"tkinter: FAILED ({exc})", file=sys.stderr)

    try:
        import numpy

        major = int(numpy.__version__.split(".")[0])
        if major >= 2:
            failures.append(f"numpy: version {numpy.__version__} is numpy 2, expected numpy<2")
            print(
                f"numpy {numpy.__version__} is numpy 2, but tensorflow 2.16 requires numpy<2",
                file=sys.stderr,
            )
    except Exception:
        pass  # already reported by check("numpy") above

    if failures:
        print(f"\n{len(failures)} check(s) failed.", file=sys.stderr)
        return 1

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
