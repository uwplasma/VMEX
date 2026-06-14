"""Report Python source-file sizes for maintainability refactors.

This diagnostic is intentionally lightweight and dependency-free.  It is meant
to make large-file refactors measurable before they become strict CI gates.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_ROOTS = ("vmec_jax", "examples/optimization", "tests")


@dataclass(frozen=True)
class SourceFileStat:
    """Line-count record for one Python source file."""

    path: Path
    lines: int


def count_source_lines(path: Path) -> int:
    """Return the number of physical lines in a Python source file."""

    with path.open("rb") as stream:
        return sum(1 for _ in stream)


def iter_python_files(roots: Iterable[Path]) -> Iterable[Path]:
    """Yield Python files below the requested roots in deterministic order."""

    for root in sorted(roots):
        if root.is_file():
            if root.suffix == ".py":
                yield root
            continue
        if root.is_dir():
            yield from sorted(root.rglob("*.py"))


def collect_source_stats(roots: Iterable[Path]) -> list[SourceFileStat]:
    """Collect source line counts sorted largest first."""

    stats = [SourceFileStat(path=path, lines=count_source_lines(path)) for path in iter_python_files(roots)]
    return sorted(stats, key=lambda item: (-item.lines, str(item.path)))


def format_source_health_report(
    stats: Iterable[SourceFileStat],
    *,
    top: int,
    warn_lines: int,
) -> str:
    """Format a source-health report for terminals and PR comments."""

    selected = list(stats)[:top]
    if not selected:
        return "No Python files found."

    path_width = max(len(str(item.path)) for item in selected)
    lines = ["Python source-health report", f"warning threshold: {warn_lines} lines", ""]
    for item in selected:
        marker = "WARN" if item.lines >= warn_lines else "    "
        lines.append(f"{marker}  {item.lines:6d}  {str(item.path):<{path_width}}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        default=list(DEFAULT_ROOTS),
        help="Files or directories to scan. Defaults to vmec_jax, examples/optimization, and tests.",
    )
    parser.add_argument("--top", type=int, default=30, help="Number of largest files to print.")
    parser.add_argument("--warn-lines", type=int, default=2000, help="Mark files at or above this line count.")
    parser.add_argument(
        "--fail-lines",
        type=int,
        default=0,
        help="Exit nonzero if any scanned file is at or above this line count. Disabled by default.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    roots = [Path(root) for root in args.roots]
    stats = collect_source_stats(roots)
    print(format_source_health_report(stats, top=args.top, warn_lines=args.warn_lines))

    if args.fail_lines > 0 and any(item.lines >= args.fail_lines for item in stats):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
