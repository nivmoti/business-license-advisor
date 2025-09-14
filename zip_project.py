#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zip_project.py — Package the whole project into a clean ZIP.

Defaults:
- Includes EVERYTHING in the repo (including data/**).
- Excludes only obvious noise: .git, venvs, node_modules, __pycache__, IDE folders, temp/cache files.

Optional flags:
- --exclude-heavy : also exclude data/raw/** and debug/** (useful when you *don't* want heavy artifacts)
- --output <file> : custom zip name
- --extra-exclude <glob> : add more exclude patterns (can repeat)

Examples:
  python zip_project.py
  python zip_project.py --output release.zip
  python zip_project.py --exclude-heavy
  python zip_project.py --extra-exclude "*.log" --extra-exclude "frontend/.next/**"
"""
from __future__ import annotations
import argparse, fnmatch, sys, zipfile
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent

# Minimal defaults: include everything except obvious noise
BASE_EXCLUDES: List[str] = [
    ".git/**", ".git/*", ".git",
    ".venv/**", "venv/**",
    "node_modules/**",
    "**/__pycache__/**",
    ".pytest_cache/**", ".mypy_cache/**", ".ruff_cache/**",
    ".idea/**", ".vscode/**",
    "*.pyc", "*.pyo", "*.pyd",
    "*.log", "*.tmp", "*.temp", "*.swp",
    ".DS_Store", "Thumbs.db",
    "dist/**", "build/**",
    ".coverage*", "coverage.xml",
]

# “Heavy” data you might want to EXCLUDE *only* when passing --exclude-heavy
HEAVY_EXCLUDES: List[str] = [
    "data/raw/**",
    "debug/**",
]

def load_gitignore_patterns(gitignore_path: Path) -> List[str]:
    if not gitignore_path.exists():
        return []
    pats = []
    for line in gitignore_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.replace("\\", "/")
        if line.endswith("/"):
            line += "**"
        pats.append(line)
    return pats

def make_matcher(patterns: List[str]):
    def _match(rel_posix: str) -> bool:
        for pat in patterns:
            if fnmatch.fnmatch(rel_posix, pat):
                return True
        return False
    return _match

def collect_files(root: Path, excludes: List[str]) -> List[Path]:
    files: List[Path] = []
    match = make_matcher(excludes)
    for p in root.rglob("*"):
        if p.is_dir():
            # We don't need to prune dirs aggressively; file-level filter is enough.
            continue
        rel = p.relative_to(root).as_posix()
        if match(rel):
            continue
        files.append(p)
    return files

def main():
    ap = argparse.ArgumentParser(description="Zip the project while ignoring noise.")
    ap.add_argument("--output", "-o", default="project.zip", help="Output zip file name")
    ap.add_argument("--exclude-heavy", action="store_true",
                    help="Also exclude data/raw/** and debug/**")
    ap.add_argument("--extra-exclude", action="append", default=[],
                    help="Extra glob(s) to exclude (can repeat)")
    args = ap.parse_args()

    excludes = list(BASE_EXCLUDES)
    excludes.extend(load_gitignore_patterns(ROOT / ".gitignore"))
    if args.exclude_heavy:
        excludes.extend(HEAVY_EXCLUDES)
    for pat in args.extra_exclude:
        excludes.append(pat.replace("\\", "/"))

    files = collect_files(ROOT, excludes)

    out = (ROOT / args.output).resolve()
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, f.relative_to(ROOT).as_posix())

    print(f"✔ Wrote {out.name} with {len(files)} files from '{ROOT.name}'")
    print("Excluded patterns:")
    for pat in sorted(set(excludes)):
        print("  -", pat)

if __name__ == "__main__":
    sys.exit(main())
