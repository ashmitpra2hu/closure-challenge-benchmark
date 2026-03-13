#!/usr/bin/env python3
"""Rebuild the NASA 2D wall-mounted hump case from the canonical benchmark repo.

This utility is intended for a "delete-and-regenerate" validation exercise:
1. Optionally back up an existing local `data/NASA_2DWMH` directory.
2. Remove the local case directory.
3. Re-download the canonical case from GitHub.
4. Compare the regenerated case against the backup (if available).

Example:
    python scripts/rebuild_nasa_hump_case.py --compare
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

CANONICAL_REPO_ZIP = "https://github.com/rmcconke/closure-challenge-benchmark/archive/refs/heads/main.zip"
CASE_REL_PATH = Path("data") / "NASA_2DWMH"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def case_manifest(case_dir: Path) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for file_path in sorted(p for p in case_dir.rglob("*") if p.is_file()):
        rel = file_path.relative_to(case_dir).as_posix()
        manifest[rel] = sha256_file(file_path)
    return manifest


def restore_from_zip(repo_root: Path, zip_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="hump_unzip_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_root)

        extracted_root = next(tmp_root.glob("closure-challenge-benchmark-*"), None)
        if extracted_root is None:
            raise RuntimeError("Could not locate extracted benchmark root in downloaded archive.")

        src = extracted_root / CASE_REL_PATH
        dst = repo_root / CASE_REL_PATH
        if not src.exists():
            raise RuntimeError(f"Canonical archive does not contain {CASE_REL_PATH}.")

        shutil.copytree(src, dst)


def restore_from_git(repo_root: Path) -> None:
    result = subprocess.run(
        ["git", "checkout", "HEAD", "--", str(CASE_REL_PATH)],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to restore NASA case from git checkout fallback:\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def download_zip(dest: Path) -> None:
    with urlopen(CANONICAL_REPO_ZIP) as response:  # nosec: B310 (fixed trusted URL)
        dest.write_bytes(response.read())


def run_case(repo_root: Path) -> int:
    run_script = repo_root / CASE_REL_PATH / "run.sh"
    if not run_script.exists():
        print(f"Skipping run: {run_script} not found.")
        return 0

    if shutil.which("simpleFoam") is None:
        print("Skipping run: OpenFOAM (simpleFoam) is not available in PATH.")
        return 0

    print("Running NASA hump case via run.sh ...")
    result = subprocess.run(["bash", str(run_script)], cwd=run_script.parent)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Path to the benchmark repository root (default: current working directory).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare regenerated case against a backup of the deleted local case.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Try to run the regenerated case with OpenFOAM (if available).",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    target_case = repo_root / CASE_REL_PATH
    backup_case = repo_root / "data" / ".NASA_2DWMH_backup"

    if target_case.exists():
        print(f"Backing up existing case to: {backup_case}")
        if backup_case.exists():
            shutil.rmtree(backup_case)
        shutil.copytree(target_case, backup_case)
        print(f"Deleting existing case: {target_case}")
        shutil.rmtree(target_case)
    else:
        print(f"No existing case found at {target_case}; proceeding with fresh restore.")

    restored = False
    with tempfile.TemporaryDirectory(prefix="hump_download_") as tmp_dir:
        zip_path = Path(tmp_dir) / "benchmark_main.zip"
        print(f"Downloading canonical NASA hump case from: {CANONICAL_REPO_ZIP}")
        try:
            download_zip(zip_path)
            print("Restoring case files from downloaded archive...")
            restore_from_zip(repo_root, zip_path)
            restored = True
        except Exception as exc:
            print(f"Download-based restore failed: {exc}")

    if not restored:
        print("Falling back to local git checkout restore...")
        restore_from_git(repo_root)

    print(f"Recreated case at: {target_case}")

    if args.compare:
        if not backup_case.exists():
            print("No backup available to compare against; use --compare only when replacing an existing case.")
        else:
            new_manifest = case_manifest(target_case)
            old_manifest = case_manifest(backup_case)
            if new_manifest == old_manifest:
                print("Comparison result: MATCH (regenerated case is byte-identical to backup).")
            else:
                print("Comparison result: DIFFERENT")
                added = sorted(set(new_manifest) - set(old_manifest))
                removed = sorted(set(old_manifest) - set(new_manifest))
                changed = sorted(
                    file for file in set(new_manifest) & set(old_manifest) if new_manifest[file] != old_manifest[file]
                )
                diff_report = {
                    "added": added,
                    "removed": removed,
                    "changed": changed,
                }
                report_path = repo_root / "data" / "NASA_2DWMH_diff_report.json"
                report_path.write_text(json.dumps(diff_report, indent=2))
                print(f"Wrote diff report: {report_path}")
                return 1

    if args.run:
        run_code = run_case(repo_root)
        if run_code != 0:
            print(f"Run failed with exit code {run_code}")
            return run_code

    return 0


if __name__ == "__main__":
    sys.exit(main())
