"""Test runner that is immune to a broken or restricted system temp directory.

Why this exists
---------------
Some of the tests use pytest's ``tmp_path`` fixture, which creates throwaway
working directories under the system temp. On locked-down machines (university
labs, corporate images, CI sandboxes) that directory can be missing, unreadable,
or held by a dead process — and every fixture-dependent test then fails with
``Access is denied`` even though the code is fine.

The fix is to run pytest with TMP/TEMP pointed at a *fresh* folder inside this
project, which is always writable. The folder name is unique per run, and old
ones are cleaned up so nothing accumulates.

Usage (from the project root):

    python scripts/run_tests.py            # full suite
    python scripts/run_tests.py -k api     # anything pytest accepts after --

Arguments after ``--`` are passed to pytest verbatim. The exit code is pytest's.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = PROJECT_ROOT / ".pytest_runs"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the test suite with a project-local temp directory"
    )
    parser.parse_known_args()

    # Pass anything after "--" straight through to pytest.
    passthrough = sys.argv[1:]
    if "--" in passthrough:
        passthrough = passthrough[passthrough.index("--") + 1 :]
    else:
        # Also accept plain pytest flags like -k without the separator.
        passthrough = [a for a in passthrough if a.startswith("-")]

    TMP_ROOT.mkdir(exist_ok=True)

    # Sweep runners older than a day; keeps the folder tidy without racing
    # a concurrent run (a live run holds files open on Windows).
    import time

    cutoff = time.time() - 86_400
    for old in TMP_ROOT.iterdir():
        try:
            if old.stat().st_mtime < cutoff:
                shutil.rmtree(old, ignore_errors=True)
        except OSError:
            pass  # unreadable entry: leave it, it is harmless

    run_dir = TMP_ROOT / f"run-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir()

    # Scratch space for anything the code under test might spill into TMP
    # (library-internal tempfiles etc.). Fixture directories are created by
    # conftest directly inside the project and never touch the system temp,
    # so this is belt-and-braces, not load-bearing.
    env_bin = run_dir / "bin"
    env_bin.mkdir()

    # Python's tempfile module (and pytest's tmp_path via TMP) now resolve
    # inside the project. PYTHONPATH must be set explicitly because we are
    # launching pytest as a module, not from an installed package.
    import os

    env = os.environ.copy()
    env["TMP"] = str(env_bin)
    env["TEMP"] = str(env_bin)
    env["TMPDIR"] = str(env_bin)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + env.get(
        "PYTHONPATH", ""
    )
    # A cache that cannot be written should warn, not fail the run.
    env["PYTEST_ADDOPTS"] = env.get("PYTEST_ADDOPTS", "") + " -p no:cacheprovider"

    command = [sys.executable, "-u", "-m", "pytest", "tests", "-q", *passthrough]

    print(f"[run_tests] scratch = {env_bin}")
    print(f"[run_tests] pytest command: {' '.join(command[2:])}")
    print()

    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), env=env)

    # Best effort cleanup; a failed sweep does not fail the run.
    shutil.rmtree(run_dir, ignore_errors=True)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
