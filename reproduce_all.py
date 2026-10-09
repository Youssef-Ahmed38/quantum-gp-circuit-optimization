"""Recheck the saved research package using only local computation."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def run(*args: str) -> None:
    command = [sys.executable, *args]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    run("scripts/verify_manifest.py")
    run("-m", "pytest", "-q", "tests")
    run("examples/quick_reproduce_pass.py")
    run("benchmarks/verify_evidence.py")
    run("benchmarks/analyze_observables.py")
    run("benchmarks/plot_observable_figures.py")
    run("benchmarks/analyze_physical.py")
    print("Local checks and analyses completed.")


if __name__ == "__main__":
    main()
