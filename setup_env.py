from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="python3.10")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    python_executable = shutil.which(args.python)
    if python_executable is None:
        raise SystemExit(f"Could not find '{args.python}' on PATH.")

    venv_dir = base_dir / ".venv"
    subprocess.run([python_executable, "-m", "venv", str(venv_dir)], cwd=base_dir, check=True)
    subprocess.run([str(venv_dir / "bin" / "python"), "-m", "pip", "install", "--upgrade", "pip"], cwd=base_dir, check=True)
    subprocess.run([str(venv_dir / "bin" / "python"), "-m", "pip", "install", "-r", "requirements.txt"], cwd=base_dir, check=True)

    print(f"Virtualenv created at {venv_dir}")
    print(f"Activate with: source {venv_dir / 'bin' / 'activate'}")


if __name__ == "__main__":
    main()
