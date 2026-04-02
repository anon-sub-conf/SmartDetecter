import os
import subprocess
import sys
from pathlib import Path

def run(command):
    base_dir = Path(__file__).resolve().parent
    path = base_dir / 'testContracts'
    files = os.listdir(path)
    counts = 0
    names = []
    for file in files:
        if '.sol' in file:
            counts += 1
            names.append(file)
    if counts == 2:
        srs_path = path / 'SRs.txt'
        if srs_path.exists():
            srs_path.unlink()
        subprocess.run([sys.executable, '-m', 'solidity_parser', 'parse', str(path / names[0])], cwd=base_dir, check=True)
        subprocess.run([sys.executable, '-m', 'solidity_parser', 'parse', str(path / names[1])], cwd=base_dir, check=True)
        subprocess.run([sys.executable, 'get_feature.py'], cwd=base_dir, check=True)
        subprocess.run([sys.executable, 'lightgbm_smart.py', command], cwd=base_dir, check=True)
    else:
        print('The number of contracts tested must be two')
        sys.exit(1)

if __name__ == "__main__":
    if not len(sys.argv) > 1 or sys.argv[1] not in ("--train", "--test"):
        print("\n- Missing subcommand.\n  Please choose --train or --test")
        sys.exit(1)
    if sys.argv[1] == "--train":
        run('--train')
    elif sys.argv[1] == "--test":
        run('--test')
