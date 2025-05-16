from os import environ
from os.path import abspath, dirname, join
from pathlib import Path
from subprocess import run
import sys

SCRIPTS_DIR = join(dirname(abspath(sys.argv[0])), "experiments")
MODES = [
    "classification",
    "detection",
    "train_classification",
    "train_detection",
]

if __name__ == "__main__":
    for script in [file for file in Path(SCRIPTS_DIR).glob("*.py") if file.is_file()]:
        for mode in MODES:
            print([sys.executable, str(script), mode])
            run([sys.executable, str(script), mode], check=True)
