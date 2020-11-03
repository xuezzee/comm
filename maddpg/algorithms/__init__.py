import sys
from pathlib import Path
base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))
sys.path.append(str(Path(__file__).resolve()))
print()