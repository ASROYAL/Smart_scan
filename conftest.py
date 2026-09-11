"""Ensure the src/ layout is importable during tests regardless of install state."""
import sys
from pathlib import Path

_SRC = Path(__file__).parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
