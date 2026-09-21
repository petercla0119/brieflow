import sys
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[2] / "workflow"
if str(WORKFLOW) not in sys.path:
    sys.path.insert(0, str(WORKFLOW))
