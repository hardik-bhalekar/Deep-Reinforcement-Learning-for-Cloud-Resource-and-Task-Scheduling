"""Root entrypoint for Streamlit dashboard.

Run with:
    streamlit run app.py
"""

import os
import sys
from pathlib import Path
import runpy

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Delegate execution to the dashboard module
dashboard_script = PROJECT_ROOT / "src" / "dashboard" / "app.py"
runpy.run_path(str(dashboard_script), run_name="__main__")
