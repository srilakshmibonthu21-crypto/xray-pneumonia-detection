"""
Streamlit Cloud Entry Point (Alias for app.py)
"""
import runpy
from pathlib import Path

# Run app.py in the same namespace
app_path = Path(__file__).resolve().parent / "app.py"
runpy.run_path(str(app_path), run_name="__main__")
