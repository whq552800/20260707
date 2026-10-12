"""Location of the input panel (repository-relative)."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
PANEL = str(ROOT / 'data' / 'under5_disease_factor_annual_panel_1990_2023.csv')
