"""Project paths and competition definitions.

Everything that is specific to *this* project (as opposed to the upstream
un-xPass code) is declared here so the rest of the pipeline stays generic.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths. ROOT is the repository root; override with the UNXPASS_ROOT env var.
# ---------------------------------------------------------------------------
import os

ROOT = Path(os.environ.get("UNXPASS_ROOT", Path(__file__).resolve().parent.parent))
STORES = ROOT / "stores"
DB_PATH = STORES / "database.sqlite"
DATASETS = STORES / "datasets"
MODELS = STORES / "models"
RESULTS = ROOT / "results"

for _d in (STORES, DATASETS, MODELS, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)


def db_uri() -> str:
    """SQLite URI for the project database.

    NOTE: un-xPass parses the URI with ``urlparse`` and uses ``.path``
    verbatim, so a relative path such as ``sqlite:///stores/db.sqlite``
    resolves to ``/stores/db.sqlite`` at the filesystem root and fails with
    "unable to open database file". An absolute path is required.
    """
    return "sqlite:///" + DB_PATH.resolve().as_posix()


# ---------------------------------------------------------------------------
# Competitions. StatsBomb open-data identifiers.
# Verified against https://github.com/statsbomb/open-data competitions.json
# ---------------------------------------------------------------------------
COMPETITIONS = {
    "euro2020": {"competition_id": 55, "season_id": 43, "label": "UEFA Euro 2020"},
    "worldcup2022": {"competition_id": 43, "season_id": 106, "label": "FIFA World Cup 2022"},
    "euro2024": {"competition_id": 55, "season_id": 282, "label": "UEFA Euro 2024"},
}

# Train/validate on Euro 2020; hold World Cup 2022 back entirely as the
# unseen test competition.
TRAIN_COMPETITION = "euro2020"
TEST_COMPETITION = "worldcup2022"

# Fraction of Euro 2020 *matches* (not passes) held out for validation.
VALIDATION_FRACTION = 0.2
SPLIT_SEED = 42
