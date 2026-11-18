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

# ---------------------------------------------------------------------------
# The three datasets.
#
# Vocabulary follows the assignment, not scikit-learn:
#   TRAIN   -- fits the models
#   TEST    -- held-out matches from the same competitions; used while
#              developing, to check the pipeline and compare with the paper
#   UNSEEN  -- an entirely separate competition, never touched until the
#              training and test work is finished ("validation" in the
#              assignment's terms)
#
# The unseen set is a different *competition* rather than a random slice of
# the same one. That is a stricter test of generalisation, and it is the
# project's research question: does a creativity metric trained on one
# tournament transfer to another?
# ---------------------------------------------------------------------------
TRAIN_COMPETITIONS = ["euro2020", "euro2024"]
UNSEEN_COMPETITION = "worldcup2022"

# Fraction of training-pool *matches* (not passes) held out as the test set.
TEST_FRACTION = 0.2
SPLIT_SEED = 42

# Name under which the pooled training competitions are stored.
POOL_NAME = "euros"

# Backwards-compatible aliases.
TRAIN_COMPETITION = TRAIN_COMPETITIONS[0]
TEST_COMPETITION = UNSEEN_COMPETITION
VALIDATION_FRACTION = TEST_FRACTION
