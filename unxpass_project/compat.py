"""Compatibility patches for running un-xPass (2023) on a modern Python stack.

Import this FIRST, before anything from ``unxpass``::

    from unxpass_project import compat  # noqa: F401

Each patch documents the breakage it fixes. Verified against:
Python 3.10, torch 1.13.1, socceraction 1.5.3, pandas 2.1-2.3, xgboost 1.7.6.
"""
import logging
import logging.config  # noqa: F401  <- PATCH 1
import re
import warnings
import os

import pandas as pd


# ---------------------------------------------------------------------------
# PATCH 1 -- unxpass/config.py calls ``logging.config.dictConfig`` after only
# ``import logging``. It used to work because some dependency imported
# logging.config as a side effect. Importing it here, before unxpass loads,
# restores the attribute.
#
# PATCH 2 -- MLflow's filesystem tracking backend is in maintenance mode and
# raises instead of logging. We opt back in; nothing here needs a DB backend.
# ---------------------------------------------------------------------------
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")


# ---------------------------------------------------------------------------
# PATCH 3 -- socceraction.vaep.features.gamestates builds the previous-action
# frames with::
#
#     x.shift(i, fill_value=nan).fillna(x.iloc[0])
#
# Under pandas >= 2.1 this raises
#     TypeError: "value" parameter must be a scalar or dict, but you passed a "list"
# because the StatsBomb 360 action frame carries list-valued columns
# (freeze_frame_360, visible_area_360) and fillna refuses a list as a fill
# value. We reimplement the identical semantics with positional slicing, which
# is dtype-agnostic.
# ---------------------------------------------------------------------------
def _shift_within_group(x: pd.DataFrame, i: int) -> pd.DataFrame:
    """Shift ``x`` down ``i`` rows, padding the top with its own first row."""
    n = len(x)
    if n == 0:
        return x.copy()
    k = min(i, n)
    head = pd.concat([x.iloc[[0]]] * k, ignore_index=True)
    tail = x.iloc[: n - k] if n - k > 0 else x.iloc[0:0]
    out = pd.concat([head, tail], ignore_index=True)
    out.index = x.index
    return out


def _gamestates(actions: pd.DataFrame, nb_prev_actions: int = 3):
    if nb_prev_actions < 1:
        raise ValueError("The game state should include at least one preceding action.")
    states = [actions]
    for i in range(1, nb_prev_actions):
        with warnings.catch_warnings():
            # pandas >= 2.2 warns that apply() sees the grouping columns. We
            # want them: the shifted frame has to keep game_id and period_id.
            warnings.filterwarnings(
                "ignore",
                message=".*DataFrameGroupBy.apply operated on the grouping columns.*",
                category=FutureWarning,
            )
            prev = actions.groupby(
                ["game_id", "period_id"], sort=False, group_keys=False
            ).apply(_shift_within_group, i)
        prev.index = actions.index.copy()
        states.append(prev)
    return states


# ---------------------------------------------------------------------------
# PATCH 5 -- unxpass.databases.connect() parses the URI with urlparse and uses
# ``.path`` verbatim. On Windows an absolute path carries a drive letter, so
#     sqlite:///C:/Users/me/stores/db.sqlite
# yields ``/C:/Users/me/stores/db.sqlite`` -- a leading slash before the drive
# letter, which sqlite3 cannot open ("unable to open database file"). We strip
# that slash. Linux and macOS paths are left untouched.
# ---------------------------------------------------------------------------
_DRIVE_PREFIX = re.compile(r"^/([A-Za-z]:[/\\])")


def _normalise_db_path(path: str) -> str:
    return _DRIVE_PREFIX.sub(r"\1", path)


def _connect(uri: str, mode: str = "r"):
    from pathlib import Path
    from urllib.parse import urlparse

    from unxpass.databases.hdf import HDFDatabase
    from unxpass.databases.sqlite import SQLiteDatabase

    parsed = urlparse(uri)
    db_path = Path(_normalise_db_path(parsed.path))
    if parsed.scheme in ("h5", "hdf5", "hdf"):
        return HDFDatabase(db_path, mode=mode)
    if parsed.scheme == "sqlite":
        return SQLiteDatabase(db_path, mode=mode)
    raise ValueError(f"Unsupported database type: {parsed.scheme!r}")


# ---------------------------------------------------------------------------
# PATCH 6 -- unxpass/config.py sets the MLflow tracking URI with
#     mlflow.set_tracking_uri("file://" + str(MODEL_REGISTRY.absolute()))
# On POSIX the path already starts with "/", so this yields a valid
# "file:///home/..." by accident. On Windows the path starts with a drive
# letter, producing "file://C:\Users\..." -- only two slashes, so MLflow reads
# "C:" as a hostname and raises
#     MlflowException: ... is not a valid remote uri.
# A bare local path does not work either -- MLflow then reads "C:" as a URI
# scheme. We wrap set_tracking_uri and rewrite the value into a canonical
# three-slash file URI with forward slashes: file:///C:/Users/...
# ---------------------------------------------------------------------------
_FILE_DRIVE_URI = re.compile(r"^file://([A-Za-z]:[/\\])")


def _patch_mlflow_tracking_uri() -> None:
    import mlflow

    if getattr(mlflow.set_tracking_uri, "_unxpass_patched", False):
        return
    original = mlflow.set_tracking_uri

    def set_tracking_uri(uri):
        if isinstance(uri, str) and _FILE_DRIVE_URI.match(uri):
            # "file://C:\\Users\\x" -> "file:///C:/Users/x"
            path = uri[len("file://") :].replace("\\", "/").lstrip("/")
            uri = "file:///" + path
        return original(uri)

    set_tracking_uri._unxpass_patched = True
    mlflow.set_tracking_uri = set_tracking_uri


def apply_patches() -> None:
    """Apply every patch. Idempotent."""
    # Must come first: unxpass.config calls set_tracking_uri at import time.
    _patch_mlflow_tracking_uri()

    import socceraction.vaep.features as sa_features

    sa_features.gamestates = _gamestates

    import unxpass.databases as sa_db

    sa_db.connect = _connect
    try:
        import socceraction.spadl.utils as sa_utils

        if hasattr(sa_utils, "gamestates"):
            sa_utils.gamestates = _gamestates
    except Exception:  # pragma: no cover
        pass
    logging.getLogger(__name__).debug("un-xPass compatibility patches applied")


apply_patches()


# ---------------------------------------------------------------------------
# PATCH 4 -- unxpass.components.base calls mlflow.{sklearn,xgboost,pytorch}
# .autolog() inside each component's train(). Modern MLflow then tries to
# register a logged model against experiment "0", which does not exist unless a
# run has been started, and raises
#     MlflowException: Could not find experiment with ID 0
# This pipeline does not use MLflow tracking -- models are pickled to
# stores/models -- so autologging is turned into a no-op. Remove this patch if
# you later want MLflow tracking, and wrap training in mlflow.start_run().
# ---------------------------------------------------------------------------
def disable_mlflow_autolog() -> None:
    import mlflow

    def _noop(*_args, **_kwargs):
        return None

    for mod in ("sklearn", "xgboost", "pytorch"):
        try:
            getattr(mlflow, mod).autolog = _noop
        except Exception:  # pragma: no cover
            pass
    try:
        mlflow.autolog(disable=True)
    except Exception:  # pragma: no cover
        pass


disable_mlflow_autolog()
