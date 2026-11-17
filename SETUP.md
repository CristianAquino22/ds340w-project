# Setup

## 1. Python 3.10

`torch==1.13.1` is pinned exactly by the upstream project and ships no wheels
for Python 3.11+. Check with `python --version`; install 3.10 if needed.

```bash
py -3.10 -m venv .venv          # Windows
.venv\Scripts\activate
```

## 2. Dependencies

```bash
pip install -r requirements.txt
```

## 3. The upstream code

```bash
git clone https://github.com/ML-KULeuven/un-xPass.git vendor/un-xPass
pip install --no-deps -e vendor/un-xPass
```

`--no-deps` matters: the upstream `pyproject.toml` resolves to versions that no
longer work together, and `requirements.txt` above is the set that does.

## 4. Verify

```bash
python -c "from unxpass_project import compat; from unxpass.ratings import CreativeDecisionRating; print('ok')"
```

---

# What broke, and why

Ten things stopped a 2023 research codebase from running in late 2026. Three
are pinned in `requirements.txt`; the rest needed code changes, all in
`unxpass_project/compat.py` and `models.py`, so the upstream tree stays
untouched.

Patches 4-10 were found by running the pipeline end to end on Linux and then
on Windows. #9 and #10 appear only on Windows: both come from code that builds
a URI by string concatenation and happens to be correct only when the path
starts with a slash.

| # | Symptom | Cause | Fix |
|---|---------|-------|-----|
| 1 | CLI crashes on import | `click>=8.2` changed its API; `typer` 0.9 expects the old one | pin `click<8.2` |
| 2 | dispatch errors in socceraction | `multimethod>=2.0` changed resolution | pin `multimethod<2.0` |
| 3 | `ModuleNotFoundError: pkg_resources` | `setuptools>=81` removed it; `lightning_fabric` still calls `declare_namespace` | pin `setuptools<81` |
| 4 | `AttributeError: module 'logging' has no attribute 'config'` | `unxpass/config.py` does `import logging` then calls `logging.config.dictConfig`; it only ever worked because a dependency imported the submodule as a side effect | `import logging.config` in `compat.py`, before `unxpass` loads |
| 5 | `sqlite3.OperationalError: unable to open database file` | `connect()` runs the URI through `urlparse` and uses `.path` verbatim, so `sqlite:///stores/db.sqlite` resolves to `/stores/db.sqlite` at filesystem root | always pass an absolute path (`config.db_uri()` does) |
| 6 | `TypeError: "value" parameter must be a scalar or dict, but you passed a "list"` | `socceraction.vaep.features.gamestates` does `x.shift(i).fillna(x.iloc[0])`; the StatsBomb 360 frame has list-valued columns (`freeze_frame_360`, `visible_area_360`) and pandas >= 2.1 refuses a list as a fill value | `compat.py` PATCH 3 reimplements `gamestates` with positional slicing |
| 7 | MLflow raises instead of logging | the filesystem tracking backend is in maintenance mode | `MLFLOW_ALLOW_FILE_STORE=true`, set in `compat.py` |
| 8 | `MlflowException: Could not find experiment with ID 0` | components call `mlflow.*.log_model()` internally, which needs an active run | `models.train()` wraps training in `mlflow.start_run()` |
| 9 | `sqlite3.OperationalError: unable to open database file` **on Windows** | same `urlparse` path as #5, but a Windows absolute path carries a drive letter, so `sqlite:///C:/...` yields `/C:/...` -- a slash before the drive letter | `compat.py` PATCH 5 replaces `connect()` and strips it |
| 10 | `MlflowException: file://C:\... is not a valid remote uri` **on Windows** | `unxpass/config.py` builds the tracking URI as `"file://" + path`. On POSIX the path starts with `/` so it accidentally forms a valid `file:///...`; on Windows it starts with a drive letter, so MLflow reads `C:` as a hostname | `compat.py` PATCH 6 rewrites it to a canonical `file:///C:/...` URI, and `models.py` falls back to running without MLflow if tracking still fails |

Item 6 is the one worth knowing about: it is silent until you compute features,
and the error names pandas rather than the 360 columns that actually cause it.
