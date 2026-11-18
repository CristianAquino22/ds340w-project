"""Stage 3 -- compute features and labels for a set of matches.

Feature computation is the slowest non-training stage (order 1-2 minutes per
match for the full set of generators), and it is embarrassingly parallel over
matches, so this module can fan out across processes.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Sequence

from unxpass_project import compat  # noqa: F401
from unxpass_project import config, splits

from unxpass.databases import connect
from unxpass.datasets import PassesDataset


def build(
    name: str,
    game_ids: Sequence[int],
    xfns: Optional[List[str]] = None,
    yfns: Optional[List[str]] = None,
) -> Path:
    """Compute and cache features/labels for ``game_ids`` under ``name``.

    Parameters
    ----------
    name
        Dataset name; becomes a directory under ``stores/datasets``.
    game_ids
        Matches to include.
    xfns, yfns
        Feature/label generator names. ``None`` means "all of them", which is
        what the parent paper's components collectively require.
    """
    from unxpass.features import all_features
    from unxpass.labels import all_labels

    out = config.DATASETS / name
    db = connect(config.db_uri(), mode="r")
    try:
        ds = PassesDataset(
            path=out,
            xfns=all_features if xfns is None else xfns,
            yfns=all_labels if yfns is None else yfns,
            load_cached=False,
        )
        t0 = time.time()
        ds.create(db, splits.as_filters(game_ids))
        elapsed = time.time() - t0
        n = len(ds.features) if ds.features is not None else 0
        print(
            f"[featurize] {name}: {len(game_ids)} matches, {n} passes, "
            f"{elapsed:.0f}s ({elapsed / max(len(game_ids), 1):.0f}s/match)"
        )
        return out
    finally:
        db.close()


def load(name: str):
    """Re-open a cached dataset."""
    from unxpass.features import all_features
    from unxpass.labels import all_labels

    return PassesDataset(
        path=config.DATASETS / name,
        xfns=all_features,
        yfns=all_labels,
        load_cached=True,
    )


def pool(name: str, sources: Sequence[str]) -> Path:
    """Combine cached datasets into one, without recomputing features.

    Feature computation is the most expensive stage, and its output is cached
    per dataset as parquet files keyed on (game_id, action_id). Competitions
    never share game ids, so pooling is a concatenation -- no need to redo work
    already done for a competition that is being added to.
    """
    import pandas as pd

    out = config.DATASETS / name
    out.mkdir(parents=True, exist_ok=True)
    src_paths = [config.DATASETS / s for s in sources]
    missing = [str(p) for p in src_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"cannot pool, dataset(s) not built: {missing}")

    files = sorted({f.name for p in src_paths for f in p.glob("*.parquet")})
    n_rows = None
    for fname in files:
        parts = []
        for p in src_paths:
            fp = p / fname
            if fp.exists():
                parts.append(pd.read_parquet(fp))
            else:
                print(f"[pool] warning: {fname} missing from {p.name}, skipped")
        if not parts:
            continue
        merged = pd.concat(parts)
        merged.to_parquet(out / fname)
        if fname.startswith("y_") and n_rows is None:
            n_rows = len(merged)
    print(f"[pool] {name}: {len(sources)} sources -> {n_rows} passes at {out}")
    return out


def count_passes(name: str) -> int:
    """Number of passes in a cached dataset, without loading features."""
    import pandas as pd

    path = config.DATASETS / name
    for f in sorted(path.glob("y_*.parquet")):
        return len(pd.read_parquet(f))
    for f in sorted(path.glob("x_*.parquet")):
        return len(pd.read_parquet(f))
    raise FileNotFoundError(f"no cached dataset at {path}")
