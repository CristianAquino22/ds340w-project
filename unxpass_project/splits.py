"""Stage 2 -- match-level train/test splitting.

This is the main methodological departure from the parent paper. un-xPass
builds its held-out set as "a random sample of 20% of these passes"
(Robberechts et al., 2023, Section 3.1), which places passes from the same
match on both sides of the split. Passes within a match share teams, tactics,
pitch and conditions, so that split leaks information and flatters validation
scores.

Here the split is taken over *matches*: every pass from a given match lands on
exactly one side.
"""
from __future__ import annotations

import json
import random
from typing import Dict, List, Sequence

from unxpass_project import config


def split_by_match(
    game_ids: Sequence[int],
    test_fraction: float = config.TEST_FRACTION,
    seed: int = config.SPLIT_SEED,
) -> Dict[str, List[int]]:
    """Partition game ids into train and test sets.

    Deterministic for a given ``seed`` so the split is reproducible.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be strictly between 0 and 1")
    ids = sorted(int(g) for g in game_ids)
    if len(ids) < 2:
        raise ValueError(f"need at least 2 matches to split, got {len(ids)}")
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n_test = max(1, round(len(ids) * test_fraction))
    n_test = min(n_test, len(ids) - 1)  # never leave the training set empty
    return {"test": sorted(shuffled[:n_test]), "train": sorted(shuffled[n_test:])}


def write_split(split: Dict[str, List[int]], name: str) -> None:
    """Persist a split so the exact partition can be reported and re-used."""
    path = config.RESULTS / f"split_{name}.json"
    path.write_text(json.dumps(split, indent=2))


def read_split(name: str) -> Dict[str, List[int]]:
    """Read a stored split.

    Earlier runs wrote the held-out key as "validation"; the assignment calls
    that set "test". Old files are migrated on read so existing work is not
    invalidated.
    """
    path = config.RESULTS / f"split_{name}.json"
    sp = json.loads(path.read_text())
    if "validation" in sp and "test" not in sp:
        sp["test"] = sp.pop("validation")
    return sp


def as_filters(game_ids: Sequence[int]) -> List[dict]:
    """Convert game ids into the filter format un-xPass datasets expect."""
    return [{"game_id": int(g)} for g in game_ids]
