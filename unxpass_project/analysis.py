"""Stage 7 -- does game state affect CDR? (the project's extension)

un-xPass reports that average creativity is unaffected by time remaining and
goal difference over a Premier League season. This module re-tests that on
tournament data and adds red card differential, which the parent paper never
examined and which Skripnikov, Cemek & Gillman (2026) find has a far larger
effect on offensive output than the scoreline.

Adjustment is **additive**, not multiplicative. Skripnikov et al. scale xG,
which is strictly positive, by a coefficient. CDR is a difference of expected
values and is negative for roughly half of all passes, so a multiplicative
adjustment is unstable near zero and flips sign. Shifting each game state's
distribution onto the neutral baseline keeps the metric interpretable:

    CDR_adjusted = CDR - (mean CDR in this state - mean CDR at neutral)
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from unxpass_project import compat  # noqa: F401
from unxpass_project import config, gamestate

# Winsorisation limits, following Skripnikov et al. (2026), who cap at the
# same values because states beyond them are too rare to estimate.
SCORE_CAP = 3
RED_CARD_CAP = 2

NEUTRAL = {"score_diff": 0, "red_card_diff": 0}


def load_ratings(dataset_name: str) -> pd.DataFrame:
    path = config.RESULTS / f"ratings_{dataset_name}.csv"
    return pd.read_csv(path).set_index(["game_id", "action_id"])


def with_game_state(ratings: pd.DataFrame) -> pd.DataFrame:
    """Join game state onto rated passes and winsorise the extremes."""
    games = ratings.index.get_level_values("game_id").unique()
    gs = gamestate.for_games(games)
    out = ratings.join(gs, how="left")
    out["score_diff"] = out["score_diff"].clip(-SCORE_CAP, SCORE_CAP)
    out["red_card_diff"] = out["red_card_diff"].clip(-RED_CARD_CAP, RED_CARD_CAP)
    return out


def _summary(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Mean CDR per level of ``by``, with a 95% interval on the mean."""
    g = df.groupby(by)["CDR"]
    out = g.agg(passes="count", mean_cdr="mean", std="std")
    out["se"] = out["std"] / np.sqrt(out["passes"])
    out["ci_low"] = out["mean_cdr"] - 1.96 * out["se"]
    out["ci_high"] = out["mean_cdr"] + 1.96 * out["se"]
    return out[["passes", "mean_cdr", "ci_low", "ci_high"]]


def adjust(df: pd.DataFrame) -> pd.DataFrame:
    """Add an ``CDR_adjusted`` column projecting every pass onto neutral state."""
    neutral_mask = (df["score_diff"] == NEUTRAL["score_diff"]) & (
        df["red_card_diff"] == NEUTRAL["red_card_diff"]
    )
    if not neutral_mask.any():
        raise ValueError("no passes in the neutral game state to anchor on")
    baseline = df.loc[neutral_mask, "CDR"].mean()

    state_means = df.groupby(["score_diff", "red_card_diff"])["CDR"].transform("mean")
    df = df.copy()
    df["CDR_adjusted"] = df["CDR"] - (state_means - baseline)
    return df


def report(dataset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Print the game-state analysis and return the two summary tables."""
    df = with_game_state(load_ratings(dataset_name))
    missing = df["score_diff"].isna().sum()
    if missing:
        print(f"[analysis] {missing} passes without game state; dropped")
        df = df.dropna(subset=["score_diff", "red_card_diff"])

    print(f"\n=== CDR by score differential: {dataset_name} ===")
    print("(positive = the passing team is ahead)")
    by_score = _summary(df, "score_diff")
    print(by_score.to_string(float_format=lambda v: f"{v:+.5f}"))

    print(f"\n=== CDR by red card differential ===")
    print("(positive = the passing team is short-handed)")
    by_red = _summary(df, "red_card_diff")
    print(by_red.to_string(float_format=lambda v: f"{v:+.5f}"))

    print(f"\n=== CDR by match stage ===")
    print("(a variable with no equivalent in league football)")
    by_stage = _summary(df, "knockout")
    by_stage.index = ["group stage" if not k else "knockout" for k in by_stage.index]
    print(by_stage.to_string(float_format=lambda v: f"{v:+.5f}"))

    if "stage" in df.columns and df["stage"].nunique() > 1:
        print("\nby individual stage:")
        detail = _summary(df, "stage").sort_values("passes", ascending=False)
        print(detail.to_string(float_format=lambda v: f"{v:+.5f}"))

    overall = df["CDR"].mean()
    print(f"\noverall mean CDR: {overall:+.5f}")

    # Does any state's interval exclude the overall mean?
    flagged = []
    for name, tbl in (("score", by_score), ("red cards", by_red), ("stage", by_stage)):
        for level, row in tbl.iterrows():
            if row["ci_low"] > overall or row["ci_high"] < overall:
                flagged.append((name, level, row["mean_cdr"], int(row["passes"])))
    if flagged:
        print("\nstates whose 95% interval excludes the overall mean:")
        for name, level, mean, n in flagged:
            lvl = f"{level:+g}" if isinstance(level, (int, float)) else str(level)
            print(f"  {name} = {lvl}: mean {mean:+.5f} (n={n})")
    else:
        print("\nNo game state's 95% interval excludes the overall mean --")
        print("consistent with the parent paper's null result.")

    adjusted = adjust(df)
    out = config.RESULTS / f"gamestate_{dataset_name}.csv"
    adjusted.to_csv(out)
    print(f"\nsaved per-pass game state and adjusted CDR -> {out}")
    return by_score, by_red
