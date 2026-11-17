"""Stage 5 -- compute CDR and aggregate it per player."""
from __future__ import annotations

import time

import pandas as pd

from unxpass_project import compat  # noqa: F401
from unxpass_project import config, models

from unxpass.databases import connect
from unxpass.ratings import CreativeDecisionRating


def rate(dataset_name: str, out_name: str | None = None) -> pd.DataFrame:
    """Compute per-pass CDR for a cached dataset.

    CDR(D = l | S) = E[U(D = l | S)] - E[U(D = l_typical | S)]

    i.e. the expected value of the pass actually played minus that of the pass
    a typical player would have played from the same game state. It can be
    negative: choosing worse than the obvious option scores below zero.
    """
    out_name = out_name or dataset_name
    db = connect(config.db_uri(), mode="r")
    try:
        rater = CreativeDecisionRating(
            pass_selection_component=models.load("pass_selection"),
            pass_success_component=models.load("pass_success"),
            pass_value_component=models.load("pass_value"),
        )
        t0 = time.time()
        df = rater.rate(db, models.dataset(dataset_name))
        print(f"[rate] {len(df)} passes in {time.time() - t0:.0f}s")
    finally:
        db.close()

    path = config.RESULTS / f"ratings_{out_name}.csv"
    df.reset_index().to_csv(path, index=False)
    print(f"[rate] saved -> {path}")
    return df


def player_rankings(
    ratings: pd.DataFrame,
    min_minutes: float = 180.0,
    min_passes: int = 0,
) -> pd.DataFrame:
    """Aggregate per-pass CDR into a per-player table.

    Reports two normalisations, because they answer different questions:

    ``cdr_per_90``
        What the parent paper reports. Comparable to its figures, but it
        conflates creativity with volume: a deep-lying playmaker making 80
        passes a match accumulates more than a winger making 25 at identical
        creativity per pass.
    ``cdr_per_pass``
        Isolates the quality of each decision from how often the player is on
        the ball.

    Neither normalisation reduces sampling noise -- that is what
    ``min_minutes`` is for. Per-90 is in fact noisiest for low-minute players,
    since a single good ball from a 20-minute substitute is divided by a small
    number.
    """
    from unxpass_project import minutes as minutes_mod

    db = connect(config.db_uri(), mode="r")
    try:
        games = ratings.index.get_level_values("game_id").unique()
        actions = pd.concat([db.actions(int(g)) for g in games])
        try:
            players = pd.read_sql("SELECT * FROM players", db.conn)
        except Exception:
            players = pd.DataFrame()
    finally:
        db.close()

    joined = ratings.join(actions[["player_id"]], how="left")
    agg = (
        joined.groupby("player_id")["CDR"]
        .agg(["count", "mean", "sum"])
        .rename(columns={"count": "passes", "mean": "cdr_per_pass", "sum": "cdr_total"})
    )

    mins = minutes_mod.minutes_for_games(games)
    agg = agg.join(mins, how="left")
    agg["minutes"] = agg["minutes"].fillna(0.0)
    agg["cdr_per_90"] = agg["cdr_total"] / agg["minutes"].replace(0, pd.NA) * 90
    agg["passes_per_90"] = agg["passes"] / agg["minutes"].replace(0, pd.NA) * 90

    agg = agg[(agg["minutes"] >= min_minutes) & (agg["passes"] >= min_passes)]

    if len(players):
        name_col = next(
            (c for c in ("player_name", "nickname", "name") if c in players.columns), None
        )
        if name_col is not None and "player_id" in players.columns:
            names = (
                players[["player_id", name_col]]
                .dropna(subset=["player_id"])
                .drop_duplicates("player_id")
                .set_index("player_id")
            )
            agg = agg.join(names, how="left")

    cols = [c for c in ("player_name", "matches", "minutes", "passes", "passes_per_90",
                        "cdr_per_pass", "cdr_per_90", "cdr_total") if c in agg.columns]
    return agg[cols].sort_values("cdr_per_90", ascending=False)
