"""Game state per pass: score differential, red card differential, minute.

This is the input to the project's extension. un-xPass checks whether average
creativity varies with time remaining and goal difference and reports no
effect (Robberechts et al., 2023, Section 5), but it never examines red cards
-- which Skripnikov, Cemek & Gillman (2026) identify as having a far larger
influence on offensive output than the scoreline: a team two players short has
its expected goals scaled by 3.20, where a lead of any size shifts output by
only 6-8%.

Tournament football also produces game states that league play rarely does:
knockout matches where elimination is immediate, group games needing a
specific result, and extra time. So the null result is worth re-testing here.

Score differential comes free -- `goalscore_diff` is already in the computed
feature set. Red cards are not in SPADL at all, so they are read from the
StatsBomb lineup files.
"""
from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd

from unxpass_project import compat  # noqa: F401
from unxpass_project import config, minutes

# Minute at which each period begins, on a cumulative match clock.
PERIOD_OFFSET: Dict[int, float] = {1: 0.0, 2: 45.0, 3: 90.0, 4: 105.0}

# Card types that remove a player from the pitch.
SENDING_OFF = {"Red Card", "Second Yellow"}

# Competition stages that are single-elimination. Everything else -- group
# stages, league rounds -- is not. StatsBomb spells these out in the games
# table's `competition_stage` column.
KNOCKOUT_STAGES = {
    "round of 16",
    "quarter-finals",
    "semi-finals",
    "final",
    "3rd place final",
    "play-offs",
}


def is_knockout(stage: str) -> bool:
    """Is this stage single-elimination?

    A variable with no equivalent in league football, where every match is
    worth the same three points. In a knockout match a draw at full time means
    extra time and then penalties; in a group match it may be enough to go
    through. Whether that changes how adventurously players pass is an open
    question, and one neither parent paper could have asked of league data.
    """
    return str(stage).strip().lower() in KNOCKOUT_STAGES


def match_minute(period_id: int, time_seconds: float) -> float:
    """Cumulative match minute. ``time_seconds`` is within-period in SPADL."""
    return PERIOD_OFFSET.get(int(period_id), 120.0) + float(time_seconds) / 60.0


def sendings_off(game_id: int) -> List[Dict]:
    """Every sending-off in a match: team name, player and match minute."""
    lineup = minutes._fetch_lineup(int(game_id))
    out: List[Dict] = []
    for team in lineup:
        for player in team.get("lineup", []):
            for card in player.get("cards", []) or []:
                if card.get("card_type") not in SENDING_OFF:
                    continue
                minute = minutes._parse_clock(card.get("time"))
                if minute is None:
                    continue
                out.append(
                    {
                        "game_id": int(game_id),
                        "team_name": team["team_name"],
                        "player_id": int(player["player_id"]),
                        "minute": minute,
                        "card_type": card["card_type"],
                    }
                )
    return out


def for_games(game_ids: Iterable[int]) -> pd.DataFrame:
    """Game state for every action in the given matches.

    Returns a frame indexed by (game_id, action_id) with:

    ``minute``
        cumulative match minute of the action
    ``score_diff``
        goals scored by the acting team minus goals conceded, before the action
    ``red_card_diff``
        own players sent off minus opponents sent off, *before* the action.
        Positive means the acting team is short-handed, matching the sign
        convention in Skripnikov et al. (2026).
    ``stage``
        the competition stage, as StatsBomb records it
    ``knockout``
        True for single-elimination matches
    """
    from socceraction.spadl.utils import add_names
    from unxpass.databases import connect

    db = connect(config.db_uri(), mode="r")
    frames = []
    try:
        teams = pd.read_sql("SELECT * FROM teams", db.conn)
        name_to_id = dict(zip(teams["team_name"], teams["team_id"]))
        games = db.games()
        for gid in game_ids:
            gid = int(gid)
            actions = add_names(db.actions(gid))
            reds = sendings_off(gid)

            minute = [
                match_minute(p, t)
                for p, t in zip(actions["period_id"], actions["time_seconds"])
            ]
            df = pd.DataFrame({"minute": minute}, index=actions.index)
            df["team_id"] = actions["team_id"].values

            # Score before each action, from the same rule socceraction uses.
            goals = actions["type_name"].str.contains("shot", na=False) & (
                actions["result_name"] == "success"
            )
            owngoals = actions["type_name"].str.contains("shot", na=False) & (
                actions["result_name"] == "owngoal"
            )
            sides = actions["team_id"].unique()
            team_a = sides[0]
            is_a = actions["team_id"] == team_a
            goals_a = (goals & is_a) | (owngoals & ~is_a)
            goals_b = (goals & ~is_a) | (owngoals & is_a)
            score_a = goals_a.cumsum() - goals_a
            score_b = goals_b.cumsum() - goals_b
            df["score_diff"] = (score_a - score_b).where(is_a, score_b - score_a).values

            # Red cards before each action.
            own, opp = [0] * len(df), [0] * len(df)
            if reds:
                for r in reds:
                    red_team = name_to_id.get(r["team_name"])
                    if red_team is None:
                        continue
                    after = df["minute"] > r["minute"]
                    is_own = df["team_id"] == red_team
                    own = own + (after & is_own).astype(int)
                    opp = opp + (after & ~is_own).astype(int)
            df["red_card_diff"] = pd.Series(own, index=df.index) - pd.Series(
                opp, index=df.index
            )
            stage = games.loc[gid, "competition_stage"] if gid in games.index else None
            df["stage"] = stage
            df["knockout"] = is_knockout(stage)
            frames.append(
                df[["minute", "score_diff", "red_card_diff", "stage", "knockout"]]
            )
    finally:
        db.close()
    return pd.concat(frames) if frames else pd.DataFrame()
