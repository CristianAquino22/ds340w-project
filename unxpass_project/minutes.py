"""Minutes played per player per match, from StatsBomb lineups.

un-xPass imports competitions, games, teams, players and actions, but not
lineups, so minutes played are not in the project database. CDR per 90 -- the
normalisation the parent paper reports -- needs them, so this module fetches
the lineup files and caches them.

A player's time on the pitch is the union of their position spells. StatsBomb
records each spell with ``from``/``to`` on a cumulative match clock (the second
half starts at 45:00, extra time at 90:00), so the strings parse directly into
match minutes. A null ``to`` means the player was still on when the match
ended.
"""
from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional

import pandas as pd

from unxpass_project import config

LINEUP_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/lineups/{game_id}.json"
CACHE = config.STORES / "lineups"
CACHE.mkdir(parents=True, exist_ok=True)

# Periods that count as playing time. 5 is the penalty shootout.
PLAYING_PERIODS = {1, 2, 3, 4}


def _parse_clock(value: Optional[str]) -> Optional[float]:
    """'68:51' -> 68.85 minutes. Returns None for a null/empty value."""
    if not value:
        return None
    parts = value.split(":")
    try:
        if len(parts) == 2:
            mm, ss = parts
            return int(mm) + int(ss) / 60.0
        if len(parts) == 3:
            hh, mm, ss = parts
            return int(hh) * 60 + int(mm) + int(ss) / 60.0
    except ValueError:
        return None
    return None


def _fetch_lineup(game_id: int) -> list:
    """Fetch one lineup file, caching it under stores/lineups."""
    path = CACHE / f"{game_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    import urllib.request

    with urllib.request.urlopen(LINEUP_URL.format(game_id=game_id), timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    path.write_text(json.dumps(payload))
    return payload


def _match_end(lineup: list, fallback: float = 90.0) -> float:
    """Latest recorded ``to`` in the match, used for players still on at the end."""
    ends = [
        _parse_clock(pos.get("to"))
        for team in lineup
        for player in team.get("lineup", [])
        for pos in player.get("positions", [])
        if pos.get("to_period") in PLAYING_PERIODS
    ]
    ends = [e for e in ends if e is not None]
    return max(ends) if ends else fallback


def minutes_for_game(game_id: int) -> pd.DataFrame:
    """Minutes played by each player in one match."""
    lineup = _fetch_lineup(int(game_id))
    end = _match_end(lineup)
    rows: List[Dict] = []
    for team in lineup:
        for player in team.get("lineup", []):
            total = 0.0
            for pos in player.get("positions", []):
                if pos.get("from_period") not in PLAYING_PERIODS:
                    continue
                start = _parse_clock(pos.get("from"))
                if start is None:
                    continue
                stop = _parse_clock(pos.get("to"))
                if stop is None:  # still on when the match ended
                    stop = end
                total += max(0.0, stop - start)
            if total > 0:
                rows.append(
                    {
                        "game_id": int(game_id),
                        "player_id": int(player["player_id"]),
                        "minutes": round(total, 2),
                    }
                )
    return pd.DataFrame(rows)


def minutes_for_games(game_ids: Iterable[int]) -> pd.DataFrame:
    """Minutes played per player, summed across matches."""
    frames = [minutes_for_game(int(g)) for g in game_ids]
    if not frames:
        return pd.DataFrame(columns=["player_id", "minutes", "matches"])
    allm = pd.concat(frames, ignore_index=True)
    return (
        allm.groupby("player_id")
        .agg(minutes=("minutes", "sum"), matches=("game_id", "nunique"))
        .sort_values("minutes", ascending=False)
    )
