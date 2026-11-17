"""Stage 1 -- download StatsBomb 360 open data and convert it to SPADL."""
from __future__ import annotations

from typing import List, Optional

from unxpass_project import compat  # noqa: F401  -- must precede unxpass imports
from unxpass_project import config

from unxpass.databases import connect


def download(competition: str, limit: Optional[int] = None) -> List[int]:
    """Download one competition into the project database.

    Parameters
    ----------
    competition
        Key into :data:`config.COMPETITIONS`, e.g. ``"euro2020"``.
    limit
        If given, download only the first ``limit`` matches. Useful for a
        smoke test before committing to a full competition.

    Returns
    -------
    The game ids now present for that competition.
    """
    spec = config.COMPETITIONS[competition]
    db = connect(config.db_uri(), mode="a")
    try:
        if limit is None:
            db.import_data(
                getter="remote",
                competition_id=spec["competition_id"],
                season_id=spec["season_id"],
            )
        else:
            # import_data has no limit; fetch the match list first, then
            # import game by game so a smoke test stays cheap.
            from statsbombpy import sb

            matches = sb.matches(
                competition_id=spec["competition_id"], season_id=spec["season_id"]
            )
            for gid in matches["match_id"].tolist()[:limit]:
                db.import_data(
                    getter="remote",
                    competition_id=spec["competition_id"],
                    season_id=spec["season_id"],
                    game_id=int(gid),
                )
        games = db.games(
            competition_id=spec["competition_id"], season_id=spec["season_id"]
        )
        return games.index.tolist()
    finally:
        db.close()


def game_ids(competition: str) -> List[int]:
    """Game ids already in the database for a competition."""
    spec = config.COMPETITIONS[competition]
    db = connect(config.db_uri(), mode="r")
    try:
        games = db.games(
            competition_id=spec["competition_id"], season_id=spec["season_id"]
        )
        return games.index.tolist()
    finally:
        db.close()
