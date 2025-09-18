import json
import os
import zipfile
from collections import Counter
from typing import Dict
import shutil
import sqlite3, json, os
import pandas as pd
from pathlib import Path



# Flat table
def get_all_routes_with_tokens(board_id:int, conn=None, limit: int | None = None):
    base_sql = """
    SELECT
      b.id    AS route_id,
      b.difficulty,
      h.value AS hold_id,
      t.token AS hold_token,
      CAST(SUBSTR(t.token, 2) AS INTEGER) AS token_num,

      -- add token type directly in SQL
      CASE
        WHEN t.token LIKE 'p%' THEN 'placement'
        WHEN t.token LIKE 'r%' THEN 'role'
        ELSE 'unknown'
      END AS token_type,

      -- placement info (only if token starts with 'p')
      p.x, p.y, p.set_id,

      -- role info (only if token starts with 'r')
      r.name  AS role_name,
      r.color AS role_color

    FROM board_samples b
    JOIN json_each(b.holds) h
    LEFT JOIN static.id_to_token t
      ON t.id = h.value
    LEFT JOIN static.placements p
      ON p.token_id = CAST(SUBSTR(t.token, 2) AS INTEGER)
     AND t.token LIKE 'p%'
    LEFT JOIN static.roles r
      ON r.role_id = CAST(SUBSTR(t.token, 2) AS INTEGER)
     AND t.token LIKE 'r%'
    WHERE b.board_id = ?
    """
    if limit is None:
        sql = base_sql  # no LIMIT
        params = [board_id]
    else:
        sql = base_sql + " LIMIT ?"
        params = [board_id, limit]
    return pd.read_sql(sql, conn, params=params)



def get_data(board_name: str, difficulty: int | None = None, limit: int | None = None) -> Dict[str, pd.DataFrame]:
    """
    Load placements and roles for a given board.

    Args:
        board_name: The name of the board (as stored in static.token_to_id).
        difficulty: Optional filter for route difficulty.
        limit: Optional row limit (None = no limit).

    Returns:
        A dict with:
          - 'placements': DataFrame of placement tokens with coordinates
          - 'roles': DataFrame of role tokens with role info
    """
    # Get the path to the current file (e.g. kilter/)
    base_dir = Path(__file__).resolve().parent.parent

    # Data folder inside kilter/
    data_dir = base_dir / "data"

    boards_db = data_dir / "boards.db"
    static_db = data_dir / "static.db"
    
    #print(f"Loading data for board '{board_name}' from {boards_db} and {static_db}")

    with sqlite3.connect(boards_db) as conn:
        # attach static db
        conn.execute(f"ATTACH DATABASE '{static_db}' AS static;")

        # get board_id from name
        board_id_df = pd.read_sql(
            "SELECT id FROM static.token_to_id WHERE token = ?",
            conn,
            params=[board_name]
        )
        if board_id_df.empty:
            raise ValueError(f"Board '{board_name}' not found in token_to_id")
        board_id = int(board_id_df.iloc[0]["id"])

        # query routes with placements + roles
        df = get_all_routes_with_tokens(board_id=board_id, conn=conn, limit=limit)

        if difficulty is not None:
            df = df[df.difficulty == difficulty]

        placements = df[df.token_type == "placement"].reset_index(drop=True)
        roles      = df[df.token_type == "role"].reset_index(drop=True)

    return {"placements": placements, "roles": roles}













