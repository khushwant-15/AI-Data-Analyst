"""SQLite persistence for uploaded analytical datasets."""

from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class DatabaseError(ValueError):
    """Raised when a dataset cannot be stored in SQLite."""


def _validate_table_name(identifier: str) -> None:
    if not _IDENTIFIER.fullmatch(identifier):
        raise DatabaseError("Database table names must contain only letters, numbers, and underscores.")


def create_database(
    frame: pd.DataFrame,
    db_path: str | Path,
    *,
    table_name: str = "dataset",
) -> Path:
    """Create a SQLite database containing a copy of the supplied DataFrame."""
    if frame.empty or len(frame.columns) == 0:
        raise DatabaseError("A dataset must contain at least one row and one column.")
    if not frame.columns.is_unique:
        raise DatabaseError("Column names must be unique before storing the dataset.")
    _validate_table_name(table_name)
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with closing(sqlite3.connect(path)) as connection:
            with connection:
                frame.to_sql(table_name, connection, if_exists="replace", index=False)
    except (sqlite3.Error, ValueError, TypeError) as exc:
        raise DatabaseError("The dataset could not be stored in the local analytical database.") from exc
    return path