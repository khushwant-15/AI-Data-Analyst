"""Read-only, resource-bounded SQLite query execution."""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd

from src.database import _IDENTIFIER, DatabaseError
from src.sql_validator import SQLValidationError, validate_sql


@dataclass
class QueryResult:
    sql: str
    dataframe: pd.DataFrame
    row_count: int
    status: str
    error: str | None = None
    truncated: bool = False


def _authorizer(table_name: str):
    allowed_actions = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}

    def authorize(action: int, arg1: str | None, arg2: str | None, database: str | None, source: str | None) -> int:
        if action in allowed_actions:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ and arg1 == table_name and database == "main":
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    return authorize


def execute_query(
    db_path: str | Path,
    sql: str,
    *,
    table_name: str = "dataset",
    max_rows: int = 5000,
    timeout_seconds: float = 10.0,
    max_vm_steps: int = 5_000_000,
) -> QueryResult:
    """Execute one validated query against a read-only database connection."""
    try:
        normalized_sql = validate_sql(sql)
        if not _IDENTIFIER.fullmatch(table_name):
            raise DatabaseError("Invalid analytical table name.")
        if max_rows < 1:
            raise ValueError("max_rows must be at least 1.")
        path = Path(db_path).resolve()
        if not path.is_file():
            raise FileNotFoundError("The analytical database does not exist.")
        uri = f"file:{quote(path.as_posix(), safe='/')}?mode=ro"
        deadline = time.monotonic() + timeout_seconds
        steps = 0

        def progress() -> int:
            nonlocal steps
            steps += 1000
            return int(steps > max_vm_steps or time.monotonic() > deadline)

        with closing(sqlite3.connect(uri, uri=True, timeout=timeout_seconds)) as connection:
            connection.set_authorizer(_authorizer(table_name))
            connection.set_progress_handler(progress, 1000)
            cursor = connection.execute(normalized_sql)
            names = [description[0] for description in cursor.description or []]
            rows = cursor.fetchmany(max_rows + 1)
            truncated = len(rows) > max_rows
            rows = rows[:max_rows]
            frame = pd.DataFrame.from_records(rows, columns=names)
        return QueryResult(
            sql=normalized_sql,
            dataframe=frame,
            row_count=len(frame),
            status="success",
            truncated=truncated,
        )
    except (SQLValidationError, DatabaseError, sqlite3.Error, OSError, ValueError) as exc:
        return QueryResult(
            sql=sql,
            dataframe=pd.DataFrame(),
            row_count=0,
            status="error",
            error=str(exc),
        )