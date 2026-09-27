"""Conservative validation for read-only analytical SQLite queries."""

from __future__ import annotations

import re


class SQLValidationError(ValueError):
    """Raised when SQL is not a single safe read-only query."""


_BLOCKED = {
    "ALTER", "ATTACH", "CREATE", "DELETE", "DETACH", "DROP", "INSERT", "REPLACE",
    "UPDATE", "VACUUM", "REINDEX", "ANALYZE", "PRAGMA", "TRUNCATE", "BEGIN",
    "COMMIT", "ROLLBACK", "SAVEPOINT", "RELEASE",
}
_BLOCKED_FUNCTIONS = {"LOAD_EXTENSION", "WRITEFILE", "READFILE"}
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*")


def _tokens(sql: str) -> list[tuple[str, str]]:
    """Lex enough SQL to ignore strings/comments while detecting statements."""
    result: list[tuple[str, str]] = []
    index = 0
    length = len(sql)
    while index < length:
        char = sql[index]
        if char.isspace():
            index += 1
        elif sql.startswith("--", index):
            end = sql.find("\n", index + 2)
            index = length if end < 0 else end + 1
        elif sql.startswith("/*", index):
            end = sql.find("*/", index + 2)
            if end < 0:
                raise SQLValidationError("SQL contains an unterminated block comment.")
            index = end + 2
        elif char in "'\"`":
            quote = char
            index += 1
            while index < length:
                if sql[index] == quote:
                    if index + 1 < length and sql[index + 1] == quote:
                        index += 2
                        continue
                    index += 1
                    break
                index += 1
            else:
                raise SQLValidationError("SQL contains an unterminated quoted value or identifier.")
            result.append(("QUOTED", ""))
        elif char == "[":
            end = sql.find("]", index + 1)
            if end < 0:
                raise SQLValidationError("SQL contains an unterminated quoted identifier.")
            result.append(("QUOTED", ""))
            index = end + 1
        elif char == ";":
            result.append(("SEMICOLON", char))
            index += 1
        else:
            match = _WORD.match(sql, index)
            if match:
                result.append(("WORD", match.group(0).upper()))
                index = match.end()
            else:
                result.append(("SYMBOL", char))
                index += 1
    return result


def validate_sql(sql: str) -> str:
    """Return normalized SQL or raise if it is not one read-only SELECT/CTE.

    SQLite's authorizer must still be enabled at execution time; this validator
    is a first line of defense, not a substitute for database-level controls.
    """
    if not isinstance(sql, str) or not sql.strip():
        raise SQLValidationError("Enter a SQL query to analyze the dataset.")
    tokens = _tokens(sql)
    meaningful = [token for token in tokens if token[0] != "SEMICOLON"]
    if not meaningful or meaningful[0] != ("WORD", "SELECT") and meaningful[0] != ("WORD", "WITH"):
        raise SQLValidationError("Only SELECT queries and read-only CTE queries are allowed.")

    semicolons = [index for index, token in enumerate(tokens) if token[0] == "SEMICOLON"]
    if semicolons and (len(semicolons) > 1 or semicolons[0] != len(tokens) - 1):
        raise SQLValidationError("Only one SQL statement is allowed.")

    words = [value for kind, value in tokens if kind == "WORD"]
    blocked = _BLOCKED.intersection(words)
    if blocked:
        raise SQLValidationError(f"Unsafe SQL operation is not allowed: {sorted(blocked)[0]}.")
    if any(word in _BLOCKED_FUNCTIONS for word in words):
        raise SQLValidationError("File access and extension-loading SQL functions are not allowed.")
    if "SELECT" not in words:
        raise SQLValidationError("The query must contain a SELECT statement.")
    return sql.strip().rstrip(";").strip()


def is_safe_sql(sql: str) -> bool:
    """Return whether the SQL passes the conservative read-only checks."""
    try:
        validate_sql(sql)
        return True
    except SQLValidationError:
        return False