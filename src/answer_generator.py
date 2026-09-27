"""Human-readable answers grounded in executed query results."""

from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd

from src.ai_provider import ChatProvider, create_chat_provider
from src.config import AIConfig, ConfigurationError
from src.query_engine import QueryResult


INSUFFICIENT_DATA_ANSWER = "The available data is insufficient to answer this question reliably."


def _format_value(value: Any) -> str:
    if value is None or value is pd.NA or value is pd.NaT:
        return "NULL"
    if isinstance(value, float) and math.isnan(value):
        return "NULL"
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _deterministic_answer(frame: pd.DataFrame, *, truncated: bool = False) -> str:
    if frame.empty:
        return INSUFFICIENT_DATA_ANSWER
    first = frame.iloc[0]
    facts = [f"{column} = {_format_value(value)}" for column, value in first.items()]
    if len(frame) == 1:
        return "The query result is " + "; ".join(facts) + "."
    qualifier = "At least " if truncated else ""
    suffix = " (the displayed results are capped)." if truncated else "."
    return (
        f"The query returned {qualifier}{len(frame)} rows. "
        f"The first returned result is " + "; ".join(facts) + suffix
    )


def generate_answer(
    question: str,
    result: QueryResult | pd.DataFrame,
    *,
    provider: ChatProvider | None = None,
) -> str:
    """Explain query output; deterministic wording uses only actual result values."""
    if isinstance(result, QueryResult):
        if result.status != "success":
            return f"The query could not be completed: {result.error or 'an execution error occurred.'}"
        frame = result.dataframe
        truncated = result.truncated
        sql = result.sql
    elif isinstance(result, pd.DataFrame):
        frame = result
        truncated = False
        sql = None
    else:
        return INSUFFICIENT_DATA_ANSWER

    if frame.empty:
        return INSUFFICIENT_DATA_ANSWER
    fallback = _deterministic_answer(frame, truncated=truncated)
    if provider is None:
        try:
            provider = create_chat_provider(AIConfig.from_env())
        except ConfigurationError:
            provider = None
    if provider is None:
        return fallback

    records = json.loads(frame.head(20).to_json(orient="records", date_format="iso"))
    payload = {
        "question": question,
        "sql": sql,
        "result_rows": records,
        "displayed_row_count": len(frame),
        "results_truncated": truncated,
    }
    system_prompt = (
        "Explain analytical query results in concise business language. Use only values and facts "
        "present in the supplied result rows. Do not invent totals, causes, context, or trends. "
        "If the result does not support a reliable answer, say so explicitly."
    )
    user_prompt = "Answer using this executed result JSON only:\n" + json.dumps(payload, ensure_ascii=True)
    try:
        answer = provider.complete(system_prompt, user_prompt).strip()
    except Exception:
        return fallback
    return answer or fallback