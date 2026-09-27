"""Schema-aware natural-language SQL generation with a deterministic fallback."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from src.ai_provider import ChatProvider, create_chat_provider
from src.config import AIConfig, ConfigurationError
from src.sql_validator import SQLValidationError, validate_sql


class UnsupportedQuestionError(ValueError):
    """Raised when deterministic rules cannot interpret a question safely."""


@dataclass(frozen=True)
class ColumnSchema:
    name: str
    data_type: str
    examples: tuple[str, ...] = ()


@dataclass(frozen=True)
class DatasetSchema:
    table_name: str
    columns: tuple[ColumnSchema, ...]


@dataclass(frozen=True)
class SQLGenerationResult:
    sql: str | None
    source: str
    error: str | None = None
    warning: str | None = None

    @property
    def success(self) -> bool:
        return self.sql is not None and self.error is None


_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_GENERIC_NAME_WORDS = {"id", "key", "name", "number", "num", "code", "description", "value"}
_DIMENSION_ALIASES = {
    "product": {"product", "item", "sku"},
    "customer": {"customer", "client", "account"},
    "city": {"city", "town"},
    "region": {"region", "territory", "state", "area"},
    "category": {"category", "segment", "type"},
}
_ENTITY_ALIASES = {
    "customer": {"customer", "client", "account"},
    "order": {"order", "transaction", "invoice"},
    "product": {"product", "item", "sku"},
}
_MEASURE_ALIASES = {
    "sales": {"sales", "sale", "revenue", "turnover", "income"},
    "profit": {"profit", "earnings", "gain"},
    "cost": {"cost", "expense", "spend"},
    "quantity": {"quantity", "unit", "units", "volume"},
    "margin": {"margin"},
}


def _words(value: str) -> list[str]:
    words = [part.casefold() for part in _WORD.findall(value.replace("_", " "))]
    normalized = []
    for word in words:
        if len(word) > 4 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        normalized.append(word)
    return normalized


def _is_date_column(name: str, data_type: str) -> bool:
    tokens = set(_words(name))
    return "date" in data_type.casefold() or "time" in data_type.casefold() or bool(tokens & {"date", "time", "timestamp"})


def _infer_column_type(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return "numeric"

    name_suggests_date = any(token in str(series.name).casefold() for token in ("date", "time", "timestamp"))
    if name_suggests_date:
        return "date"
    sample = series.dropna().head(20)
    if len(sample) >= 2:
        values = sample.astype(str)
        date_shaped = values.str.contains(r"\d{1,4}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}", regex=True).mean()
        if date_shaped >= 0.6:
            parsed = pd.to_datetime(values, errors="coerce", format="mixed")
            if parsed.notna().mean() >= 0.6:
                return "date"
    return "categorical"


def build_schema(frame: pd.DataFrame, *, table_name: str = "dataset") -> DatasetSchema:
    """Build a compact schema from a DataFrame for rules and model prompts."""
    if not _SAFE_IDENTIFIER.fullmatch(table_name):
        raise ValueError("The analytical table name must be a simple SQL identifier.")
    columns = []
    for name in frame.columns:
        series = frame[name]
        examples = tuple(str(value)[:100] for value in series.dropna().head(3).tolist())
        columns.append(ColumnSchema(str(name), _infer_column_type(series), examples))
    return DatasetSchema(table_name, tuple(columns))


def _coerce_schema(schema: DatasetSchema | pd.DataFrame | Mapping[str, Any]) -> DatasetSchema:
    if isinstance(schema, DatasetSchema):
        return schema
    if isinstance(schema, pd.DataFrame):
        return build_schema(schema)
    if isinstance(schema, Mapping):
        table_name = str(schema.get("table_name", "dataset"))
        raw_columns = schema.get("columns", ())
        columns = []
        if isinstance(raw_columns, Mapping):
            raw_columns = [{"name": name, "data_type": dtype} for name, dtype in raw_columns.items()]
        for item in raw_columns:
            if isinstance(item, ColumnSchema):
                columns.append(item)
            elif isinstance(item, Mapping):
                columns.append(
                    ColumnSchema(
                        name=str(item.get("name", item.get("column", ""))),
                        data_type=str(item.get("data_type", item.get("dtype", "categorical"))),
                        examples=tuple(str(value) for value in item.get("examples", ())),
                    )
                )
            else:
                columns.append(ColumnSchema(str(item), "categorical"))
        if not _SAFE_IDENTIFIER.fullmatch(table_name):
            raise ValueError("The analytical table name must be a simple SQL identifier.")
        return DatasetSchema(table_name, tuple(column for column in columns if column.name))
    raise TypeError("schema must be a DatasetSchema, DataFrame, or schema mapping.")


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _column_slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").casefold() or "measure"


def _column_tokens(column: ColumnSchema) -> list[str]:
    return [token for token in _words(column.name) if token not in _GENERIC_NAME_WORDS]


def _column_score(column: ColumnSchema, question: str, question_tokens: set[str]) -> float:
    phrase = " ".join(_column_tokens(column))
    normalized_question = " ".join(_words(question))
    if phrase and re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", normalized_question):
        return 100.0 + len(phrase)
    tokens = set(_column_tokens(column))
    if not tokens:
        return 0.0
    overlap = len(tokens & question_tokens)
    return overlap / len(tokens) * 10.0


def _is_numeric(column: ColumnSchema) -> bool:
    kind = column.data_type.casefold()
    return any(token in kind for token in ("numeric", "integer", "int", "float", "double", "decimal", "real"))


def _best_explicit_column(columns: Sequence[ColumnSchema], question: str) -> ColumnSchema | None:
    question_tokens = set(_words(question))
    ranked = sorted(
        (( _column_score(column, question, question_tokens), column) for column in columns),
        key=lambda item: item[0],
        reverse=True,
    )
    if not ranked or ranked[0][0] <= 0:
        return None
    if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
        return None
    return ranked[0][1]


def _pick_measure(question: str, numeric_columns: Sequence[ColumnSchema]) -> ColumnSchema | None:
    direct = _best_explicit_column(numeric_columns, question)
    if direct is not None:
        return direct
    question_tokens = set(_words(question))
    alias_matches = []
    for column in numeric_columns:
        column_tokens = set(_column_tokens(column))
        if any(question_tokens & aliases and column_tokens & aliases for aliases in _MEASURE_ALIASES.values()):
            alias_matches.append(column)
    return alias_matches[0] if len(alias_matches) == 1 else None


def _pick_entity(question: str, columns: Sequence[ColumnSchema]) -> ColumnSchema | None:
    question_tokens = set(_words(question))
    matching_aliases = [aliases for aliases in _ENTITY_ALIASES.values() if question_tokens & aliases]
    if len(matching_aliases) != 1:
        return None
    aliases = matching_aliases[0]
    candidates = [column for column in columns if set(_column_tokens(column)) & aliases]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda column: (
            0 if set(_words(column.name)) & {"id", "key", "code"} else 1,
            column.name.casefold(),
        ),
    )[0]


def _pick_dimensions(question: str, columns: Sequence[ColumnSchema], excluded: set[str] | None = None) -> list[ColumnSchema]:
    excluded = excluded or set()
    question_tokens = set(_words(question))
    categorical = [
        column for column in columns
        if not _is_numeric(column) and not _is_date_column(column.name, column.data_type)
        and column.name not in excluded
    ]
    selected = [column for column in categorical if _column_score(column, question, question_tokens) > 0]
    for label, aliases in _DIMENSION_ALIASES.items():
        if question_tokens & aliases:
            for column in categorical:
                if set(_column_tokens(column)) & aliases and column not in selected:
                    selected.append(column)
    return selected


def rule_based_sql(question: str, schema: DatasetSchema | pd.DataFrame | Mapping[str, Any]) -> str:
    """Generate SQL for supported analytical patterns without using a model."""
    schema = _coerce_schema(schema)
    question = question.strip()
    if not question:
        raise UnsupportedQuestionError("Enter a question about the dataset.")

    tokens = set(_words(question))
    columns = list(schema.columns)
    numeric = [column for column in columns if _is_numeric(column)]
    dates = [column for column in columns if _is_date_column(column.name, column.data_type)]
    count_intent = "count" in tokens or "how many" in question.casefold() or "number" in tokens
    ranking = bool(tokens & {"highest", "most", "top", "largest", "lowest", "least", "smallest", "best"})
    time_intent = bool(tokens & {"month", "monthly", "year", "yearly", "trend", "trends", "time"}) or "over time" in question.casefold()
    group_intent = bool(tokens & {"by", "per", "each", "distribution", "breakdown", "percentage", "share"})

    dimensions: list[ColumnSchema] = []
    entity = _pick_entity(question, columns) if count_intent else None
    if count_intent and entity is None:
        raise UnsupportedQuestionError("The requested entity is not identifiable from the dataset columns.")

    if time_intent:
        date_column = _best_explicit_column(dates, question) if dates else None
        if date_column is None and len(dates) == 1:
            date_column = dates[0]
        if date_column is None:
            raise UnsupportedQuestionError("A date or time column is required for a time-trend question.")
        dimensions.append(date_column)

    if not count_intent or group_intent:
        excluded = {entity.name} if entity else set()
        for column in _pick_dimensions(question, columns, excluded):
            if column not in dimensions:
                dimensions.append(column)

    if ranking and not dimensions:
        raise UnsupportedQuestionError("Name a category, product, customer, or time period to rank.")

    measure = _pick_measure(question, numeric)
    if not count_intent and measure is None:
        raise UnsupportedQuestionError("The question does not identify a numeric measure in the dataset.")
    if not count_intent and not dimensions and not tokens & {"total", "sum", "average", "mean"}:
        raise UnsupportedQuestionError("The built-in rules cannot determine the requested grouping or aggregation.")

    select_parts: list[str] = []
    group_parts: list[str] = []
    output_dimension_names: list[str] = []
    date_dimension = dimensions[0] if dimensions and _is_date_column(dimensions[0].name, dimensions[0].data_type) else None
    for column in dimensions:
        quoted = _quote(column.name)
        if _is_date_column(column.name, column.data_type):
            if tokens & {"year", "yearly"}:
                expression = f"strftime('%Y', {quoted})"
                alias = "year"
            elif tokens & {"day", "daily"}:
                expression = f"strftime('%Y-%m-%d', {quoted})"
                alias = "day"
            else:
                expression = f"strftime('%Y-%m', {quoted})"
                alias = "month"
        else:
            expression = quoted
            alias = column.name
        select_parts.append(f"{expression} AS {_quote(alias)}")
        group_parts.append(expression)
        output_dimension_names.append(alias)

    if count_intent:
        aggregate = f"COUNT(DISTINCT {_quote(entity.name)})" if entity else "COUNT(*)"
        aggregate_alias = "entity_count" if entity else "row_count"
    else:
        assert measure is not None
        if tokens & {"average", "mean", "avg"}:
            aggregate = f"AVG({_quote(measure.name)})"
            aggregate_alias = f"average_{_column_slug(measure.name)}"
        else:
            aggregate = f"SUM({_quote(measure.name)})"
            aggregate_alias = f"total_{_column_slug(measure.name)}"

    percentage = bool(tokens & {"percentage", "percent", "share"})
    if percentage and not count_intent:
        aggregate = f"100.0 * {aggregate} / NULLIF(SUM({aggregate}) OVER (), 0)"
        aggregate_alias = "percentage_of_total"
    select_parts.append(f"{aggregate} AS {_quote(aggregate_alias)}")

    sql = f"SELECT {', '.join(select_parts)} FROM {_quote(schema.table_name)}"
    if group_parts:
        sql += " GROUP BY " + ", ".join(group_parts)

    descending = not bool(tokens & {"lowest", "least", "smallest"})
    if ranking:
        sql += f" ORDER BY {_quote(aggregate_alias)} {'DESC' if descending else 'ASC'}"
    elif time_intent and date_dimension is not None:
        sql += f" ORDER BY {_quote(output_dimension_names[0])} ASC"
    elif group_parts:
        sql += f" ORDER BY {_quote(aggregate_alias)} DESC"

    top_match = re.search(r"\btop\s+(\d+)\b", question.casefold())
    if ranking:
        limit = min(int(top_match.group(1)), 1000) if top_match else 1
        sql += f" LIMIT {limit}"

    return validate_sql(sql)


def _schema_prompt(schema: DatasetSchema) -> str:
    shown_columns = [
        {"name": column.name, "type": column.data_type, "examples": list(column.examples)}
        for column in schema.columns[:100]
    ]
    payload = {
        "table": schema.table_name,
        "columns": shown_columns,
        "omitted_columns": max(0, len(schema.columns) - len(shown_columns)),
    }
    return json.dumps(payload, ensure_ascii=True)


def _fallback_result(question: str, schema: DatasetSchema, warning: str | None = None) -> SQLGenerationResult:
    try:
        return SQLGenerationResult(rule_based_sql(question, schema), "rules", warning=warning)
    except (UnsupportedQuestionError, SQLValidationError) as exc:
        return SQLGenerationResult(None, "rules", error=str(exc), warning=warning)


def generate_sql(
    question: str,
    schema: DatasetSchema | pd.DataFrame | Mapping[str, Any],
    *,
    provider: ChatProvider | None = None,
) -> SQLGenerationResult:
    """Generate SQL via a configured provider or a deterministic local fallback."""
    try:
        normalized_schema = _coerce_schema(schema)
    except (TypeError, ValueError) as exc:
        return SQLGenerationResult(None, "rules", error=str(exc))

    if provider is None:
        try:
            provider = create_chat_provider(AIConfig.from_env())
        except ConfigurationError:
            return _fallback_result(question, normalized_schema, "Invalid AI configuration; deterministic rules were used.")
    if provider is None:
        return _fallback_result(question, normalized_schema)

    system_prompt = (
        "You generate analytical SQL for SQLite. Return exactly one read-only SELECT query, "
        "using only the supplied table and columns. Never use DDL, DML, PRAGMA, comments, "
        "or multiple statements. If the data cannot answer the question, return only UNSUPPORTED."
    )
    user_prompt = (
        f"Dataset schema JSON:\n{_schema_prompt(normalized_schema)}\n\n"
        f"Question:\n{question.strip()}\n\nReturn SQL only."
    )
    try:
        completion = provider.complete(system_prompt, user_prompt).strip()
    except Exception:
        return _fallback_result(question, normalized_schema, "The model was unavailable; deterministic rules were used.")

    completion = re.sub(r"^```(?:sql)?\s*|\s*```$", "", completion, flags=re.IGNORECASE).strip()
    if completion.casefold() == "unsupported":
        return SQLGenerationResult(None, "model", error="The question is not answerable from the available schema.")
    try:
        safe_sql = validate_sql(completion)
    except SQLValidationError as exc:
        return SQLGenerationResult(None, "model", error=f"Generated SQL was rejected: {exc}")
    return SQLGenerationResult(safe_sql, "model")