"""Dataset quality profiling that leaves the source DataFrame untouched."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _missing_mask(series: pd.Series) -> pd.Series:
    blank_text = series.map(lambda value: isinstance(value, str) and not value.strip())
    return series.isna() | blank_text


def _date_parse(series: pd.Series) -> tuple[pd.Series, bool]:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce"), True
    name_suggests_date = any(token in str(series.name).casefold() for token in ("date", "time", "day"))
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return pd.to_datetime(series, errors="coerce"), False

    values = series[~_missing_mask(series)]
    if values.empty:
        return pd.to_datetime(series, errors="coerce"), False
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    if parsed.isna().all():
        parsed = values.map(lambda value: pd.to_datetime(value, errors="coerce"))
    ratio = float(parsed.notna().mean())
    detected = ratio >= 0.6 or (name_suggests_date and ratio > 0)
    full_parsed = pd.to_datetime(series, errors="coerce", format="mixed") if detected else pd.Series(pd.NaT, index=series.index)
    if detected and full_parsed.isna().all():
        full_parsed = series.map(lambda value: pd.to_datetime(value, errors="coerce"))
    return full_parsed, detected


def profile_data(frame: pd.DataFrame) -> dict[str, Any]:
    """Return dataset-level and per-column data quality details."""
    rows, columns = frame.shape
    column_profiles: list[dict[str, Any]] = []
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    date_columns: list[str] = []
    statistics: dict[str, dict[str, float]] = {}
    invalid_dates = 0

    for name in frame.columns:
        series = frame[name]
        missing = _missing_mask(series)
        parsed_dates, is_date = _date_parse(series)
        if is_date:
            date_columns.append(str(name))
            invalid_dates += int((~missing & parsed_dates.isna()).sum())
            dtype = "datetime"
        elif pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            numeric_columns.append(str(name))
            dtype = str(series.dtype)
            values = pd.to_numeric(series, errors="coerce").dropna()
            if not values.empty:
                q1, q3 = values.quantile([0.25, 0.75])
                iqr = q3 - q1
                outlier_count = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())
                statistics[str(name)] = {
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "outlier_count": outlier_count,
                }
        else:
            categorical_columns.append(str(name))
            dtype = str(series.dtype)

        examples = [str(value) for value in series[~missing].head(3).tolist()]
        column_profiles.append(
            {
                "column": str(name),
                "data_type": dtype,
                "missing": int(missing.sum()),
                "unique_values": int(series.nunique(dropna=True)),
                "examples": examples,
            }
        )

    total_missing = int(sum(item["missing"] for item in column_profiles))
    duplicate_rows = int(frame.duplicated().sum()) if rows else 0
    return {
        "rows": int(rows),
        "columns": int(columns),
        "missing_values": total_missing,
        "duplicate_rows": duplicate_rows,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "date_columns": date_columns,
        "invalid_dates": invalid_dates,
        "column_profiles": column_profiles,
        "statistics": statistics,
    }