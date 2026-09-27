"""Download helpers and factual insights for executed query results."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import plotly.graph_objects as go

from src.answer_generator import INSUFFICIENT_DATA_ANSWER


class ExportError(RuntimeError):
    """Raised when an analysis artifact cannot be generated."""


def dataframe_to_csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


def dataframe_to_excel(frame: pd.DataFrame) -> bytes:
    output = BytesIO()
    try:
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            frame.to_excel(writer, index=False, sheet_name="Query Results")
    except (ImportError, ValueError, OSError) as exc:
        raise ExportError("The query results could not be exported to Excel.") from exc
    return output.getvalue()


def chart_to_png(figure: go.Figure) -> bytes:
    try:
        return figure.to_image(format="png", width=1200, height=760, scale=2)
    except Exception as exc:
        raise ExportError("PNG export is unavailable. Verify that Kaleido is installed correctly.") from exc


def _display_value(value: object) -> str:
    if pd.isna(value):
        return "NULL"
    if isinstance(value, float):
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    return str(value)


def build_key_insight(question: str, frame: pd.DataFrame) -> str:
    """Describe only values present in the actual query result."""
    if frame.empty:
        return INSUFFICIENT_DATA_ANSWER

    numeric_columns = [
        column for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column]) and not pd.api.types.is_bool_dtype(frame[column])
    ]
    if not numeric_columns:
        return f"The query returned {len(frame):,} rows. No numeric measure was included in the result."

    measure = numeric_columns[-1]
    dimensions = [column for column in frame.columns if column not in numeric_columns]
    if len(frame) == 1:
        record = frame.iloc[0]
        details = [f"{column} = {_display_value(record[column])}" for column in frame.columns]
        return "Key result: " + "; ".join(details) + "."

    values = pd.to_numeric(frame[measure], errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return f"The query returned {len(frame):,} rows, but no numeric values for {measure} were available."

    high_index = valid.idxmax()
    low_index = valid.idxmin()
    high_value = _display_value(frame.loc[high_index, measure])
    low_value = _display_value(frame.loc[low_index, measure])
    if dimensions:
        dimension = dimensions[0]
        high_label = _display_value(frame.loc[high_index, dimension])
        low_label = _display_value(frame.loc[low_index, dimension])
        return (
            f"Across {len(frame):,} returned groups, {high_label} has the highest {measure} "
            f"({high_value}) and {low_label} has the lowest ({low_value})."
        )
    return (
        f"Across {len(frame):,} returned values, the highest {measure} is {high_value} "
        f"and the lowest is {low_value}."
    )


def build_analysis_report(
    *,
    question: str,
    sql: str,
    frame: pd.DataFrame,
    key_insight: str,
    chart_png: bytes | None = None,
    truncated: bool = False,
) -> bytes:
    """Build a ZIP containing a Markdown summary, CSV, and optional PNG chart."""
    row_note = f"At least {len(frame):,} rows are included (display limit reached)." if truncated else f"{len(frame):,} rows are included."
    preview = frame.head(100).to_csv(index=False)
    markdown = "\n".join(
        [
            "# AI Data Analyst Report",
            "",
            "## Question",
            question,
            "",
            "## Generated SQL",
            "```sql",
            sql,
            "```",
            "",
            "## Result Summary",
            row_note,
            "",
            "```csv",
            preview.rstrip(),
            "```",
            "",
            "## Key Insight",
            key_insight,
            "",
            "Chart included as `chart.png`." if chart_png else "No chart was generated for this result.",
            "",
        ]
    )

    output = BytesIO()
    try:
        with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("analysis_report.md", markdown)
            archive.writestr("query_results.csv", dataframe_to_csv(frame))
            if chart_png:
                archive.writestr("chart.png", chart_png)
    except OSError as exc:
        raise ExportError("The analysis report could not be packaged for download.") from exc
    return output.getvalue()