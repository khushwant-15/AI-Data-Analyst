"""Choose and create useful Plotly charts from analytical query results."""

from __future__ import annotations

import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _is_date_column(frame: pd.DataFrame, column: str) -> bool:
    if pd.api.types.is_datetime64_any_dtype(frame[column]):
        return True
    normalized = re.sub(r"[^a-z]", "", str(column).casefold())
    return normalized in {"date", "day", "month", "year", "datetime", "timestamp"}


def select_chart_type(frame: pd.DataFrame, question: str = "") -> str | None:
    """Select a chart type from result shape and question wording."""
    if frame.empty:
        return None
    date_columns = [column for column in frame.columns if _is_date_column(frame, column)]
    numeric_columns = [
        column for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column]) and not pd.api.types.is_bool_dtype(frame[column])
    ]
    categorical_columns = [column for column in frame.columns if column not in date_columns + numeric_columns]
    question_text = question.casefold()

    if date_columns and numeric_columns and len(frame) > 1:
        return "line"
    if categorical_columns and numeric_columns:
        share_request = any(word in question_text for word in ("share", "percentage", "percent"))
        category_count = frame[categorical_columns[0]].nunique(dropna=True)
        if share_request and 2 <= category_count <= 6 and (frame[numeric_columns[0]] >= 0).all():
            return "pie"
        return "bar"
    if len(numeric_columns) >= 2 and len(frame) > 1:
        return "scatter"
    if len(numeric_columns) == 1 and len(frame) > 1 and frame[numeric_columns[0]].nunique(dropna=True) > 1:
        return "histogram"
    if categorical_columns and frame[categorical_columns[0]].nunique(dropna=True) > 1:
        return "bar"
    return None


def _axis_label(column: str) -> str:
    label = str(column).replace("_", " ").strip()
    for prefix in ("total ", "average ", "percentage of "):
        if label.casefold().startswith(prefix):
            return prefix.title() + label[len(prefix):].title()
    return label.title()


def create_chart(frame: pd.DataFrame, question: str = "") -> go.Figure | None:
    """Create a labeled chart, or return None when the result has no useful visual."""
    chart_type = select_chart_type(frame, question)
    if chart_type is None:
        return None

    date_columns = [column for column in frame.columns if _is_date_column(frame, column)]
    numeric_columns = [
        column for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column]) and not pd.api.types.is_bool_dtype(frame[column])
    ]
    categorical_columns = [column for column in frame.columns if column not in date_columns + numeric_columns]
    title = question.strip().rstrip("?") or "Analysis result"
    title = title[:90]
    labels = {column: _axis_label(column) for column in frame.columns}

    if chart_type == "line":
        point_labels = frame[numeric_columns[0]].map(lambda value: f"{value:,.4g}") if len(frame) <= 12 else None
        figure = px.line(
            frame,
            x=date_columns[0],
            y=numeric_columns[0],
            markers=True,
            title=title,
            labels=labels,
            text=point_labels,
        )
        figure.update_traces(line={"width": 3}, marker={"size": 8}, textposition="top center")
    elif chart_type == "pie":
        figure = px.pie(
            frame,
            names=categorical_columns[0],
            values=numeric_columns[0],
            hole=0.48,
            title=title,
            labels=labels,
        )
        figure.update_traces(
            textinfo="label+percent",
            texttemplate="%{label}<br>%{value:,.4g} (%{percent})",
            textposition="auto",
            textfont_size=12,
        )
    elif chart_type == "bar" and categorical_columns and numeric_columns:
        figure = px.bar(
            frame,
            x=numeric_columns[0],
            y=categorical_columns[0],
            orientation="h",
            title=title,
            labels=labels,
            text_auto=True,
        )
        figure.update_traces(textposition="outside", cliponaxis=False, textfont_size=12)
    elif chart_type == "scatter":
        figure = px.scatter(
            frame,
            x=numeric_columns[0],
            y=numeric_columns[1],
            title=title,
            labels=labels,
        )
    elif chart_type == "histogram":
        figure = px.histogram(
            frame,
            x=numeric_columns[0],
            nbins=min(30, max(5, len(frame) // 2)),
            title=title,
            labels=labels,
        )
    else:
        category = categorical_columns[0]
        counts = frame[category].value_counts(dropna=True).rename_axis(category).reset_index(name="records")
        figure = px.bar(
            counts,
            x="records",
            y=category,
            orientation="h",
            title=title,
            labels={category: _axis_label(category), "records": "Records"},
        )

    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        height=470,
        autosize=True,
        font={"family": "DM Sans, sans-serif", "color": "#25332d", "size": 13},
        title={"font": {"size": 20, "color": "#1f2b26"}, "x": 0.025, "xanchor": "left", "y": 0.97},
        margin={"l": 28, "r": 34, "t": 82, "b": 72},
        showlegend=chart_type == "pie",
    )
    if chart_type != "pie":
        figure.update_xaxes(
            automargin=True,
            title_standoff=12,
            title_font={"size": 14, "color": "#34443b"},
            tickfont={"size": 12, "color": "#52635a"},
            gridcolor="#e8eeea" if chart_type in {"bar", "scatter", "histogram"} else "rgba(0,0,0,0)",
            zerolinecolor="#d3ded6",
        )
        figure.update_yaxes(
            automargin=True,
            title_standoff=14,
            title_font={"size": 14, "color": "#34443b"},
            tickfont={"size": 12, "color": "#52635a"},
            gridcolor="#e8eeea" if chart_type == "line" else "rgba(0,0,0,0)",
            zerolinecolor="#d3ded6",
        )
        if chart_type == "bar":
            figure.update_yaxes(autorange="reversed")
    return figure