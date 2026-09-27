from __future__ import annotations

import hashlib
import html
import logging
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.answer_generator import generate_answer
from src.ai_provider import create_chat_provider
from src.chart_generator import create_chart
from src.config import AIConfig, ConfigurationError
from src.data_loader import DataLoadError, load_dataset
from src.data_profiler import profile_data
from src.database import create_database
from src.export_utils import (
    ExportError,
    build_analysis_report,
    build_key_insight,
    chart_to_png,
    dataframe_to_csv,
    dataframe_to_excel,
)
from src.query_engine import execute_query
from src.sql_generator import build_schema, generate_sql


LOGGER = logging.getLogger(__name__)
SAMPLE_PATH = Path(__file__).parent / "sample_data" / "sample_sales.csv"
SAMPLE_QUESTIONS = (
    "Which product generated the highest sales?",
    "Show monthly sales trend",
    "Which city has the lowest profit?",
    "How many customers are there?",
)


st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="AD",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
    :root {
        --ink: #1f2b26;
        --muted: #69766f;
        --green: #17765c;
        --green-dark: #115b48;
        --mint: #e7f2ec;
        --line: #dfe7e1;
        --canvas: #f6f8f6;
        --white: #ffffff;
        --amber: #b66a35;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', 'Segoe UI', sans-serif; color: var(--ink); }
    .stApp {
        background-color: var(--canvas);
        background-image: linear-gradient(rgba(47, 89, 68, .025) 1px, transparent 1px),
                          linear-gradient(90deg, rgba(47, 89, 68, .025) 1px, transparent 1px);
        background-size: 28px 28px;
    }
    [data-testid="stAppViewContainer"] { background: transparent; }
    [data-testid="stHeader"] { background: rgba(246, 248, 246, .88); }
    [data-testid="stAppViewContainer"] .main .block-container {
        max-width: 1380px; padding: 2rem 3rem 4rem;
    }
    [data-testid="stSidebar"] {
        background: #eef3ef; border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
    h1, h2, h3, h4 { color: var(--ink); font-family: 'DM Sans', 'Segoe UI', sans-serif; letter-spacing: 0; }
    h1 { font-size: 2.1rem; font-weight: 700; line-height: 1.12; margin-bottom: .2rem; }
    h2 { font-size: 1.45rem; font-weight: 650; }
    h3, h4 { font-size: 1.05rem; font-weight: 650; }
    p, label, [data-testid="stCaptionContainer"] { color: var(--muted); }
    .product-lockup { display: flex; align-items: center; gap: .65rem; margin-bottom: .35rem; }
    .brand-mark {
        width: 34px; height: 34px; display: inline-flex; align-items: center; justify-content: center;
        background: var(--green); border-radius: 7px; color: white; font-weight: 700; font-size: .78rem;
    }
    .eyebrow, .section-kicker {
        color: var(--green); font-size: .69rem; font-weight: 700; letter-spacing: .09em;
        text-transform: uppercase; margin: 0 0 .5rem;
    }
    .subtitle { color: var(--muted); font-size: 1rem; margin-top: .1rem; }
    .header-row { border-bottom: 1px solid var(--line); padding-bottom: 1.2rem; margin-bottom: 1.45rem; }
    .mode-pill {
        display: inline-flex; align-items: center; gap: .45rem; border: 1px solid #c9ddd1;
        background: var(--mint); color: var(--green-dark); border-radius: 999px;
        padding: .43rem .68rem; font-size: .71rem; font-weight: 700; white-space: nowrap;
    }
    .mode-dot { width: 7px; height: 7px; background: #23825f; border-radius: 50%; }
    [data-testid="stMetric"] {
        background: var(--white); border: 1px solid var(--line); border-radius: 7px;
        padding: .9rem 1rem; min-height: 112px;
    }
    [data-testid="stMetricLabel"] p { font-size: .72rem; font-weight: 650; color: var(--muted); }
    [data-testid="stMetricValue"] { font-size: 1.55rem; font-weight: 700; color: var(--ink); }
    [data-testid="stMetricDelta"] { font-size: .74rem; }
    .section-heading { margin: 1.6rem 0 .3rem; }
    .section-subtitle { color: var(--muted); font-size: .88rem; margin: 0 0 1rem; }
    .empty-panel {
        background: rgba(255,255,255,.84); border: 1px dashed #b8c9bd; border-radius: 8px;
        padding: 2.2rem; text-align: center; margin: 1.25rem 0;
    }
    .empty-panel h3 { margin: .2rem 0 .4rem; }
    .empty-panel p { margin: 0; }
    .answer-panel {
        background: #edf5ef; border-left: 3px solid var(--green); border-radius: 0 7px 7px 0;
        padding: 1.1rem 1.25rem; margin: .45rem 0 1rem; color: var(--ink);
    }
    .answer-panel p { margin: 0; color: var(--ink); font-size: 1.02rem; line-height: 1.55; }
    .sidebar-brand { display: flex; align-items: center; gap: .6rem; margin: 0 0 1.2rem; color: var(--ink); font-weight: 700; }
    .dataset-chip { border: 1px solid var(--line); border-radius: 6px; background: #f8faf8; padding: .65rem .7rem; }
    .dataset-chip strong { display: block; overflow-wrap: anywhere; color: var(--ink); font-size: .82rem; }
    .dataset-chip span { color: var(--muted); font-size: .74rem; }
    [data-testid="stTextArea"] textarea {
        background: white; border: 1px solid #cbd8cf; border-radius: 7px; color: var(--ink);
        font-size: .98rem; line-height: 1.5;
    }
    [data-testid="stTextArea"] textarea:focus { border-color: var(--green); box-shadow: 0 0 0 2px rgba(23,118,92,.12); }
    .stButton > button, [data-testid="stFormSubmitButton"] button {
        border: 1px solid #cbd8cf; border-radius: 6px; min-height: 42px; font-weight: 600;
        color: var(--ink); background: white; transition: background .15s ease, border-color .15s ease;
    }
    .stButton > button:hover { color: var(--green-dark); border-color: var(--green); background: #f3f8f4; }
    [data-testid="stFormSubmitButton"] button[kind="primary"] { background: var(--green); border-color: var(--green); color: white; }
    [data-testid="stFormSubmitButton"] button[kind="primary"]:hover { background: var(--green-dark); border-color: var(--green-dark); }
    [data-testid="stFileUploader"] section { background: rgba(255,255,255,.65); border: 1px dashed #b8c9bd; border-radius: 7px; }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 7px; overflow: hidden; }
    [data-testid="stPlotlyChart"] {
        min-height: 490px; width: 100%; padding: .45rem; box-sizing: border-box;
        background: #ffffff; border: 1px solid var(--line); border-radius: 8px;
    }
    [data-testid="stAlert"] { border-radius: 6px; }
    code, pre, [data-testid="stCode"] { font-family: 'IBM Plex Mono', Consolas, monospace !important; }
    hr { border-color: var(--line); }
    @media (max-width: 760px) {
        [data-testid="stAppViewContainer"] .main .block-container { padding: 1.3rem 1rem 2.5rem; }
        h1 { font-size: 1.7rem; }
        [data-testid="stMetric"] { min-height: 95px; padding: .7rem; }
        [data-testid="stMetricValue"] { font-size: 1.2rem; }
        .empty-panel { padding: 1.45rem 1rem; }
        [data-testid="stPlotlyChart"] { min-height: 420px; padding: .2rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _initialize_state() -> None:
    defaults = {
        "dataset_frame": None,
        "dataset_name": None,
        "dataset_profile": None,
        "dataset_schema": None,
        "database_path": None,
        "dataset_token": None,
        "last_upload_attempt": None,
        "analysis": None,
        "question_input": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _save_dataset(frame: pd.DataFrame, name: str, token: str) -> None:
    descriptor, database_path = tempfile.mkstemp(prefix="ai_data_analyst_", suffix=".sqlite")
    os.close(descriptor)
    try:
        create_database(frame, database_path)
        profile = profile_data(frame)
        schema = build_schema(frame)
    except Exception:
        Path(database_path).unlink(missing_ok=True)
        raise

    old_path = st.session_state.get("database_path")
    if old_path and old_path != database_path:
        Path(old_path).unlink(missing_ok=True)
    st.session_state.dataset_frame = frame
    st.session_state.dataset_name = name
    st.session_state.dataset_profile = profile
    st.session_state.dataset_schema = schema
    st.session_state.database_path = database_path
    st.session_state.dataset_token = token
    st.session_state.analysis = None


def _load_sample() -> None:
    frame = load_dataset(SAMPLE_PATH)
    _save_dataset(frame, SAMPLE_PATH.name, "sample-sales-v1")


def _load_upload(uploaded_file) -> None:
    contents = uploaded_file.getvalue()
    digest = hashlib.sha256(contents).hexdigest()
    token = f"{uploaded_file.name}:{digest}"
    if token == st.session_state.dataset_token or token == st.session_state.last_upload_attempt:
        return
    st.session_state.last_upload_attempt = token
    frame = load_dataset(uploaded_file)
    _save_dataset(frame, uploaded_file.name, token)


def _render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand"><span class="brand-mark">AD</span><span>ANALYST WORKSPACE</span></div>',
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Workspace",
            ["Ask Your Data", "Dashboard", "Data Health", "Executive Summary"],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown('<div class="section-kicker">DATASET</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"], help="Maximum file size: 50 MB")
        if uploaded_file is not None:
            try:
                _load_upload(uploaded_file)
            except DataLoadError as exc:
                st.error(str(exc))
            except Exception:
                LOGGER.exception("Dataset upload failed")
                st.error("The dataset could not be prepared. Check the file and try again.")

        if st.button("Load sample sales", use_container_width=True, key="load_sample_sales"):
            try:
                _load_sample()
            except Exception:
                LOGGER.exception("Sample dataset load failed")
                st.error("The sample dataset could not be loaded.")

        frame = st.session_state.dataset_frame
        if frame is not None:
            st.markdown(
                f'<div class="dataset-chip"><strong>{html.escape(st.session_state.dataset_name)}</strong>'
                f'<span>{len(frame):,} rows / {len(frame.columns):,} columns</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("No dataset loaded")

        st.divider()
        try:
            config = AIConfig.from_env()
            provider = create_chat_provider(config)
            mode = "MODEL READY" if provider else "LOCAL RULES"
        except ConfigurationError:
            mode = "LOCAL RULES"
        st.markdown(
            f'<div class="mode-pill"><span class="mode-dot"></span>{mode}</div>',
            unsafe_allow_html=True,
        )
        st.caption("No model configured: analysis stays local.")
    return page


def _render_header() -> None:
    left, right = st.columns([5, 1.2], vertical_alignment="center")
    with left:
        st.markdown('<div class="eyebrow">DATA INTELLIGENCE / WORKSPACE</div>', unsafe_allow_html=True)
        st.title("AI Data Analyst")
        st.markdown('<div class="subtitle">Ask questions. Discover insights. Make data-driven decisions.</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="mode-pill"><span class="mode-dot"></span>READY TO ANALYZE</div>', unsafe_allow_html=True)
    st.divider()


def _render_metrics(profile: dict) -> None:
    metrics = [
        ("Rows", f"{profile['rows']:,}"),
        ("Columns", f"{profile['columns']:,}"),
        ("Missing values", f"{profile['missing_values']:,}"),
        ("Duplicate rows", f"{profile['duplicate_rows']:,}"),
    ]
    cols = st.columns(4)
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.metric(label, value)


def _analyze(question: str) -> None:
    with st.spinner("Analyzing your question against the dataset..."):
        try:
            generated = generate_sql(question, st.session_state.dataset_schema)
            if not generated.success:
                st.session_state.analysis = {
                    "question": question,
                    "sql": None,
                    "answer": None,
                    "error": generated.error or "The question could not be translated into a safe query.",
                    "warning": generated.warning,
                    "source": generated.source,
                    "frame": pd.DataFrame(),
                    "truncated": False,
                    "key_insight": None,
                    "chart_png": None,
                }
                return

            result = execute_query(st.session_state.database_path, generated.sql)
            answer = generate_answer(question, result)
            st.session_state.analysis = {
                "question": question,
                "sql": generated.sql,
                "answer": answer,
                "error": result.error,
                "warning": generated.warning,
                "source": generated.source,
                "frame": result.dataframe,
                "truncated": result.truncated,
                "key_insight": build_key_insight(question, result.dataframe) if result.status == "success" else None,
                "chart_png": None,
            }
        except Exception:
            LOGGER.exception("Question analysis failed")
            st.session_state.analysis = {
                "question": question,
                "sql": None,
                "answer": None,
                "error": "Analysis could not be completed. Check the question and dataset, then try again.",
                "warning": None,
                "source": "rules",
                "frame": pd.DataFrame(),
                "truncated": False,
                "key_insight": None,
                "chart_png": None,
            }


def _render_analysis() -> None:
    analysis = st.session_state.analysis
    if not analysis:
        return
    st.markdown('<div class="section-heading"></div>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="section-kicker">ANALYSIS RESULT</div>', unsafe_allow_html=True)
    st.markdown(f"#### {analysis['question']}")
    if analysis.get("warning"):
        st.warning(analysis["warning"])
    if analysis.get("error"):
        st.error(analysis["error"])
    elif analysis.get("answer"):
        st.markdown(
            f'<div class="answer-panel"><div class="section-kicker">ANSWER</div>'
            f'<p>{html.escape(analysis["answer"])}</p></div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Generated with {analysis['source']} analysis")
        result_frame = analysis["frame"]
        if not result_frame.empty:
            st.markdown("#### Query result")
            row_label = "At least" if analysis["truncated"] else ""
            st.caption(f"{row_label} {len(result_frame):,} rows returned".strip())
            st.dataframe(result_frame, hide_index=True, use_container_width=True)
            st.markdown("#### Visualization")
            try:
                figure = create_chart(result_frame, analysis["question"])
                if figure is None:
                    st.caption("No chart is useful for this result shape.")
                else:
                    st.plotly_chart(
                        figure,
                        use_container_width=True,
                        height=500,
                        config={
                            "displayModeBar": True,
                            "displaylogo": False,
                            "toImageButtonOptions": {
                                "format": "png",
                                "filename": "ai-data-analyst-chart",
                                "width": 1400,
                                "height": 900,
                                "scale": 2,
                            },
                        },
                    )

                    insight = analysis.get("key_insight") or build_key_insight(analysis["question"], result_frame)
                    st.markdown(
                        f'<div class="answer-panel"><div class="section-kicker">KEY INSIGHT</div>'
                        f'<p>{html.escape(insight)}</p></div>',
                        unsafe_allow_html=True,
                    )

                    csv_col, excel_col = st.columns(2)
                    csv_col.download_button(
                        "Download query result as CSV",
                        data=dataframe_to_csv(result_frame),
                        file_name="query_results.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_query_csv",
                    )
                    excel_col.download_button(
                        "Download query result as Excel",
                        data=dataframe_to_excel(result_frame),
                        file_name="query_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        key="download_query_excel",
                    )

                    png_data = analysis.get("chart_png")
                    png_error = None
                    if png_data is None:
                        try:
                            png_data = chart_to_png(figure)
                            analysis["chart_png"] = png_data
                        except ExportError as exc:
                            png_error = str(exc)
                    png_col, report_col = st.columns(2)
                    png_col.download_button(
                        "Download chart as PNG",
                        data=png_data or b"",
                        file_name="ai-data-analyst-chart.png",
                        mime="image/png",
                        disabled=png_data is None,
                        use_container_width=True,
                        key="download_chart_png",
                    )
                    report_data = build_analysis_report(
                        question=analysis["question"],
                        sql=analysis["sql"],
                        frame=result_frame,
                        key_insight=insight,
                        chart_png=png_data,
                        truncated=analysis["truncated"],
                    )
                    report_col.download_button(
                        "Download analysis report",
                        data=report_data,
                        file_name="ai-data-analyst-report.zip",
                        mime="application/zip",
                        use_container_width=True,
                        key="download_analysis_report",
                    )
                    if png_error:
                        st.warning(png_error)
            except Exception:
                LOGGER.exception("Chart generation failed")
                st.warning("The result is available, but a chart could not be prepared.")

    if analysis.get("sql"):
        with st.expander("Generated SQL", expanded=False):
            st.code(analysis["sql"], language="sql")


def _render_question_area() -> None:
    frame = st.session_state.dataset_frame
    if frame is None:
        st.markdown(
            '<div class="empty-panel"><div class="eyebrow">WORKSPACE EMPTY</div>'
            '<h3>No dataset loaded</h3><p>Upload a file or load the sample sales dataset from the sidebar.</p></div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown('<div class="section-heading"><div class="section-kicker">ASK YOUR DATA</div></div>', unsafe_allow_html=True)
    st.markdown("### What would you like to know?")
    st.markdown('<p class="section-subtitle">Choose a starting question or write your own.</p>', unsafe_allow_html=True)
    sample_cols = st.columns(2)
    for index, sample in enumerate(SAMPLE_QUESTIONS):
        with sample_cols[index % 2]:
            if st.button(sample, key=f"sample_question_{index}", use_container_width=True):
                st.session_state.question_input = sample

    with st.form("analysis_form"):
        st.text_area(
            "Question",
            key="question_input",
            height=90,
            placeholder="For example: Which product generated the highest sales?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Analyze", type="primary", use_container_width=True)
    if submitted:
        question = st.session_state.question_input.strip()
        if not question:
            st.warning("Enter a question before analyzing the dataset.")
        else:
            _analyze(question)
    _render_analysis()


def _render_dashboard() -> None:
    profile = st.session_state.dataset_profile
    if profile is None:
        _render_question_area()
        return
    _render_metrics(profile)
    has_issues = profile["missing_values"] or profile["duplicate_rows"] or profile["invalid_dates"]
    if has_issues:
        st.warning(
            f"Data checks found {profile['missing_values']:,} missing values, "
            f"{profile['duplicate_rows']:,} duplicate rows, and {profile['invalid_dates']:,} invalid dates."
        )
    else:
        st.success("No missing values, duplicate rows, or invalid dates detected.")
    _render_question_area()


def _render_data_health() -> None:
    frame = st.session_state.dataset_frame
    profile = st.session_state.dataset_profile
    if frame is None or profile is None:
        _render_question_area()
        return
    st.markdown('<div class="section-kicker">DATA HEALTH</div>', unsafe_allow_html=True)
    st.markdown("### Dataset quality")
    _render_metrics(profile)
    type_cols = st.columns(3)
    type_cols[0].metric("Numeric columns", len(profile["numeric_columns"]))
    type_cols[1].metric("Categorical columns", len(profile["categorical_columns"]))
    type_cols[2].metric("Date columns", len(profile["date_columns"]))

    quality = pd.DataFrame(profile["column_profiles"])
    if not quality.empty:
        quality["examples"] = quality["examples"].map(lambda values: ", ".join(values))
        quality = quality.rename(
            columns={
                "column": "Column",
                "data_type": "Data type",
                "missing": "Missing",
                "unique_values": "Unique values",
                "examples": "Example values",
            }
        )
        st.markdown("#### Column quality")
        st.dataframe(quality, hide_index=True, use_container_width=True)

    if profile["invalid_dates"]:
        st.warning(f"{profile['invalid_dates']:,} non-empty values could not be parsed as dates.")
    if profile["statistics"]:
        st.markdown("#### Numeric profile")
        statistics = pd.DataFrame.from_dict(profile["statistics"], orient="index").rename_axis("Column").reset_index()
        st.dataframe(statistics, hide_index=True, use_container_width=True)


def _render_executive_summary() -> None:
    frame = st.session_state.dataset_frame
    profile = st.session_state.dataset_profile
    if frame is None or profile is None:
        _render_question_area()
        return
    st.markdown('<div class="section-kicker">EXECUTIVE SUMMARY</div>', unsafe_allow_html=True)
    st.markdown("### Dataset at a glance")
    _render_metrics(profile)

    numeric = [
        column for column in profile["numeric_columns"]
        if pd.api.types.is_numeric_dtype(frame[column])
    ]
    if numeric:
        st.markdown("#### Numeric indicators")
        columns = st.columns(min(4, len(numeric)))
        for index, name in enumerate(numeric[:4]):
            total = pd.to_numeric(frame[name], errors="coerce").sum()
            columns[index % len(columns)].metric(f"Total {name}", f"{total:,.2f}".rstrip("0").rstrip("."))

    findings = []
    for name in profile["categorical_columns"]:
        unique_count = int(frame[name].nunique(dropna=True))
        if 2 <= unique_count <= min(25, max(2, len(frame) // 2)):
            counts = frame[name].value_counts(dropna=True)
            if not counts.empty:
                findings.append(f"{name}: {counts.index[0]} is the most frequent value ({int(counts.iloc[0]):,} rows).")
        if len(findings) >= 3:
            break
    st.markdown("#### Observed findings")
    if findings:
        st.dataframe(pd.DataFrame({"Observed finding": findings}), hide_index=True, use_container_width=True)
    else:
        st.caption("No low-cardinality category summaries are available for this dataset.")

    analysis = st.session_state.analysis
    if analysis and analysis.get("answer"):
        st.markdown("#### Latest analysis")
        st.markdown(f"**{analysis['question']}**")
        st.markdown(analysis["answer"])


_initialize_state()
current_page = _render_sidebar()
_render_header()

if current_page == "Dashboard":
    _render_dashboard()
elif current_page == "Data Health":
    _render_data_health()
elif current_page == "Executive Summary":
    _render_executive_summary()
else:
    _render_question_area()