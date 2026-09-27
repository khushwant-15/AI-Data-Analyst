from io import BytesIO
from zipfile import ZipFile

import pandas as pd

from src.chart_generator import create_chart
from src.export_utils import (
    build_analysis_report,
    build_key_insight,
    chart_to_png,
    dataframe_to_csv,
    dataframe_to_excel,
)


def test_insight_uses_only_actual_result_values():
    result = pd.DataFrame({"month": ["2025-01", "2025-02", "2025-03"], "total_sales": [100, 250, 130]})
    insight = build_key_insight("Show monthly sales trend", result)
    assert "2025-02" in insight
    assert "250" in insight
    assert "2025-01" in insight
    assert "100" in insight
    assert "because" not in insight


def test_empty_result_insight_reports_insufficient_data():
    assert "insufficient" in build_key_insight("question", pd.DataFrame()).casefold()


def test_csv_and_excel_exports_contain_result_rows():
    frame = pd.DataFrame({"product": ["A", "B"], "sales": [12, 25]})
    csv_data = dataframe_to_csv(frame).decode("utf-8-sig")
    excel_data = dataframe_to_excel(frame)

    assert "product,sales" in csv_data
    assert "B,25" in csv_data
    restored = pd.read_excel(BytesIO(excel_data), engine="openpyxl")
    assert restored.to_dict("records") == frame.to_dict("records")


def test_chart_png_and_downloadable_report_include_chart_and_summary():
    frame = pd.DataFrame({"region": ["East", "West"], "total_sales": [100, 200]})
    figure = create_chart(frame, "Sales by region")
    png = chart_to_png(figure)
    report = build_analysis_report(
        question="Which region generated more sales?",
        sql='SELECT "region", SUM("sales") AS "total_sales" FROM "dataset" GROUP BY "region"',
        frame=frame,
        key_insight="West has the higher returned sales value (200).",
        chart_png=png,
    )

    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    with ZipFile(BytesIO(report)) as archive:
        assert set(archive.namelist()) == {"analysis_report.md", "query_results.csv", "chart.png"}
        summary = archive.read("analysis_report.md").decode("utf-8")
        assert "Which region generated more sales?" in summary
        assert "West has the higher returned sales value" in summary
        assert archive.read("chart.png").startswith(b"\x89PNG\r\n\x1a\n")