import pandas as pd

from src.chart_generator import create_chart, select_chart_type


def test_time_series_selects_line_chart():
    frame = pd.DataFrame({"month": ["2025-01", "2025-02"], "sales": [10, 15]})
    figure = create_chart(frame, "Show monthly sales trend")
    assert select_chart_type(frame, "Show monthly sales trend") == "line"
    assert figure.data[0].type == "scatter"
    assert figure.layout.height == 470
    assert figure.layout.title.text == "Show monthly sales trend"
    assert figure.layout.xaxis.title.text == "Month"
    assert figure.layout.yaxis.title.text == "Sales"


def test_category_and_measure_selects_bar_chart():
    frame = pd.DataFrame({"region": ["East", "West"], "sales": [10, 15]})
    figure = create_chart(frame, "Sales by region")
    assert select_chart_type(frame, "Sales by region") == "bar"
    assert figure.data[0].type == "bar"
    assert figure.data[0].orientation == "h"
    assert figure.layout.xaxis.title.text == "Sales"
    assert figure.layout.yaxis.title.text == "Region"


def test_small_percentage_distribution_selects_donut_chart():
    frame = pd.DataFrame({"region": ["East", "West", "North"], "share": [50, 30, 20]})
    figure = create_chart(frame, "Percentage share by region")
    assert select_chart_type(frame, "Percentage share by region") == "pie"
    assert figure.data[0].type == "pie"
    assert figure.data[0].hole == 0.48


def test_numeric_pair_and_distribution_select_scatter_and_histogram():
    paired = pd.DataFrame({"sales": [10, 15, 20], "profit": [2, 3, 5]})
    values = pd.DataFrame({"sales": [10, 15, 20]})
    assert select_chart_type(paired) == "scatter"
    assert select_chart_type(values) == "histogram"


def test_meaningless_single_value_result_has_no_chart():
    assert select_chart_type(pd.DataFrame({"total_sales": [100]})) is None
    assert create_chart(pd.DataFrame()) is None