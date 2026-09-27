from __future__ import annotations

from io import BytesIO

import pandas as pd
import pytest

from src.data_loader import DataLoadError, EmptyDatasetError, load_dataset
from src.data_profiler import profile_data
from src.database import create_database
from src.query_engine import execute_query
from src.sql_validator import SQLValidationError, is_safe_sql, validate_sql


def test_load_csv_and_make_duplicate_headers_unique():
    frame = load_dataset(b"sales,sales,,\n10,20,x,\n",)
    assert frame.iloc[0].iloc[:3].tolist() == [10, 20, "x"]
    assert pd.isna(frame.iloc[0].iloc[3])
    assert frame.columns.tolist() == ["sales", "sales_2", "column_3", "column_4"]


def test_load_excel_workbook():
    pytest.importorskip("openpyxl")
    workbook = BytesIO()
    pd.DataFrame({"product": ["A"], "sales": [12]}).to_excel(workbook, index=False, engine="openpyxl")
    workbook.name = "sample.xlsx"
    assert load_dataset(workbook).to_dict("records") == [{"product": "A", "sales": 12}]


@pytest.mark.parametrize("contents", [b"", b"name,sales\n"])
def test_empty_dataset_is_rejected(contents):
    with pytest.raises(EmptyDatasetError):
        load_dataset(contents)


def test_unsupported_and_too_large_files_are_rejected():
    with pytest.raises(DataLoadError):
        load_dataset(b"value\n1\n", max_size_mb=0)
    invalid = BytesIO(b"x")
    invalid.name = "sample.txt"
    with pytest.raises(DataLoadError):
        load_dataset(invalid)


def test_profile_missing_duplicates_dates_and_outliers():
    frame = pd.DataFrame(
        {
            "order_date": ["2025-01-01", "not-a-date", "2025-01-01", "2025-01-02"],
            "sales": [10, 10, 11, 1000],
            "category": ["A", None, "A", "B"],
        }
    )
    report = profile_data(frame)
    assert report["rows"] == 4
    assert report["columns"] == 3
    assert report["missing_values"] == 1
    assert report["duplicate_rows"] == 0
    assert report["date_columns"] == ["order_date"]
    assert report["invalid_dates"] == 1
    assert report["statistics"]["sales"]["outlier_count"] == 1
    assert frame.loc[1, "order_date"] == "not-a-date"


def test_profile_counts_duplicate_rows():
    report = profile_data(pd.DataFrame({"value": [1, 1, 2]}))
    assert report["duplicate_rows"] == 1


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE sales;",
        "DELETE FROM sales;",
        "UPDATE sales SET sales = 0;",
        "INSERT INTO sales VALUES (1);",
        "ALTER TABLE sales ADD COLUMN x;",
        "TRUNCATE TABLE sales;",
        "SELECT 1; DELETE FROM sales;",
        "WITH x AS (SELECT 1) DELETE FROM sales;",
        "SELECT load_extension('evil');",
        "PRAGMA table_info(sales);",
    ],
)
def test_dangerous_sql_is_rejected(sql):
    assert not is_safe_sql(sql)
    with pytest.raises(SQLValidationError):
        validate_sql(sql)


def test_select_and_read_only_cte_are_accepted():
    assert validate_sql("SELECT 'delete' AS label FROM dataset;") == "SELECT 'delete' AS label FROM dataset"
    assert is_safe_sql("WITH totals AS (SELECT SUM(sales) AS amount FROM dataset) SELECT amount FROM totals")


def test_query_execution_returns_real_results_and_enforces_read_only(tmp_path):
    path = create_database(pd.DataFrame({"product": ["A", "B"], "sales": [5, 12]}), tmp_path / "data.sqlite")
    result = execute_query(path, "SELECT product, sales FROM dataset ORDER BY sales DESC")
    assert result.status == "success"
    assert result.row_count == 2
    assert result.dataframe.iloc[0].to_dict() == {"product": "B", "sales": 12}

    rejected = execute_query(path, "DELETE FROM dataset")
    assert rejected.status == "error"

    metadata = execute_query(path, "SELECT name FROM sqlite_master")
    assert metadata.status == "error"


def test_query_execution_limits_rows_and_reports_errors(tmp_path):
    path = create_database(pd.DataFrame({"value": [1, 2, 3]}), tmp_path / "data.sqlite")
    result = execute_query(path, "SELECT value FROM dataset", max_rows=2)
    assert result.status == "success"
    assert result.row_count == 2
    assert result.truncated is True

    missing_table = execute_query(path, "SELECT * FROM missing")
    assert missing_table.status == "error"
    assert "no such table" in missing_table.error.lower()