from __future__ import annotations

import json

import pandas as pd
import pytest

from src.answer_generator import INSUFFICIENT_DATA_ANSWER, generate_answer
from src.ai_provider import OpenAICompatibleProvider, create_chat_provider
from src.config import AIConfig
from src.database import create_database
from src.query_engine import execute_query
from src.sql_generator import build_schema, generate_sql
from src.sql_validator import SQLValidationError, validate_sql


class FixedProvider:
    def __init__(self, completion):
        self.completion = completion
        self.prompts = []

    def complete(self, system_prompt, user_prompt):
        self.prompts.append((system_prompt, user_prompt))
        if isinstance(self.completion, Exception):
            raise self.completion
        return self.completion


@pytest.fixture
def sales_frame():
    return pd.DataFrame(
        {
            "order_date": pd.to_datetime(
                ["2025-01-03", "2025-01-16", "2025-02-02", "2025-02-12", "2025-03-05", "2025-03-14"]
            ),
            "product": ["A", "A", "B", "B", "C", "C"],
            "city": ["NY", "NY", "LA", "NY", "LA", "LA"],
            "customer_id": ["C1", "C2", "C1", "C3", "C4", "C4"],
            "sales": [100, 50, 200, 25, 70, 30],
            "profit": [20, 10, 30, 5, -2, -1],
        }
    )


def test_missing_model_uses_no_provider(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "ollama")
    monkeypatch.delenv("AI_MODEL", raising=False)
    monkeypatch.delenv("AI_API_KEY", raising=False)

    assert create_chat_provider() is None


def test_local_provider_needs_model_but_not_api_key(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "ollama")
    monkeypatch.setenv("AI_MODEL", "local-analyst")
    monkeypatch.delenv("AI_API_KEY", raising=False)

    provider = create_chat_provider()
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.model == "local-analyst"
    assert provider.api_key is None


def test_remote_provider_requires_an_api_key():
    config = AIConfig(
        provider="openai_compatible",
        base_url="https://api.example.test/v1",
        model="analyst-model",
    )
    assert create_chat_provider(config) is None


def test_openai_compatible_provider_sends_configured_model(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"SELECT 1"}}]}'

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("src.ai_provider.urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider(
        base_url="http://localhost:11434/v1",
        model="local-analyst",
        timeout_seconds=7,
        api_key="test-key",
    )

    assert provider.complete("system", "question") == "SELECT 1"
    request = captured["request"]
    body = json.loads(request.data)
    assert request.full_url == "http://localhost:11434/v1/chat/completions"
    assert body["model"] == "local-analyst"
    assert body["temperature"] == 0
    assert captured["timeout"] == 7


def test_rule_fallback_generates_valid_sql_and_executes(sales_frame, tmp_path, monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "none")
    schema = build_schema(sales_frame)
    generated = generate_sql("Which product generated the highest sales?", schema)

    assert generated.success
    assert generated.source == "rules"
    assert validate_sql(generated.sql) == generated.sql

    database = create_database(sales_frame, tmp_path / "sales.sqlite")
    query_result = execute_query(database, generated.sql)
    assert query_result.status == "success"
    assert query_result.dataframe.iloc[0].to_dict() == {"product": "B", "total_sales": 225}


def test_rule_fallback_generates_monthly_sales_trend(sales_frame, tmp_path, monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "none")
    generated = generate_sql("Show monthly sales trend", build_schema(sales_frame))
    assert generated.success
    assert "strftime('%Y-%m'" in generated.sql

    database = create_database(sales_frame, tmp_path / "sales.sqlite")
    query_result = execute_query(database, generated.sql)
    assert query_result.status == "success"
    assert query_result.dataframe.to_dict("records") == [
        {"month": "2025-01", "total_sales": 150},
        {"month": "2025-02", "total_sales": 225},
        {"month": "2025-03", "total_sales": 100},
    ]


def test_rule_fallback_handles_lowest_category_and_distinct_count(sales_frame, monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "none")
    schema = build_schema(sales_frame)
    lowest = generate_sql("Which city has the lowest profit?", schema)
    count = generate_sql("How many customers are there?", schema)

    assert lowest.success and 'ORDER BY "total_profit" ASC LIMIT 1' in lowest.sql
    assert count.success and 'COUNT(DISTINCT "customer_id")' in count.sql


def test_unsupported_question_does_not_guess(sales_frame, monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "none")
    result = generate_sql("What is the average temperature on Mars?", build_schema(sales_frame))
    assert not result.success
    assert "numeric measure" in result.error


def test_sql_generation_falls_back_when_provider_has_no_model(sales_frame, monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "ollama")
    monkeypatch.delenv("AI_MODEL", raising=False)
    monkeypatch.delenv("AI_API_KEY", raising=False)

    result = generate_sql("Which product generated the highest sales?", sales_frame)
    assert result.success
    assert result.source == "rules"


def test_model_sql_is_validated_and_dangerous_sql_is_rejected(sales_frame):
    schema = build_schema(sales_frame)
    valid = FixedProvider('SELECT "product", SUM("sales") AS "total_sales" FROM "dataset" GROUP BY "product" ORDER BY "total_sales" DESC LIMIT 1')
    safe_result = generate_sql("Which product generated the highest sales?", schema, provider=valid)
    assert safe_result.success
    assert safe_result.source == "model"
    validate_sql(safe_result.sql)

    malicious = FixedProvider("DROP TABLE dataset;")
    rejected = generate_sql("Which product generated the highest sales?", schema, provider=malicious)
    assert not rejected.success
    assert rejected.sql is None
    assert "rejected" in rejected.error
    with pytest.raises(SQLValidationError):
        validate_sql("DROP TABLE dataset;")


def test_answer_is_grounded_in_executed_results(sales_frame, tmp_path, monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "none")
    generated = generate_sql("Which product generated the highest sales?", sales_frame)
    database = create_database(sales_frame, tmp_path / "sales.sqlite")
    query_result = execute_query(database, generated.sql)
    answer = generate_answer("Which product generated the highest sales?", query_result)

    assert "product = B" in answer
    assert "total_sales = 225" in answer
    assert generate_answer("unknown", pd.DataFrame()) == INSUFFICIENT_DATA_ANSWER


def test_answer_provider_receives_actual_query_results(sales_frame, tmp_path):
    database = create_database(sales_frame, tmp_path / "sales.sqlite")
    result = execute_query(database, 'SELECT "product", SUM("sales") AS "sales_total" FROM "dataset" GROUP BY "product" ORDER BY "sales_total" DESC LIMIT 1')
    provider = FixedProvider("Product B returned the highest query result: 225 sales.")

    answer = generate_answer("Which product generated the highest sales?", result, provider=provider)
    assert answer == provider.completion
    payload = json.loads(provider.prompts[0][1].split("\n", 1)[1])
    assert payload["result_rows"] == [{"product": "B", "sales_total": 225}]