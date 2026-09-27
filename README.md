# AI Data Analyst

AI Data Analyst is a Streamlit application for exploring tabular data with natural-language questions. Upload a CSV or Excel workbook, inspect data quality, ask an analytical question, and receive validated SQL, a result table, a grounded answer, and an appropriate chart. A bundled sales dataset makes the application usable immediately after installation.

## Features

- CSV, XLSX, and XLS upload with a 50 MB limit and validation
- Sample sales dataset for a quick demonstration
- Dataset profiling for missing values, duplicate rows, dates, types, and numeric outliers
- Natural-language questions translated into read-only SQLite SQL
- Deterministic local SQL and answer fallbacks that work without an external model
- Optional Ollama or OpenAI-compatible model integration
- SQL validation and read-only, resource-bounded query execution
- Automatic charts and exports for query results, including CSV, Excel, PNG, and a report archive
- Dashboard, data-health, and executive-summary views

## Tech Stack

- Python 3.10+
- Streamlit for the application interface
- pandas for tabular data processing
- SQLite for temporary local analytical databases
- Plotly and Kaleido for charts and PNG export
- openpyxl and xlrd for Excel support
- pytest for automated tests

## Architecture

```text
app.py                    Streamlit entry point and workflow orchestration
src/
  data_loader.py          CSV/Excel loading and validation
  data_profiler.py        Data quality profiling
  database.py             Temporary SQLite database creation
  sql_generator.py        Rule-based and optional model-assisted SQL generation
  sql_validator.py        Read-only SQL validation
  query_engine.py         Resource-bounded SQLite execution
  answer_generator.py     Grounded natural-language answers
  chart_generator.py      Plotly chart selection and creation
  export_utils.py         CSV, Excel, PNG, and report exports
  ai_provider.py          OpenAI-compatible provider client
  config.py               Environment-backed AI configuration
sample_data/
  sample_sales.csv        Demonstration dataset
tests/                    Automated test suite
```

## Installation

1. Clone the repository and enter its directory.
2. Create and activate a virtual environment.
3. Install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

## Run the App

```powershell
streamlit run app.py
```

The application starts in local rules mode by default, so no API key or model server is required. Open the local URL printed by Streamlit, then select **Load sample sales** or upload your own data.

## Use CSV or Excel Data

1. Open the **Ask Your Data** workspace.
2. Use **Upload CSV or Excel** in the sidebar and choose a `.csv`, `.xlsx`, or `.xls` file.
3. Review the dataset preview and data-health information.
4. Enter a question such as `Which product generated the highest sales?` and submit it.
5. Review the answer, generated SQL, results, chart, and available downloads.

Uploads must contain a header row and at least one data row. Duplicate or blank column headers are assigned deterministic names. The source file is not modified.

## Natural-Language-to-SQL Flow

1. The uploaded frame is profiled and copied into a temporary SQLite database.
2. A schema containing the table and column metadata is built.
3. If an AI provider is configured, the schema and question are sent to an OpenAI-compatible chat-completions endpoint. Otherwise, deterministic local rules are used.
4. The generated statement is limited to a single read-only `SELECT` query and is validated before execution.
5. The query runs against a read-only SQLite connection with row, time, and VM-step limits.
6. The executed result is used to create the answer, chart, and export artifacts. Model-generated answers are grounded in returned rows and fall back to deterministic wording when unavailable.

### Optional Model Configuration

Copy `.env.example` as a reference, then provide these variables through your environment when using a model:

```text
AI_PROVIDER=ollama
AI_BASE_URL=http://localhost:11434/v1
AI_MODEL=your-model-name
AI_API_KEY=
AI_TIMEOUT_SECONDS=30
```

`AI_PROVIDER` may be `none`, `ollama`, or `openai_compatible`. Ollama is local and does not require an API key. Remote OpenAI-compatible endpoints require `AI_API_KEY`. Keep real credentials outside the repository.

## Testing

Run the complete suite from the repository root:

```powershell
python -m pytest
```

The tests cover loading and profiling, SQL generation and validation, query execution, charting, exports, provider fallback, and the application flow.

## Future Improvements

- Add authentication and per-user dataset/session isolation for hosted deployments.
- Add more configurable query and upload limits for larger deployments.
- Expand natural-language intent coverage and provider-specific configuration options.
- Add CI workflows for supported Python versions and coverage reporting.
- Add persistent, opt-in storage for saved analyses and dashboards.

## License

No license has been specified yet.