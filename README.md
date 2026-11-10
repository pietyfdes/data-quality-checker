# 🧪 Data Quality Checker

A Streamlit app that profiles a CSV/Excel dataset, finds data quality problems, lets you
clean them with a few clicks and exports the cleaned data plus an HTML quality report.

## Features
- Missing values, duplicate rows (full-row and by key), type issues (numbers/dates stored as text, mixed types)
- IQR outlier detection with adjustable multiplier
- Inconsistent categories (case, whitespace, `USA` vs `U.S.A`, typos)
- Email / phone / date format validation, negative values, future dates
- Constant and high-cardinality columns
- Overall and per-column Data Quality Score (0-100, formula shown in the app)
- Cleaning: drop duplicates, trim/standardize text, fill missing values, cap/remove outliers, convert types
- Downloads: cleaned CSV, cleaned Excel, HTML report

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```
Upload a file, or click **Use sample data** (`sample_data.csv`, ~300 deliberately messy rows).
Files above 200,000 rows are randomly sampled.

## Test
```bash
pip install -r requirements-dev.txt
pytest
```

## Layout
- `app.py` – Streamlit UI
- `quality.py` – checks, scoring, cleaning, report (no Streamlit imports)
- `tests/` – pytest tests (including a Streamlit `AppTest` smoke test)
