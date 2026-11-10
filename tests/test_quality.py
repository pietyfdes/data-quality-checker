import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import quality as q  # noqa: E402


def issues_for(df, check, **kw):
    return [i for i in q.run_all_checks(df, **kw) if i.check == check]


def test_missing_counts_nulls_and_blank_strings():
    df = pd.DataFrame({"a": [1, None, 3, None], "b": ["x", " ", "y", "z"]})
    summary = q.missing_summary(df).set_index("column")
    assert summary.loc["a", "missing"] == 2
    assert summary.loc["b", "missing"] == 1
    assert summary.loc["a", "missing_%"] == 50.0


def test_duplicates_full_and_key():
    df = pd.DataFrame({"id": [1, 1, 2, 3], "v": ["a", "a", "b", "c"]})
    assert q.check_duplicates(df)[0].count == 1
    df2 = pd.DataFrame({"id": [1, 1, 2], "v": ["a", "b", "c"]})
    res = q.check_duplicates(df2, ["id"])
    assert len(res) == 1 and res[0].count == 1


def test_numbers_stored_as_text_and_mixed():
    df = pd.DataFrame({"n": ["1", "2", "3", "4", "5", "6"], "m": ["1", "2", "3", "4", "5", "oops"]})
    found = {i.column: i for i in q.check_types(df)}
    assert "numbers stored as text" in found["n"].description.lower()
    assert found["m"].count == 1 and "oops" in found["m"].description


def test_dates_stored_as_text():
    df = pd.DataFrame({"when": ["2024-01-01", "2024-02-03", "2024-03-04"]})
    assert any(i.suggestion == "datetime" for i in q.check_types(df))


def test_outliers_iqr_and_multiplier():
    df = pd.DataFrame({"x": [10, 11, 12, 11, 10, 12, 11, 500]})
    assert q.check_outliers(df, 1.5)[0].count == 1
    assert q.check_outliers(df, 1000) == []


def test_inconsistent_categories():
    df = pd.DataFrame({"country": ["USA"] * 5 + ["usa", "U.S.A", "USA ", "Canada", "Canada", "canada"]})
    mapping = q.category_variants(df["country"])
    assert mapping["usa"] == "USA" and mapping["U.S.A"] == "USA" and mapping["USA "] == "USA"
    assert mapping["canada"] == "Canada"
    assert issues_for(df, "Inconsistent categories")


def test_format_validation():
    df = pd.DataFrame({
        "email": ["a@b.com", "bad@", "x y@z.com"],
        "phone": ["+1-555-123-4567", "123", "(555) 123-4567"],
        "signup_date": ["2024-01-31", "2024-02-30", "garbage"],
    })
    by_col = {i.column: i.count for i in q.check_formats(df)}
    assert by_col == {"email": 2, "phone": 1, "signup_date": 2}


def test_invalid_ranges_negative_and_future():
    df = pd.DataFrame({"age": [25, -3, 200, 40], "signup_date": ["2020-01-01", "2999-01-01", "2021-01-01", "2022-01-01"]})
    res = q.check_ranges(df)
    descs = " | ".join(i.description for i in res)
    assert "negative" in descs and "maximum" in descs and "future" in descs


def test_constant_and_high_cardinality():
    df = pd.DataFrame({"c": ["x"] * 25, "id": [f"id{i}" for i in range(25)], "g": ["a", "b"] * 12 + ["a"]})
    msgs = {i.column: i.description for i in q.check_structure(df)}
    assert "Constant" in msgs["c"] and "High cardinality" in msgs["id"] and "g" not in msgs


def test_score_perfect_and_worse_with_problems():
    clean = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": list("uvwxyz")})
    assert q.overall_score(clean, q.run_all_checks(clean)) == 100.0
    dirty = pd.DataFrame({"a": [1, 2, None, None, 5, 6], "b": list("uvwxyz")})
    assert q.overall_score(dirty, q.run_all_checks(dirty)) < 100.0
    assert 0 <= q.overall_score(dirty, q.run_all_checks(dirty)) <= 100


def test_cleaning_functions_do_not_mutate_input():
    df = pd.DataFrame({"c": [" USA", "usa", "UK", None], "n": [1.0, None, 3.0, 400.0]})
    snapshot = df.copy()
    q.standardize_text(df, ["c"], "upper", True)
    q.fill_missing(df, "median", ["n"])
    q.handle_outliers(df, ["n"])
    pd.testing.assert_frame_equal(df, snapshot)


def test_standardize_and_fill_and_drop():
    df = pd.DataFrame({"c": [" USA", "usa", "U.S.A", "UK", "UK"], "n": [1.0, None, 3.0, 5.0, 7.0]})
    out, _ = q.standardize_text(df, ["c"], "upper", True)
    assert set(out["c"]) == {"USA", "UK"}
    filled, _ = q.fill_missing(df, "mean", ["n"])
    assert filled["n"].isna().sum() == 0 and filled["n"].iloc[1] == pytest.approx(4.0)
    dropped, _ = q.fill_missing(df, "drop rows", ["n"])
    assert len(dropped) == 4
    out, _ = q.drop_duplicates(pd.DataFrame({"a": [1, 1, 2]}))
    assert len(out) == 2


def test_cap_and_remove_outliers():
    df = pd.DataFrame({"x": [10, 11, 12, 11, 10, 12, 11, 500]})
    capped, _ = q.handle_outliers(df, ["x"], "cap")
    assert capped["x"].max() < 500 and len(capped) == len(df)
    removed, _ = q.handle_outliers(df, ["x"], "remove")
    assert len(removed) == 7


def test_convert_types():
    df = pd.DataFrame({"n": ["1", "2", "x"], "d": ["2024-01-01", "2024-01-02", "bad"]})
    out, _ = q.convert_types(df, {"n": "numeric", "d": "datetime"})
    assert pd.api.types.is_numeric_dtype(out["n"]) and out["n"].isna().sum() == 1
    assert pd.api.types.is_datetime64_any_dtype(out["d"])


def test_read_table_errors_and_sampling():
    with pytest.raises(ValueError, match="Unsupported"):
        q.read_table(b"abc", "file.txt")
    with pytest.raises(ValueError, match="empty"):
        q.read_table(b"", "a.csv")
    with pytest.raises(ValueError):
        q.read_table(b"a,b\n", "a.csv")
    df, warn = q.read_table(b"a\n" + b"\n".join(str(i).encode() for i in range(50)), "a.csv", max_rows=10)
    assert len(df) == 10 and warn


def test_sample_data_pipeline_and_report():
    df = pd.read_csv(Path(__file__).resolve().parent.parent / "sample_data.csv")
    assert 280 <= len(df) <= 320
    issues = q.run_all_checks(df, ["email"])
    checks = {i.check for i in issues}
    assert set(q.CHECK_ORDER) <= checks
    before = q.overall_score(df, issues)
    cleaned, log = q.apply_cleaning(df, {
        "drop_duplicates": {"subset": None},
        "text": {"columns": ["country", "gender", "name"], "case": "title", "merge_variants": True},
        "convert": {"age": "numeric"},
        "fill": {"strategy": "median", "columns": ["age", "salary"]},
        "outliers": {"columns": ["salary"], "mode": "cap", "multiplier": 1.5},
    })
    after = q.overall_score(cleaned, q.run_all_checks(cleaned, ["email"]))
    assert after > before and len(cleaned) < len(df) and len(log) == 5
    html = q.build_html_report("sample", df, issues, before, log, cleaned, after)
    assert "Data Quality Report" in html and "Dropped" in html
