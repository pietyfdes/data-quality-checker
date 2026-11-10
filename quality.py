"""Data quality checks, scoring, cleaning and reporting.

Pure pandas/numpy code with no Streamlit imports so it can be unit-tested.
All public functions return new objects and never modify their input.
"""
from __future__ import annotations

import difflib
import html
import io
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

MAX_ROWS = 200_000
HIGH_CARDINALITY_RATIO = 0.9
EMAIL_RE = re.compile(r"^[\w.+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)+$")
PHONE_RE = re.compile(r"^\+?[\d\s().-]{7,20}$")
DATE_RE = re.compile(r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}([ T]\d{1,2}:\d{2}(:\d{2})?)?$")

NON_NEGATIVE_TOKENS = {
    "age", "price", "salary", "amount", "quantity", "qty", "count",
    "income", "revenue", "cost", "total", "weight", "height", "years",
}
MAX_VALUES = {"age": 120}
DATE_TOKENS = {"date", "dob", "birthday", "timestamp", "time", "created", "updated"}
FUTURE_OK_TOKENS = {"expiry", "expires", "due", "next", "renewal", "end", "deadline"}
PHONE_TOKENS = {"phone", "mobile", "tel", "telephone", "cell", "fax"}

CHECK_ORDER = [
    "Missing values", "Duplicates", "Data types", "Outliers",
    "Inconsistent categories", "Format validation", "Invalid ranges",
    "Column structure",
]
# Penalty weight: points removed from a column's score per 1% of affected cells.
WEIGHTS = {
    "Missing values": 1.0, "Data types": 1.0, "Format validation": 1.0,
    "Invalid ranges": 1.0, "Outliers": 0.5, "Inconsistent categories": 0.5,
}
FLAT_PENALTY_TEXT_TYPE = 5.0
FLAT_PENALTY_CONSTANT = 5.0
FLAT_PENALTY_HIGH_CARD = 2.0

SCORE_FORMULA = (
    "**Column score** = 100 − Σ penalties, floored at 0. A penalty is "
    "`weight × % of affected cells`: missing 1.0, invalid format 1.0, invalid range 1.0, "
    "mixed types 1.0, outliers 0.5, inconsistent categories 0.5. "
    "Numbers/dates stored as text cost a flat 5 points, constant columns 5, "
    "high-cardinality columns 2.\n\n"
    "**Overall score** = average column score − % of duplicate rows (1 point per 1%)."
)


@dataclass
class Issue:
    check: str
    column: str | None
    severity: str  # High / Medium / Low
    count: int
    description: str
    rows: pd.Index = field(default_factory=lambda: pd.Index([]))
    penalty: float = 0.0
    suggestion: str | None = None  # e.g. "numeric" / "datetime" for type issues


# ----------------------------------------------------------------- helpers
def _tokens(name: str) -> set[str]:
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", str(name))
    return {t for t in re.split(r"[^a-z]+", spaced.lower()) if t}


def is_text(s: pd.Series) -> bool:
    return bool(pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s))


def _sev(frac: float, high: float, med: float) -> str:
    return "High" if frac >= high else "Medium" if frac >= med else "Low"


def _penalty(check: str, count: int, n: int) -> float:
    return WEIGHTS[check] * 100.0 * count / n if n else 0.0


def missing_mask(df: pd.DataFrame) -> pd.DataFrame:
    """True where a cell is NA or a blank/whitespace-only string."""
    mask = df.isna()
    for col in df.columns:
        if is_text(df[col]):
            mask[col] = mask[col] | df[col].astype("string").str.strip().eq("").fillna(False)
    return mask


def _parse_dates(s: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(s, format="mixed", errors="coerce")


def numeric_view(s: pd.Series) -> pd.Series | None:
    """Series as floats if numeric, or text where >=80% of values parse as numbers."""
    if pd.api.types.is_bool_dtype(s):
        return None
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    if not is_text(s):
        return None
    clean = s.dropna().astype("string").str.strip()
    clean = clean[clean != ""]
    if clean.empty:
        return None
    num = _to_numeric_text(s)
    if num.loc[clean.index].notna().mean() >= 0.8:
        return num
    return None


def _to_numeric_text(s: pd.Series) -> pd.Series:
    cleaned = s.astype("string").str.replace(r"[,$€£\s]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def column_kind(name: str, s: pd.Series) -> str | None:
    """Guess 'email', 'phone' or 'date' from the column name, then the content."""
    tokens = _tokens(name)
    if "email" in tokens or "mail" in tokens:
        return "email"
    if tokens & PHONE_TOKENS:
        return "phone"
    if tokens & DATE_TOKENS or (tokens and list(re.split(r"[^a-z]+", name.lower()))[-1] == "at"):
        return "date"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "date"
    if not is_text(s) or numeric_view(s) is not None:
        return None
    vals = s.dropna().astype("string").str.strip()
    vals = vals[vals != ""].head(500)
    if vals.empty:
        return None
    if vals.map(lambda v: bool(EMAIL_RE.match(v))).mean() >= 0.6:
        return "email"
    if vals.map(lambda v: bool(DATE_RE.match(v))).mean() >= 0.8:
        return "date"
    return None


# ------------------------------------------------------------------ checks
def check_missing(df: pd.DataFrame) -> list[Issue]:
    mask, n = missing_mask(df), len(df)
    issues = []
    for col in df.columns:
        cnt = int(mask[col].sum())
        if cnt:
            pct = cnt / n
            issues.append(Issue(
                "Missing values", col, _sev(pct, 0.2, 0.05), cnt,
                f"{cnt} missing values ({pct:.1%})", df.index[mask[col]],
                _penalty("Missing values", cnt, n),
            ))
    return issues


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    mask = missing_mask(df)
    counts = mask.sum()
    pct = (counts / len(df) * 100).round(2) if len(df) else counts * 0.0
    return pd.DataFrame({"column": df.columns, "missing": counts.values, "missing_%": pct.values})


def check_duplicates(df: pd.DataFrame, key_cols: list[str] | None = None) -> list[Issue]:
    n, issues = len(df), []
    full = int(df.duplicated().sum())
    if full:
        issues.append(Issue(
            "Duplicates", None, _sev(full / n, 0.05, 0.01), full,
            f"{full} fully duplicated rows ({full / n:.1%}); all copies shown",
            df.index[df.duplicated(keep=False)],
        ))
    if key_cols:
        keyed = int(df.duplicated(subset=key_cols).sum())
        if keyed:
            issues.append(Issue(
                "Duplicates", None, _sev(keyed / n, 0.05, 0.01), keyed,
                f"{keyed} rows repeat the key ({', '.join(key_cols)})",
                df.index[df.duplicated(subset=key_cols, keep=False)],
            ))
    return issues


def check_types(df: pd.DataFrame) -> list[Issue]:
    issues = []
    for col in df.columns:
        s = df[col]
        if s.dtype == object and s.dropna().map(type).nunique() > 1:
            issues.append(Issue(
                "Data types", col, "Medium", int(s.notna().sum()),
                "Mixed Python types in one column: "
                + ", ".join(sorted(s.dropna().map(lambda v: type(v).__name__).unique())),
                df.index[s.notna()],
            ))
            continue
        if not is_text(s):
            continue
        present = s.dropna().astype("string").str.strip()
        present = present[present != ""]
        if present.empty:
            continue
        num = _to_numeric_text(s).loc[present.index]
        rate = num.notna().mean()
        if rate >= 0.8:
            bad = present.index[num.isna()]
            if len(bad) == 0:
                issues.append(Issue(
                    "Data types", col, "Low", len(present),
                    "Numbers stored as text", df.index[:0], FLAT_PENALTY_TEXT_TYPE, "numeric",
                ))
            else:
                issues.append(Issue(
                    "Data types", col, _sev(len(bad) / len(df), 0.05, 0.01), len(bad),
                    f"Mixed types: mostly numeric, but {len(bad)} values are not numbers "
                    f"(e.g. {', '.join(map(repr, present.loc[bad].unique()[:3]))})",
                    bad, _penalty("Data types", len(bad), len(df)) + FLAT_PENALTY_TEXT_TYPE, "numeric",
                ))
        elif column_kind(col, s) == "date" and \
                present.map(lambda v: bool(DATE_RE.match(v))).mean() >= 0.8:
            parsed = _parse_dates(present)
            if parsed.notna().mean() >= 0.8:
                issues.append(Issue(
                    "Data types", col, "Low", len(present),
                    "Dates stored as text", df.index[:0], FLAT_PENALTY_TEXT_TYPE, "datetime",
                ))
    return issues


def iqr_bounds(num: pd.Series, multiplier: float) -> tuple[float, float] | None:
    vals = num.dropna()
    if len(vals) < 4 or vals.nunique() <= 2:
        return None
    q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return None
    return float(q1 - multiplier * iqr), float(q3 + multiplier * iqr)


def check_outliers(df: pd.DataFrame, multiplier: float = 1.5) -> list[Issue]:
    issues = []
    for col in df.columns:
        num = numeric_view(df[col])
        if num is None:
            continue
        bounds = iqr_bounds(num, multiplier)
        if bounds is None:
            continue
        lo, hi = bounds
        out = (num < lo) | (num > hi)
        cnt = int(out.sum())
        if cnt:
            issues.append(Issue(
                "Outliers", col, _sev(cnt / len(df), 0.05, 0.02), cnt,
                f"{cnt} values outside [{lo:,.2f}, {hi:,.2f}] (IQR × {multiplier})",
                df.index[out], _penalty("Outliers", cnt, len(df)),
            ))
    return issues


def _norm_key(v: str) -> str:
    return re.sub(r"[^a-z0-9]", "", v.lower())


def category_variants(s: pd.Series) -> dict[str, str]:
    """Map every variant spelling to the most frequent spelling in its group."""
    vals = s.dropna().astype("string")
    vals = vals[vals.str.strip() != ""]
    counts = vals.value_counts()
    groups: dict[str, list[str]] = {}
    for raw in counts.index:
        groups.setdefault(_norm_key(raw.strip()), []).append(raw)
    keys = list(groups)
    parent = {k: k for k in keys}

    def find(k: str) -> str:
        while parent[k] != k:
            k = parent[k]
        return k

    for k in keys:  # merge near-duplicate keys (typos) using fuzzy matching
        if len(k) < 5:
            continue
        for m in difflib.get_close_matches(k, [x for x in keys if x != k], n=3, cutoff=0.88):
            if abs(len(m) - len(k)) <= 2:
                parent[find(m)] = find(k)
    merged: dict[str, list[str]] = {}
    for k in keys:
        merged.setdefault(find(k), []).extend(groups[k])
    mapping = {}
    for members in merged.values():
        if len(members) > 1:
            canonical = max(members, key=lambda r: (counts[r], r == r.strip()))
            canonical = canonical.strip()
            for m in members:
                if m != canonical:
                    mapping[m] = canonical
    return mapping


def check_categories(df: pd.DataFrame) -> list[Issue]:
    issues = []
    for col in df.columns:
        s = df[col]
        if not is_text(s) or numeric_view(s) is not None or column_kind(col, s) is not None:
            continue
        nun = s.nunique()
        if nun < 2 or nun > min(60, max(10, len(df) * 0.2)):
            continue
        mapping = category_variants(s)
        if not mapping:
            continue
        affected = s.isin(list(mapping))
        cnt = int(affected.sum())
        examples = "; ".join(f"{k!r} → {v!r}" for k, v in list(mapping.items())[:4])
        issues.append(Issue(
            "Inconsistent categories", col, "Medium", cnt,
            f"{len(mapping)} variant spellings (case, spaces, punctuation or typos): {examples}",
            df.index[affected], _penalty("Inconsistent categories", cnt, len(df)),
        ))
    return issues


def _format_invalid(kind: str, s: pd.Series) -> pd.Series:
    """Boolean Series, True where a non-blank value has an invalid format."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.Series(False, index=s.index)
    txt = s.astype("string").str.strip()
    present = txt.notna() & (txt != "")
    if kind == "email":
        ok = txt.map(lambda v: bool(EMAIL_RE.match(v)) if isinstance(v, str) else False)
    elif kind == "phone":
        ok = txt.map(lambda v: bool(PHONE_RE.match(v)) and 7 <= sum(c.isdigit() for c in v) <= 15
                     if isinstance(v, str) else False)
    else:
        parsed = _parse_dates(txt.where(present))
        ok = txt.map(lambda v: bool(DATE_RE.match(v)) if isinstance(v, str) else False) & parsed.notna()
    return present & ~ok.astype(bool)


def check_formats(df: pd.DataFrame) -> list[Issue]:
    issues = []
    for col in df.columns:
        kind = column_kind(col, df[col])
        if kind is None or pd.api.types.is_numeric_dtype(df[col]):
            continue
        bad = _format_invalid(kind, df[col])
        cnt = int(bad.sum())
        if cnt:
            label = {"email": "email addresses", "phone": "phone numbers", "date": "dates (incl. impossible dates)"}[kind]
            ex = ", ".join(map(repr, df.loc[bad, col].astype(str).unique()[:3]))
            issues.append(Issue(
                "Format validation", col, _sev(cnt / len(df), 0.05, 0.01), cnt,
                f"{cnt} invalid {label} (e.g. {ex})", df.index[bad],
                _penalty("Format validation", cnt, len(df)),
            ))
    return issues


def check_ranges(df: pd.DataFrame, today: datetime | None = None) -> list[Issue]:
    today = today or datetime.now()
    issues = []
    for col in df.columns:
        tokens = _tokens(col)
        num = numeric_view(df[col])
        if num is not None and tokens & NON_NEGATIVE_TOKENS:
            neg = num < 0
            if neg.sum():
                issues.append(Issue(
                    "Invalid ranges", col, "High", int(neg.sum()),
                    f"{int(neg.sum())} negative values where negatives are not allowed",
                    df.index[neg], _penalty("Invalid ranges", int(neg.sum()), len(df)),
                ))
            for tok, cap in MAX_VALUES.items():
                if tok in tokens:
                    big = num > cap
                    if big.sum():
                        issues.append(Issue(
                            "Invalid ranges", col, "High", int(big.sum()),
                            f"{int(big.sum())} values above the plausible maximum of {cap}",
                            df.index[big], _penalty("Invalid ranges", int(big.sum()), len(df)),
                        ))
        if column_kind(col, df[col]) == "date" and not tokens & FUTURE_OK_TOKENS:
            dates = df[col] if pd.api.types.is_datetime64_any_dtype(df[col]) else _parse_dates(df[col])
            fut = dates > pd.Timestamp(today)
            if fut.sum():
                issues.append(Issue(
                    "Invalid ranges", col, "High", int(fut.sum()),
                    f"{int(fut.sum())} dates in the future", df.index[fut],
                    _penalty("Invalid ranges", int(fut.sum()), len(df)),
                ))
    return issues


def check_structure(df: pd.DataFrame) -> list[Issue]:
    issues = []
    for col in df.columns:
        s = df[col]
        nun = s.nunique(dropna=True)
        if len(df) > 1 and nun <= 1:
            val = "all values missing" if nun == 0 else f"always {s.dropna().iloc[0]!r}"
            issues.append(Issue("Column structure", col, "Low", len(df),
                                f"Constant column ({val})", df.index[:0], FLAT_PENALTY_CONSTANT))
        elif len(df) >= 20 and is_text(s) and nun / s.notna().sum() > HIGH_CARDINALITY_RATIO:
            issues.append(Issue("Column structure", col, "Low", nun,
                                f"High cardinality: {nun} distinct values (likely an identifier or free text)",
                                df.index[:0], FLAT_PENALTY_HIGH_CARD))
    return issues


def run_all_checks(df: pd.DataFrame, key_cols: list[str] | None = None,
                   multiplier: float = 1.5) -> list[Issue]:
    issues = (check_missing(df) + check_duplicates(df, key_cols) + check_types(df)
              + check_outliers(df, multiplier) + check_categories(df) + check_formats(df)
              + check_ranges(df) + check_structure(df))
    return sorted(issues, key=lambda i: CHECK_ORDER.index(i.check))


# ----------------------------------------------------------------- scoring
def column_scores(df: pd.DataFrame, issues: list[Issue]) -> pd.DataFrame:
    pen = {c: 0.0 for c in df.columns}
    for i in issues:
        if i.column in pen:
            pen[i.column] += i.penalty
    return pd.DataFrame({"column": list(pen), "score": [round(max(0.0, 100 - p), 1) for p in pen.values()]})


def overall_score(df: pd.DataFrame, issues: list[Issue]) -> float:
    if df.empty or df.shape[1] == 0:
        return 0.0
    cols = column_scores(df, issues)["score"].mean()
    dup_pct = df.duplicated().sum() / len(df) * 100
    return round(float(max(0.0, min(100.0, cols - dup_pct))), 1)


def profile(df: pd.DataFrame) -> dict[str, float]:
    cells = df.size or 1
    return {
        "rows": len(df), "columns": df.shape[1],
        "missing_pct": round(float(missing_mask(df).values.sum()) / cells * 100, 2),
        "duplicates": int(df.duplicated().sum()),
    }


# ------------------------------------------------------------------ loading
def read_table(data: bytes, filename: str, max_rows: int = MAX_ROWS) -> tuple[pd.DataFrame, str | None]:
    """Parse CSV/Excel bytes. Returns (dataframe, warning). Raises ValueError on bad input."""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext not in {"csv", "xlsx", "xls"}:
        raise ValueError(f"Unsupported file type '.{ext}'. Please upload a CSV or Excel (.xlsx) file.")
    if not data:
        raise ValueError("The file is empty.")
    try:
        df = pd.read_csv(io.BytesIO(data)) if ext == "csv" else pd.read_excel(io.BytesIO(data))
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The file is empty.") from exc
    except Exception as exc:  # noqa: BLE001 - surface any parser failure as a friendly error
        raise ValueError(f"Could not read the file: {exc}") from exc
    if df.empty or df.shape[1] == 0:
        raise ValueError("The file contains no data rows.")
    warn = None
    if len(df) > max_rows:
        warn = f"Large file ({len(df):,} rows): a random sample of {max_rows:,} rows is used."
        df = df.sample(max_rows, random_state=42).sort_index()
    return df, warn


# ----------------------------------------------------------------- cleaning
def drop_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> tuple[pd.DataFrame, str]:
    out = df.drop_duplicates(subset=subset or None)
    where = f"on key ({', '.join(subset)})" if subset else "(full rows)"
    return out, f"Dropped {len(df) - len(out)} duplicate rows {where}"


def standardize_text(df: pd.DataFrame, columns: list[str], case: str = "keep",
                     merge_variants: bool = False) -> tuple[pd.DataFrame, str]:
    out, changed = df.copy(), 0
    for col in columns:
        if col not in out or not is_text(out[col]):
            continue
        s = out[col].astype("string").str.strip()
        s = s.mask(s == "")
        if case in {"lower", "upper", "title"}:
            s = getattr(s.str, case)()
        if merge_variants:
            s = s.replace(category_variants(s))
        changed += int((s.fillna("") != df[col].astype("string").fillna("")).sum())
        out[col] = s
    parts = ["trimmed whitespace"] + ([f"{case} case"] if case != "keep" else []) \
        + (["merged variant spellings"] if merge_variants else [])
    return out, f"Text cleanup ({', '.join(parts)}) on {len(columns)} columns; {changed} cells changed"


def convert_types(df: pd.DataFrame, conversions: dict[str, str]) -> tuple[pd.DataFrame, str]:
    out, notes = df.copy(), []
    for col, target in conversions.items():
        if col not in out:
            continue
        before_na = int(out[col].isna().sum())
        if target == "numeric":
            out[col] = _to_numeric_text(out[col]) if is_text(out[col]) else pd.to_numeric(out[col], errors="coerce")
        elif target == "datetime":
            out[col] = _parse_dates(out[col])
        elif target == "text":
            out[col] = out[col].astype("string")
        else:
            continue
        lost = int(out[col].isna().sum()) - before_na
        notes.append(f"{col}→{target}" + (f" ({lost} unparseable → missing)" if lost > 0 else ""))
    return out, "Converted types: " + ", ".join(notes)


def fill_missing(df: pd.DataFrame, strategy: str, columns: list[str],
                 constant: str = "") -> tuple[pd.DataFrame, str]:
    out = df.copy()
    for col in columns:  # blank strings count as missing
        if is_text(out[col]):
            s = out[col].astype("string")
            out[col] = s.mask(s.str.strip() == "")
    if strategy == "drop rows":
        res = out.dropna(subset=columns)
        return res, f"Dropped {len(out) - len(res)} rows with missing values in {len(columns)} columns"
    filled, skipped = 0, []
    for col in columns:
        miss = out[col].isna()
        if not miss.any():
            continue
        value: Any = None
        if strategy in {"mean", "median"}:
            num = numeric_view(out[col])
            if num is None:
                skipped.append(col)
                continue
            out[col] = num
            value = num.mean() if strategy == "mean" else num.median()
        elif strategy == "mode":
            m = out[col].mode(dropna=True)
            value = m.iloc[0] if len(m) else None
        elif strategy == "constant":
            value = constant
            if pd.api.types.is_numeric_dtype(out[col]):
                try:
                    value = float(constant)
                except ValueError:
                    skipped.append(col)
                    continue
        if value is None or (isinstance(value, float) and np.isnan(value)):
            skipped.append(col)
            continue
        out[col] = out[col].fillna(value)
        filled += int(miss.sum())
    msg = f"Filled {filled} missing values using {strategy}"
    if skipped:
        msg += f" (skipped non-applicable: {', '.join(skipped)})"
    return out, msg


def handle_outliers(df: pd.DataFrame, columns: list[str], mode: str = "cap",
                    multiplier: float = 1.5) -> tuple[pd.DataFrame, str]:
    out, touched = df.copy(), 0
    drop_mask = pd.Series(False, index=out.index)
    for col in columns:
        num = numeric_view(out[col])
        bounds = iqr_bounds(num, multiplier) if num is not None else None
        if bounds is None:
            continue
        lo, hi = bounds
        mask = (num < lo) | (num > hi)
        touched += int(mask.sum())
        if mode == "cap":
            out[col] = num.clip(lo, hi)
        else:
            drop_mask |= mask.fillna(False)
    if mode == "remove":
        out = out[~drop_mask]
        return out, f"Removed {int(drop_mask.sum())} rows containing outliers (IQR × {multiplier})"
    return out, f"Capped {touched} outlier values to IQR bounds (× {multiplier})"


def apply_cleaning(df: pd.DataFrame, options: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    """Run the chosen cleaning steps in a safe order on a copy of ``df``."""
    out, log = df.copy(), []
    if opt := options.get("text"):
        out, msg = standardize_text(out, opt["columns"], opt.get("case", "keep"), opt.get("merge_variants", False))
        log.append(msg)
    if opt := options.get("convert"):
        out, msg = convert_types(out, opt)
        log.append(msg)
    if opt := options.get("drop_duplicates"):
        out, msg = drop_duplicates(out, opt.get("subset"))
        log.append(msg)
    if opt := options.get("fill"):
        out, msg = fill_missing(out, opt["strategy"], opt["columns"], opt.get("constant", ""))
        log.append(msg)
    if opt := options.get("outliers"):
        out, msg = handle_outliers(out, opt["columns"], opt.get("mode", "cap"), opt.get("multiplier", 1.5))
        log.append(msg)
    return out, log


# ------------------------------------------------------------------- report
def build_html_report(source: str, before: pd.DataFrame, issues: list[Issue], score_before: float,
                      actions: list[str], after: pd.DataFrame | None = None,
                      score_after: float | None = None) -> str:
    esc = html.escape
    rows = "".join(
        f"<tr><td>{esc(i.check)}</td><td>{esc(str(i.column or '(rows)'))}</td>"
        f"<td class='sev {i.severity}'>{i.severity}</td><td>{i.count}</td><td>{esc(i.description)}</td></tr>"
        for i in issues
    ) or "<tr><td colspan='5'>No issues found 🎉</td></tr>"
    acts = "".join(f"<li>{esc(a)}</li>" for a in actions) or "<li>No cleaning actions applied.</li>"
    after_html = ""
    if after is not None and score_after is not None:
        after_html = (f"<p>After cleaning: <b>{score_after}</b> / 100 &middot; {len(after):,} rows "
                      f"(before: {len(before):,})</p>")
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Data Quality Report</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;background:#0e1117;color:#fafafa;margin:2rem auto;max-width:1000px;padding:0 1rem}}
h1,h2{{color:#ff4b4b}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #333;padding:6px 10px;text-align:left;vertical-align:top}}
th{{background:#262730}}.High{{color:#ff4b4b;font-weight:700}}.Medium{{color:#ffa421;font-weight:700}}.Low{{color:#21c354;font-weight:700}}
.score{{font-size:2.5rem;font-weight:700}}</style></head><body>
<h1>📊 Data Quality Report</h1>
<p>Source: <b>{esc(source)}</b> &middot; Generated {datetime.now():%Y-%m-%d %H:%M}</p>
<h2>Summary</h2><p class="score">{score_before} / 100</p>
<p>{len(before):,} rows &times; {before.shape[1]} columns &middot; {len(issues)} issues found</p>{after_html}
<h2>Issues found</h2><table><tr><th>Check</th><th>Column</th><th>Severity</th><th>Affected</th><th>Details</th></tr>{rows}</table>
<h2>Actions applied</h2><ul>{acts}</ul>
<h2>Scoring formula</h2><p>{esc(SCORE_FORMULA).replace(chr(10), '<br>')}</p>
</body></html>"""
