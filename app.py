"""Data Quality Checker - Streamlit UI (all checks live in quality.py)."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import quality as q

SAMPLE_PATH = Path(__file__).parent / "sample_data.csv"
SEVERITY_COLOR = {"High": "red", "Medium": "orange", "Low": "green"}

st.set_page_config(page_title="Data Quality Checker", page_icon="🧪", layout="wide")


# ------------------------------------------------------------------ loading
@st.cache_data(show_spinner="Reading file…")
def load_upload(data: bytes, filename: str) -> tuple[pd.DataFrame, str | None]:
    return q.read_table(data, filename)


@st.cache_data
def load_sample() -> tuple[pd.DataFrame, str | None]:
    return q.read_table(SAMPLE_PATH.read_bytes(), SAMPLE_PATH.name)


@st.cache_data(show_spinner="Running checks…")
def cached_checks(df: pd.DataFrame, key_cols: tuple[str, ...], multiplier: float) -> list[q.Issue]:
    return q.run_all_checks(df, list(key_cols), multiplier)


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    out = df.copy()
    for col in out.select_dtypes(include=["datetimetz"]).columns:
        out[col] = out[col].dt.tz_localize(None)
    out.to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


def init_state() -> None:
    for key, default in {"original": None, "cleaned": None, "actions": [], "source": "", "warning": None}.items():
        st.session_state.setdefault(key, default)


def set_dataset(df: pd.DataFrame, source: str, warning: str | None) -> None:
    st.session_state.update(original=df.copy(), cleaned=None, actions=[], source=source, warning=warning)


# ------------------------------------------------------------------ sidebar
def render_sidebar() -> None:
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("Upload a CSV or Excel file to profile it, spot data problems, "
                 "clean them and export a report. Your original upload is never modified.")
        st.header("✨ Features")
        st.markdown(
            "- Missing values, duplicates, type issues\n- IQR outlier detection\n"
            "- Inconsistent categories & format validation\n- Invalid ranges, constant columns\n"
            "- 0-100 quality score (overall + per column)\n- One-click cleaning with before/after\n"
            "- CSV, Excel and HTML report downloads"
        )
        if st.session_state.original is not None:
            st.header("⚙️ Check settings")
            st.session_state.multiplier = st.slider("Outlier IQR multiplier", 1.0, 3.0, 1.5, 0.1,
                                                    help="Values beyond Q1/Q3 ± multiplier × IQR are outliers.")
            st.session_state.keys = st.multiselect("Key columns for duplicate check",
                                                   list(st.session_state.original.columns),
                                                   help="Optional: also flag rows that repeat these columns.")


def render_loader() -> None:
    st.title("🧪 Data Quality Checker")
    st.caption("Automated dataset profiling and cleaning")
    file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
    if file is not None:
        sig = (file.name, file.size)
        if st.session_state.get("upload_sig") != sig:
            try:
                df, warn = load_upload(file.getvalue(), file.name)
            except ValueError as exc:
                st.error(str(exc))
                return
            st.session_state.upload_sig = sig
            set_dataset(df, file.name, warn)
    elif st.session_state.original is None:
        st.info("No file uploaded yet.")
        if st.button("Use sample data", type="primary"):
            df, warn = load_sample()
            set_dataset(df, "sample_data.csv", warn)
            st.rerun()
    if st.session_state.warning:
        st.warning(st.session_state.warning)


# ------------------------------------------------------------------ tabs
def gauge(score: float, title: str = "Data Quality Score") -> go.Figure:
    color = "#21c354" if score >= 80 else "#ffa421" if score >= 60 else "#ff4b4b"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, title={"text": title},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
               "steps": [{"range": [0, 60], "color": "#3a1a1a"}, {"range": [60, 80], "color": "#3a2e12"},
                         {"range": [80, 100], "color": "#14301c"}]},
    ))
    fig.update_layout(height=280, margin=dict(t=60, b=10, l=30, r=30), paper_bgcolor="rgba(0,0,0,0)",
                      font={"color": "#fafafa"})
    return fig


def missing_heatmap(df: pd.DataFrame) -> go.Figure:
    mask = q.missing_mask(df)
    if len(mask) > 400:
        mask = mask.iloc[:: len(mask) // 400 + 1]
    fig = go.Figure(go.Heatmap(z=mask.astype(int).T.values, x=mask.index.astype(str), y=list(mask.columns),
                               colorscale=[[0, "#262730"], [1, "#ff4b4b"]], showscale=False,
                               hovertemplate="row %{x}, %{y}<extra></extra>"))
    fig.update_layout(height=max(260, 24 * mask.shape[1] + 80), margin=dict(t=10, b=10),
                      xaxis_title="Row (red = missing)", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font={"color": "#fafafa"})
    fig.update_xaxes(showticklabels=False)
    return fig


def tab_overview(df: pd.DataFrame, issues: list[q.Issue]) -> None:
    score = q.overall_score(df, issues)
    prof = q.profile(df)
    left, right = st.columns([1, 2])
    left.plotly_chart(gauge(score), use_container_width=True)
    with right:
        k = st.columns(4)
        k[0].metric("Rows", f"{prof['rows']:,}")
        k[1].metric("Columns", prof["columns"])
        k[2].metric("Missing cells", f"{prof['missing_pct']}%")
        k[3].metric("Duplicate rows", prof["duplicates"])
        st.info(q.SCORE_FORMULA)
    st.subheader("Score per column")
    scores = q.column_scores(df, issues).sort_values("score")
    fig = go.Figure(go.Bar(x=scores["score"], y=scores["column"], orientation="h", marker_color="#ff4b4b"))
    fig.update_layout(height=max(250, 26 * len(scores) + 60), xaxis_range=[0, 100], margin=dict(t=10, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#fafafa"})
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Missing-value heatmap")
    st.plotly_chart(missing_heatmap(df), use_container_width=True)
    st.subheader("Missing values per column")
    st.dataframe(q.missing_summary(df), hide_index=True, use_container_width=True)
    with st.expander("Preview data"):
        st.dataframe(df.head(100), use_container_width=True)


def tab_issues(df: pd.DataFrame, issues: list[q.Issue]) -> None:
    if not issues:
        st.success("No issues found 🎉")
        return
    for check in q.CHECK_ORDER:
        group = [i for i in issues if i.check == check]
        if not group:
            continue
        with st.expander(f"{check} — {len(group)} finding(s)", expanded=False):
            for n, issue in enumerate(group):
                st.badge(issue.severity, color=SEVERITY_COLOR[issue.severity])
                st.markdown(f"**{issue.column or 'Whole rows'}** · {issue.description}")
                if len(issue.rows):
                    st.caption(f"Showing up to 200 of {len(issue.rows)} affected rows")
                    st.dataframe(df.loc[issue.rows].head(200), use_container_width=True)
                if n < len(group) - 1:
                    st.divider()


def clean_options(df: pd.DataFrame, issues: list[q.Issue], multiplier: float, keys: list[str]) -> dict:
    opts: dict = {}
    text_cols = [c for c in df.columns if q.is_text(df[c])]
    flagged_text = [i.column for i in issues if i.check == "Inconsistent categories"]
    miss_cols = [i.column for i in issues if i.check == "Missing values"]
    out_cols = [i.column for i in issues if i.check == "Outliers"]
    num_cols = [c for c in df.columns if q.numeric_view(df[c]) is not None]
    suggest = {i.column: i.suggestion for i in issues if i.check == "Data types" and i.suggestion}

    if st.checkbox("Drop duplicates", key="c_dup", help="Uses the key columns from the sidebar, or full rows."):
        opts["drop_duplicates"] = {"subset": keys or None}
    if st.checkbox("Trim whitespace & standardize casing", key="c_text"):
        cols = st.multiselect("Text columns", text_cols, default=flagged_text, key="c_text_cols")
        case = st.selectbox("Casing", ["keep", "lower", "upper", "title"], key="c_case")
        merge = st.checkbox("Merge variant spellings (e.g. 'USA' / 'U.S.A')", value=True, key="c_merge")
        opts["text"] = {"columns": cols, "case": case, "merge_variants": merge}
    if st.checkbox("Fill missing values", key="c_fill"):
        strategy = st.selectbox("Strategy", ["mean", "median", "mode", "constant", "drop rows"], key="c_strat")
        cols = st.multiselect("Columns", list(df.columns), default=miss_cols, key="c_fill_cols")
        const = st.text_input("Constant value", "Unknown", key="c_const") if strategy == "constant" else ""
        opts["fill"] = {"strategy": strategy, "columns": cols, "constant": const}
    if st.checkbox("Cap or remove outliers", key="c_out"):
        mode = st.radio("Mode", ["cap", "remove"], horizontal=True, key="c_out_mode")
        cols = st.multiselect("Numeric columns", num_cols, default=out_cols,
                              key="c_out_cols")
        opts["outliers"] = {"columns": cols, "mode": mode, "multiplier": multiplier}
    if st.checkbox("Convert column types", key="c_conv"):
        cols = st.multiselect("Columns to convert", list(df.columns), default=list(suggest), key="c_conv_cols")
        conv = {}
        for col in cols:
            options = ["numeric", "datetime", "text"]
            conv[col] = st.selectbox(f"{col} →", options, index=options.index(suggest.get(col, "text")),
                                     key=f"c_conv_{col}")
        opts["convert"] = conv
    return {k: v for k, v in opts.items() if v not in (None, {}) and not (isinstance(v, dict) and v.get("columns") == [])}


def delta_metric(col, label: str, before: float, after: float, fmt: str = "{:,}") -> None:
    col.metric(label, fmt.format(after), delta=round(after - before, 2),
               delta_color="normal" if label == "Quality score" else "off")


def tab_clean(original: pd.DataFrame, issues: list[q.Issue], multiplier: float, keys: list[str]) -> None:
    st.caption("Cleaning always starts from your original upload, so you can change options and re-apply.")
    opts = clean_options(original, issues, multiplier, keys)
    c1, c2 = st.columns([1, 5])
    if c1.button("Apply", type="primary", disabled=not opts):
        try:
            cleaned, log = q.apply_cleaning(original, opts)
            st.session_state.cleaned, st.session_state.actions = cleaned, log
        except Exception as exc:  # noqa: BLE001 - keep the UI alive on unexpected data
            st.error(f"Cleaning failed: {exc}")
    if c2.button("Reset cleaning"):
        st.session_state.cleaned, st.session_state.actions = None, []
        st.rerun()
    cleaned = st.session_state.cleaned
    if cleaned is None:
        return
    st.subheader("Before / after")
    after_issues = cached_checks(cleaned, tuple(keys), multiplier)
    sb, sa = q.overall_score(original, issues), q.overall_score(cleaned, after_issues)
    pb, pa = q.profile(original), q.profile(cleaned)
    m = st.columns(4)
    delta_metric(m[0], "Quality score", sb, sa, "{:.1f}")
    delta_metric(m[1], "Row count", len(original), len(cleaned))
    delta_metric(m[2], "Missing cells %", pb["missing_pct"], pa["missing_pct"], "{:.2f}")
    delta_metric(m[3], "Duplicate rows", pb["duplicates"], pa["duplicates"])
    fig = go.Figure(go.Bar(x=["Before", "After"], y=[sb, sa], marker_color=["#808495", "#ff4b4b"]))
    fig.update_layout(height=240, yaxis_range=[0, 100], margin=dict(t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font={"color": "#fafafa"})
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Actions applied**")
    for a in st.session_state.actions:
        st.markdown(f"- {a}")
    st.dataframe(cleaned.head(100), use_container_width=True)


def tab_report(original: pd.DataFrame, issues: list[q.Issue], multiplier: float, keys: list[str]) -> None:
    cleaned = st.session_state.cleaned
    final = cleaned if cleaned is not None else original
    st.subheader("Downloads")
    st.caption("The dataset below is the cleaned version" if cleaned is not None
               else "No cleaning applied yet — this is the original data.")
    score_before = q.overall_score(original, issues)
    score_after = q.overall_score(cleaned, cached_checks(cleaned, tuple(keys), multiplier)) if cleaned is not None else None
    report = q.build_html_report(st.session_state.source, original, issues, score_before,
                                 st.session_state.actions, cleaned, score_after)
    d = st.columns(3)
    d[0].download_button("⬇️ Cleaned data (CSV)", final.to_csv(index=False).encode("utf-8"),
                         "cleaned_data.csv", "text/csv", use_container_width=True)
    d[1].download_button("⬇️ Cleaned data (Excel)", to_excel_bytes(final), "cleaned_data.xlsx",
                         "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                         use_container_width=True)
    d[2].download_button("⬇️ Quality report (HTML)", report, "quality_report.html", "text/html",
                         use_container_width=True)
    st.subheader("Report preview")
    st.iframe(report, height=600)


# --------------------------------------------------------------------- main
def main() -> None:
    init_state()
    render_sidebar()
    render_loader()
    original: pd.DataFrame | None = st.session_state.original
    if original is None:
        return
    multiplier = st.session_state.get("multiplier", 1.5)
    keys = st.session_state.get("keys", [])
    keys = [k for k in keys if k in original.columns]
    working = st.session_state.cleaned if st.session_state.cleaned is not None else original
    issues_now = cached_checks(working, tuple(keys), multiplier)
    issues_orig = issues_now if working is original else cached_checks(original, tuple(keys), multiplier)
    st.caption(f"📄 {st.session_state.source} — "
               + ("showing **cleaned** data" if working is not original else "showing **original** data"))
    t_over, t_issues, t_clean, t_report = st.tabs(["📊 Overview", "🔎 Issues", "🧹 Clean Data", "📝 Report"])
    with t_over:
        tab_overview(working, issues_now)
    with t_issues:
        tab_issues(working, issues_now)
    with t_clean:
        tab_clean(original, issues_orig, multiplier, keys)
    with t_report:
        tab_report(original, issues_orig, multiplier, keys)


main()
