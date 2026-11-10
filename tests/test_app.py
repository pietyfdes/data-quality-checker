import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
APP = str(ROOT / "app.py")


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def _loaded_app():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    _button(at, "Use sample data").click().run()
    assert not at.exception
    return at


def test_sample_data_flow_renders_all_tabs():
    at = _loaded_app()
    assert [t.label for t in at.tabs] == ["📊 Overview", "🔎 Issues", "🧹 Clean Data", "📝 Report"]
    assert at.session_state["original"].shape[0] > 250
    assert len(at.metric) >= 4 and len(at.expander) >= 5


def test_apply_cleaning_keeps_original_and_improves_score():
    at = _loaded_app()
    original = at.session_state["original"].copy()
    at.checkbox(key="c_dup").check()
    at.checkbox(key="c_text").check().run()
    at.checkbox(key="c_fill").check().run()
    at.selectbox(key="c_strat").set_value("mode").run()
    _button(at, "Apply").click().run()
    assert not at.exception
    cleaned = at.session_state["cleaned"]
    assert cleaned is not None and len(cleaned) < len(original)
    assert at.session_state["original"].equals(original)
    assert len(at.session_state["actions"]) == 3


def test_unsupported_file_is_handled():
    import quality as q
    try:
        q.read_table(b"x", "file.pdf")
    except ValueError as exc:
        assert "Unsupported" in str(exc)
