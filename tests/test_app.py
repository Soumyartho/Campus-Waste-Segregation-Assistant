"""Streamlit app tests: every page renders, and (when a trained model exists) the classify,
feedback, batch and analytics flows work end to end. Logs go to a temporary database."""

from __future__ import annotations

import io
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest
import yaml
from streamlit.testing.v1 import AppTest

from tests.conftest import make_image
from waste.utils import PROJECT_ROOT, load_yaml

# Point the app at a copy of inference.yaml whose database and feedback folder are temporary.
_TMP = Path(tempfile.mkdtemp(prefix="wsa_app_test_"))
_cfg = load_yaml("configs/inference.yaml")
_cfg["app"].update(db_path=str(_TMP / "app.db"), feedback_dir=str(_TMP / "feedback"))
(_TMP / "inference.yaml").write_text(yaml.safe_dump(_cfg), encoding="utf-8")
os.environ["WASTE_INFERENCE_CONFIG"] = str(_TMP / "inference.yaml")

APP = PROJECT_ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

PAGES = ["streamlit_app.py", "pages/classify.py", "pages/batch.py", "pages/analytics.py",
         "pages/insights.py", "pages/about.py"]
HAS_MODEL = (PROJECT_ROOT / _cfg["model"]["models_dir"] / _cfg["model"]["run_name"]
             / "best.pt").exists()
needs_model = pytest.mark.skipif(not HAS_MODEL, reason="no trained checkpoint")


def _png(seed: int) -> bytes:
    buf = io.BytesIO()
    make_image(seed, 320).save(buf, "PNG")
    return buf.getvalue()


def _page(name: str) -> AppTest:
    return AppTest.from_file(str(APP / name), default_timeout=180)


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exceptions(page):
    at = _page(page).run()
    assert not at.exception, [e.value for e in at.exception]


@needs_model
def test_classify_upload_gradcam_and_feedback():
    at = _page("pages/classify.py").run()
    at.file_uploader[0].upload("item.png", _png(3), "image/png").run()
    assert not at.exception, [e.value for e in at.exception]
    html = " ".join(m.value for m in at.markdown)
    assert "wsa-card" in html and ("bin" in html or "retake" in html.lower() or "doubt" in html)

    at.toggle[0].set_value(True).run()
    assert not at.exception and at.session_state["cam"] is not None

    at.button[0].click().run()
    assert at.session_state["feedback_sent"]
    from waste.logstore import LogStore

    store = LogStore(_cfg["app"]["db_path"])
    assert len(store.predictions()) >= 1 and len(store.feedback()) == 1


@needs_model
def test_batch_with_zip_and_analytics():
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        for i in range(3):
            zf.writestr(f"photos/item_{i}.png", _png(10 + i))
        zf.writestr("__MACOSX/._junk.png", b"not an image")
    at = _page("pages/batch.py").run()
    at.file_uploader[0].set_value([("one.png", _png(20), "image/png"),
                                   ("batch.zip", archive.getvalue(), "application/zip")]).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.session_state["batch_rows"]) == 4

    at = _page("pages/analytics.py").run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.metric[0].value.replace(",", "").isdigit()
