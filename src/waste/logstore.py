"""SQLite store for app predictions and user feedback (S4, S5).

Images are never stored for plain predictions; only images attached to feedback are saved,
so they can be reviewed and used for later retraining.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image

from waste.utils import resolve_path

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    source TEXT NOT NULL,
    predicted_class TEXT NOT NULL,
    confidence REAL NOT NULL,
    state TEXT NOT NULL,
    bin TEXT,
    recyclable INTEGER,
    model_name TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    prediction_id INTEGER REFERENCES predictions(id),
    predicted_class TEXT NOT NULL,
    correct_class TEXT NOT NULL,
    image_path TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class LogStore:
    """Thin wrapper over the SQLite database at ``db_path``."""

    def __init__(self, db_path: str | Path) -> None:
        self.path = resolve_path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def log_prediction(
        self,
        source: str,
        predicted_class: str,
        confidence: float,
        state: str,
        bin_name: str | None,
        recyclable: bool | None,
        model_name: str,
    ) -> int:
        """Insert one prediction and return its id."""
        with self._connect() as con:
            cur = con.execute(
                "INSERT INTO predictions (ts, source, predicted_class, confidence, state, bin, "
                "recyclable, model_name) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (_now(), source, predicted_class, float(confidence), state, bin_name,
                 None if recyclable is None else int(recyclable), model_name),
            )
            return int(cur.lastrowid)

    def log_feedback(
        self,
        prediction_id: int | None,
        predicted_class: str,
        correct_class: str,
        image: Image.Image | None,
        feedback_dir: str | Path,
    ) -> int:
        """Store a correction; the image is saved under feedback_dir/<correct_class>/."""
        image_path = None
        if image is not None:
            folder = resolve_path(feedback_dir) / correct_class
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / f"{uuid.uuid4().hex}.jpg"
            image.convert("RGB").save(target, "JPEG", quality=92)
            image_path = str(target)
        with self._connect() as con:
            cur = con.execute(
                "INSERT INTO feedback (ts, prediction_id, predicted_class, correct_class, "
                "image_path) VALUES (?, ?, ?, ?, ?)",
                (_now(), prediction_id, predicted_class, correct_class, image_path),
            )
            return int(cur.lastrowid)

    def predictions(self) -> pd.DataFrame:
        """All logged predictions with a parsed timestamp column."""
        with self._connect() as con:
            df = pd.read_sql_query("SELECT * FROM predictions ORDER BY ts", con)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df

    def feedback(self) -> pd.DataFrame:
        """All feedback rows."""
        with self._connect() as con:
            return pd.read_sql_query("SELECT * FROM feedback ORDER BY ts", con)
