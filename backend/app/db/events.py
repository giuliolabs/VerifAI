import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path("data") / "verifai_events.sqlite"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            filename TEXT,
            label TEXT,
            prob_fake REAL
        )
        """)

def log_prediction(filename: str, label: str, prob_fake: float):
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT INTO predictions(timestamp, filename, label, prob_fake) VALUES(?,?,?,?)",
            (datetime.utcnow().isoformat(), filename, label, prob_fake),
        )
