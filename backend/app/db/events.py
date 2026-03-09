"""
Database Event Logging Module
=============================

This module manages lightweight persistence for prediction events
within the VerifAI system. It creates a local SQLite database (if it
does not already exist) and stores metadata about each deepfake
prediction for monitoring, auditing, and basic analytics purposes.

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

# Import sqlite3 to create and interact with a local SQLite database
import sqlite3

# Import Path to manage filesystem paths in a clean and platform-independent way
from pathlib import Path

# Import datetime to record the exact time of each prediction
from datetime import datetime


# Define the database file location inside a "data" folder
# Using Path improves readability and cross-platform compatibility
DB_PATH = Path("data") / "verifai_events.sqlite"

# Ensure the parent directory exists before attempting to use the database
# This prevents runtime errors if the folder has not been created yet
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def init_db():
    """
    Initialize the database by creating the predictions table if it does not already exist.
    """

    # Open a connection to the SQLite database
    # The 'with' statement ensures the connection is safely closed
    with sqlite3.connect(DB_PATH) as con:
        # Create the predictions table with relevant fields:
        # - id: unique identifier for each entry
        # - timestamp: when the prediction occurred
        # - filename: name of the analysed video
        # - label: model classification result (Real/Fake)
        # - prob_fake: probability score stored as numeric value
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
    """
    Insert a new prediction record into the database.
    This function is called after a successful inference.
    """

    # Open a database connection for inserting the record
    with sqlite3.connect(DB_PATH) as con:
        # Insert the prediction data into the table.
        # Parameterised queries are used to prevent SQL injection and ensure safe handling of user-provided data.
        con.execute(
            "INSERT INTO predictions(timestamp, filename, label, prob_fake) VALUES(?,?,?,?)",
            (
                datetime.utcnow().isoformat(),  # Store time in UTC for consistency
                filename,
                label,
                prob_fake,
            ),
        )