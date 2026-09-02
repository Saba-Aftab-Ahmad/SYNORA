"""
Synora — Database Connection
============================
SQLAlchemy setup for PostgreSQL on Supabase.
Reads DATABASE_URL from environment variable.
"""

import os
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv

load_dotenv()

db = SQLAlchemy()


def init_db(app):
    """
    Initialise database connection with Flask app.

    Args:
        app: Flask application instance
    """
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL environment variable not set. "
            "Copy .env.example to .env and fill in your "
            "Supabase connection string."
        )

    # Fix for SQLAlchemy — Supabase uses postgres://
    # but SQLAlchemy requires postgresql://
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)

    app.config["SQLALCHEMY_DATABASE_URI"] = database_url
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_pre_ping": True,
        "pool_recycle": 300,
    }

    db.init_app(app)
    print(f"Database connected successfully")
