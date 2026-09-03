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
            "Add it to Render environment variables."
        )

    # Fix 1: SQLAlchemy requires postgresql:// not postgres://
    if database_url.startswith("postgres://"):
        database_url = database_url.replace(
            "postgres://", "postgresql://", 1
        )

    # Fix 2: Use psycopg3 driver syntax
    # Replace postgresql:// with postgresql+psycopg://
    if "postgresql://" in database_url and "+psycopg" not in database_url:
        database_url = database_url.replace(
            "postgresql://",
            "postgresql+psycopg://",
            1
        )

    app.config["SQLALCHEMY_DATABASE_URI"] = database_url
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_pre_ping": True,
        "pool_recycle": 300,
        "pool_size": 5,
        "max_overflow": 2,
    }

    db.init_app(app)
    print("Database connected successfully")