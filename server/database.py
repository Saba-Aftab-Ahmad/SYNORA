import os
import threading
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv

load_dotenv()

db = SQLAlchemy()


def _normalise_url(database_url: str) -> str:
    # Fix postgres:// to postgresql://
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)

    # Force psycopg2 driver explicitly (SQLite URLs are left untouched so the
    # project can also be tested locally without touching Supabase).
    if "postgresql+psycopg://" in database_url:
        database_url = database_url.replace("postgresql+psycopg://", "postgresql+psycopg2://", 1)
    elif database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return database_url


def init_db(app, database_url=None):
    """
    Initialise database connection with Flask app.

    DATABASE_URL examples
      production : postgresql://user:pass@host:5432/postgres   (Supabase)
      local test : sqlite:///local_test.db                      (no server needed)
    """
    database_url = database_url or os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set.")

    database_url = _normalise_url(database_url)

    app.config["SQLALCHEMY_DATABASE_URI"] = database_url
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    if database_url.startswith("sqlite"):
        app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"connect_args": {"check_same_thread": False, "timeout": 30}}
    else:
        app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
            "pool_pre_ping": True,
            "pool_recycle": 300,
            "pool_size": 5,
            "max_overflow": 5,
        }

    db.init_app(app)
    kind = "SQLite (local test)" if database_url.startswith("sqlite") else "PostgreSQL"
    print(f"Database configured: {kind}")


_schema_lock = threading.Lock()


def ensure_schema(app):
    """
    Create missing tables AND add missing columns to existing tables.

    Why: the Supabase database already contains the v1 tables (clients, rounds,
    experiment_logs, ...). `db.create_all()` never alters an existing table, so
    the new v2 columns (session_id, session_token, f1, ...) would be missing and
    every query would fail.  This function adds them (all new columns are
    nullable) - it is safe to run on every start and from several gunicorn
    workers at once (PostgreSQL advisory lock).
    """
    from sqlalchemy import inspect, text

    with _schema_lock, app.app_context():
        engine = db.engine
        is_pg = engine.dialect.name == "postgresql"
        with engine.begin() as conn:
            if is_pg:
                conn.execute(text("SELECT pg_advisory_xact_lock(727274)"))
            db.metadata.create_all(bind=conn)
            insp = inspect(conn)
            for table in db.metadata.sorted_tables:
                existing = {c["name"] for c in insp.get_columns(table.name)}
                for col in table.columns:
                    if col.name in existing:
                        continue
                    coltype = col.type.compile(dialect=engine.dialect)
                    conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {coltype}'))
                    print(f"[schema] added column {table.name}.{col.name}")
