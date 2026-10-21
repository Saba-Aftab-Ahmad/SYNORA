"""
pytest bootstrap.

SAFETY: the test-suite ALWAYS runs against a throw-away SQLite file - never
against the Supabase database named in .env.  DATABASE_URL is forced here
before the Flask app is imported (python-dotenv never overrides an already
set variable).
"""
import os
import tempfile

_tmp = os.path.join(tempfile.mkdtemp(prefix="synora_test_"), "test.db")
_url = os.environ.get("SYNORA_TEST_DATABASE_URL", f"sqlite:///{_tmp}")
# Opt-in PostgreSQL run (see RUN_GUIDE): only a LOCAL database is accepted.
assert _url.startswith("sqlite") or "localhost" in _url or "127.0.0.1" in _url, (
    "SYNORA_TEST_DATABASE_URL must point to a local database, never to Supabase"
)
os.environ["DATABASE_URL"] = _url
os.environ.pop("ADMIN_KEY", None)
