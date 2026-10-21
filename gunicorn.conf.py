"""
Gunicorn production server configuration (Render).

All coordination state lives in PostgreSQL, so several workers are safe.
gthread = 2 processes x 4 threads: many browser clients polling every ~2 s
never block a request that is aggregating or serving a model.
"""

import os

bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"

workers = int(os.getenv("WEB_CONCURRENCY", "2"))
threads = int(os.getenv("GUNICORN_THREADS", "4"))
worker_class = "gthread"
timeout = 120
keepalive = 5

accesslog = "-"
errorlog = "-"
loglevel = "info"

proc_name = "synora-coordination-server"
