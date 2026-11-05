import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

# Durable job storage: the uploaded file, the search settings, the job status and every finished row.
# A server restart / crash resumes running jobs where they stopped instead of losing them.
STORE_DIR = os.environ.get("JOB_STORE_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(STORE_DIR, exist_ok=True)
DB_PATH = os.path.join(STORE_DIR, "jobs.sqlite3")

_lock = threading.Lock()
_conn = sqlite3.connect(DB_PATH, check_same_thread=False, isolation_level=None)
_conn.execute("PRAGMA journal_mode=WAL")
_conn.execute("PRAGMA synchronous=NORMAL")
_conn.execute("""CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY, filename TEXT, file BLOB, config TEXT, status TEXT, created REAL, updated REAL)""")
_conn.execute("""CREATE TABLE IF NOT EXISTS rows (
    job_id TEXT, row_index INTEGER, data TEXT, PRIMARY KEY (job_id, row_index))""")


def save_upload(job_id: str, filename: str, content: bytes):
    now = time.time()
    with _lock:
        _conn.execute("INSERT OR REPLACE INTO jobs (job_id, filename, file, config, status, created, updated) "
                      "VALUES (?, ?, ?, NULL, NULL, ?, ?)", (job_id, filename, content, now, now))


def save_config(job_id: str, config: Dict[str, Any]):
    with _lock:
        _conn.execute("UPDATE jobs SET config=?, updated=? WHERE job_id=?", (json.dumps(config), time.time(), job_id))
        # a new start replaces earlier results of this job
        _conn.execute("DELETE FROM rows WHERE job_id=?", (job_id,))


def save_status(job_id: str, status: Dict[str, Any]):
    with _lock:
        _conn.execute("UPDATE jobs SET status=?, updated=? WHERE job_id=?", (json.dumps(status), time.time(), job_id))


def save_row(job_id: str, row_index: int, row_json: str):
    with _lock:
        _conn.execute("INSERT OR REPLACE INTO rows VALUES (?, ?, ?)", (job_id, row_index, row_json))


def load_jobs(max_age_days: float = 30) -> List[Dict[str, Any]]:
    """Jobs started within max_age_days, with their stored rows."""
    cutoff = time.time() - max_age_days * 86400
    with _lock:
        jobs = _conn.execute("SELECT job_id, filename, file, config, status FROM jobs "
                             "WHERE config IS NOT NULL AND updated >= ? ORDER BY created", (cutoff,)).fetchall()
    out = []
    for job_id, filename, file, config, status in jobs:
        with _lock:
            rows = _conn.execute("SELECT data FROM rows WHERE job_id=?", (job_id,)).fetchall()
        out.append({
            "job_id": job_id, "filename": filename, "file": file,
            "config": json.loads(config), "status": json.loads(status) if status else None,
            "rows": [json.loads(r[0]) for r in rows],
        })
    return out


def list_jobs(limit: int = 50) -> List[Dict[str, Any]]:
    with _lock:
        rows = _conn.execute("SELECT job_id, filename, status, created FROM jobs WHERE config IS NOT NULL "
                             "ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
    return [{"job_id": j, "filename": f, "status": json.loads(s) if s else None, "created": c} for j, f, s, c in rows]
