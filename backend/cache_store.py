import json
import os
import sqlite3
import threading
import time
from collections import OrderedDict
from typing import Any, Optional

# Persistent cache: re-running a file, "Retry Not Found", or resuming after a server restart
# skips every search and OCR already done. Lives next to the backend in cache/image_cache.sqlite3.
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
DB_PATH = os.path.join(CACHE_DIR, "image_cache.sqlite3")

SEARCH_TTL_SECONDS = 14 * 24 * 3600   # image search results go stale (links die); OCR text never does
MEMORY_ITEMS = 20000

_lock = threading.Lock()
_conn = sqlite3.connect(DB_PATH, check_same_thread=False, isolation_level=None)
_conn.execute("PRAGMA journal_mode=WAL")
_conn.execute("PRAGMA synchronous=NORMAL")
_conn.execute("CREATE TABLE IF NOT EXISTS ocr (key TEXT PRIMARY KEY, data TEXT, ts REAL)")
_conn.execute("CREATE TABLE IF NOT EXISTS search (key TEXT PRIMARY KEY, data TEXT, ts REAL)")


class _LRU:
    def __init__(self, cap: int):
        self.cap = cap
        self.items: "OrderedDict[str, Any]" = OrderedDict()

    def get(self, key: str):
        if key in self.items:
            self.items.move_to_end(key)
            return self.items[key]
        return None

    def put(self, key: str, value: Any):
        self.items[key] = value
        self.items.move_to_end(key)
        while len(self.items) > self.cap:
            self.items.popitem(last=False)


_mem = {"ocr": _LRU(MEMORY_ITEMS), "search": _LRU(MEMORY_ITEMS)}


def _get(table: str, key: str, ttl: Optional[float] = None) -> Optional[Any]:
    hit = _mem[table].get(key)
    if hit is not None:
        return hit
    with _lock:
        row = _conn.execute(f"SELECT data, ts FROM {table} WHERE key=?", (key,)).fetchone()
    if not row or (ttl and time.time() - row[1] > ttl):
        return None
    value = json.loads(row[0])
    _mem[table].put(key, value)
    return value


def _put(table: str, key: str, value: Any):
    _mem[table].put(key, value)
    with _lock:
        _conn.execute(f"INSERT OR REPLACE INTO {table} VALUES (?, ?, ?)", (key, json.dumps(value), time.time()))


def get_ocr(url: str):
    return _get("ocr", url)


def put_ocr(url: str, words):
    _put("ocr", url, words)


def get_search(key: str):
    return _get("search", key, SEARCH_TTL_SECONDS)


def put_search(key: str, result: dict):
    _put("search", key, result)
