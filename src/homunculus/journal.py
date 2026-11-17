"""Durable request identity and result persistence.

Every finite mutation is written here, with its original request and resolved
command, before anything is dispatched. A retry carrying the same
(principal, robot, request_id) finds the first operation instead of creating a
second one, including after the runtime restarts.

A journal file has one owner at a time. Two runtimes writing the same file
would each answer retries with the other's operations, so opening a journal
that is already open is refused.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from .models import RobotError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    principal    TEXT NOT NULL,
    robot_id     TEXT NOT NULL,
    request_id   TEXT NOT NULL,
    body_sha256  TEXT NOT NULL,
    kind         TEXT NOT NULL,
    operation_id TEXT NOT NULL,
    boot_id      TEXT NOT NULL,
    context_id   TEXT NOT NULL,
    request      TEXT NOT NULL,
    document     TEXT NOT NULL,
    finished     INTEGER NOT NULL,
    PRIMARY KEY (principal, robot_id, request_id)
);
CREATE INDEX IF NOT EXISTS requests_by_operation ON requests (robot_id, operation_id);
"""
_COLUMNS = (
    "principal, robot_id, request_id, body_sha256, kind, operation_id, boot_id, context_id, request, document, finished"
)


@dataclass(frozen=True)
class Entry:
    principal: str
    robot_id: str
    request_id: str
    body_sha256: str
    kind: str
    operation_id: str
    boot_id: str
    context_id: str
    request: dict[str, Any]
    document: dict[str, Any]
    finished: bool


def digest(body: dict[str, Any]) -> str:
    """Identity of a request body, independent of key order."""
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class JournalInUse(Exception):
    """The journal file is held by another process, or by another Journal in this one."""


class Journal:
    """One SQLite file, held exclusively. Each write commits, and is on disk, before the call returns."""

    def __init__(self, path: str | Path, boot_id: str, quota: int = 10_000):
        self.boot_id = boot_id
        self.quota = quota
        self._lock = threading.Lock()
        # The timeout is how long the claim below keeps trying: runtimes that open one journal at the same
        # instant block each other for a moment, and waiting it out is what lets exactly one of them win.
        try:
            self._db = sqlite3.connect(str(path), timeout=1.0, check_same_thread=False, isolation_level=None)
        except sqlite3.OperationalError as error:  # a directory that is missing or cannot be written
            raise OSError(f"cannot open the journal {path}: {error}") from None
        try:
            self._db.execute("BEGIN EXCLUSIVE")
            # Switched while the lock is held, this mode keeps it past the commit and until close(). (Set
            # before the lock is won, it would also keep a loser's read lock and stall the winner.) The
            # operating system drops the lock when the process dies, however it dies, so a restart can
            # always reopen its journal.
            self._db.execute("PRAGMA locking_mode = EXCLUSIVE")
            self._db.execute("COMMIT")
            self._db.execute("PRAGMA synchronous = FULL")
            self._db.executescript(_SCHEMA)
        except sqlite3.DatabaseError as error:
            self._db.close()
            if error.sqlite_errorcode == sqlite3.SQLITE_BUSY:
                raise JournalInUse(str(path)) from None
            raise OSError(f"cannot use the journal {path}: {error}") from None  # not a database, or damaged

    def lookup(self, principal: str, robot_id: str, request_id: str) -> Entry | None:
        return self._one("principal = ? AND robot_id = ? AND request_id = ?", (principal, robot_id, request_id))

    def find(self, robot_id: str, operation_id: str) -> Entry | None:
        return self._one("robot_id = ? AND operation_id = ?", (robot_id, operation_id))

    def unfinished(self, robot_id: str) -> list[Entry]:
        """Operations an earlier runtime incarnation accepted and never finished."""
        with self._lock:
            rows = self._db.execute(
                f"SELECT {_COLUMNS} FROM requests WHERE robot_id = ? AND finished = 0 AND boot_id != ?",
                (robot_id, self.boot_id),
            ).fetchall()
        return [_entry(row) for row in rows]

    def admit(
        self,
        *,
        principal: str,
        robot_id: str,
        request_id: str,
        kind: str,
        operation_id: str,
        context_id: str,
        request: dict[str, Any],
        document: dict[str, Any],
    ) -> None:
        """Record a new, unfinished operation, or refuse it when the journal is full.

        At the quota, only results from retired contexts are dropped: stale-context
        rejection already prevents those requests from executing again. Identities
        in the active context are never deleted to make room.
        """
        with self._lock:
            count = "SELECT COUNT(*) FROM requests WHERE robot_id = ?"
            if self._db.execute(count, (robot_id,)).fetchone()[0] >= self.quota:
                self._db.execute(
                    "DELETE FROM requests WHERE robot_id = ? AND finished = 1 AND context_id != ?",
                    (robot_id, context_id),
                )
                if self._db.execute(count, (robot_id,)).fetchone()[0] >= self.quota:
                    raise RobotError(
                        "journal_full",
                        "the request journal is at its storage quota; no new operation was admitted",
                        quota=self.quota,
                    )
            self._db.execute(
                f"INSERT INTO requests ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    principal,
                    robot_id,
                    request_id,
                    digest(request),
                    kind,
                    operation_id,
                    self.boot_id,
                    context_id,
                    json.dumps(request),
                    json.dumps(document),
                    0,
                ),
            )

    def record(self, robot_id: str, operation_id: str, document: dict[str, Any], *, finished: bool) -> None:
        """Replace an operation's stored state. Used before dispatch and at its end."""
        with self._lock:
            self._db.execute(
                "UPDATE requests SET document = ?, finished = ? WHERE robot_id = ? AND operation_id = ?",
                (json.dumps(document), int(finished), robot_id, operation_id),
            )

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _one(self, where: str, values: tuple[str, ...]) -> Entry | None:
        with self._lock:
            row = self._db.execute(f"SELECT {_COLUMNS} FROM requests WHERE {where}", values).fetchone()
        return None if row is None else _entry(row)


def held(path: str | Path) -> bool:
    """Whether a Journal, in any process, holds this file now. Nothing is created or changed."""
    try:
        db = sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True, timeout=0.2)
    except sqlite3.OperationalError:  # no such file
        return False
    with closing(db):
        try:
            db.execute("SELECT 1 FROM sqlite_master").fetchone()
        except sqlite3.DatabaseError as error:
            # Any other error is from a file no Journal could be holding.
            return error.sqlite_errorcode == sqlite3.SQLITE_BUSY
    return False


def _entry(row: tuple[Any, ...]) -> Entry:
    return Entry(*row[:8], request=json.loads(row[8]), document=json.loads(row[9]), finished=bool(row[10]))
