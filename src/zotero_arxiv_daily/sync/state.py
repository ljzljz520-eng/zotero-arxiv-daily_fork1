"""SQLite-backed persistent state for incremental Zotero sync.

The store is deliberately a *thin data-access layer*: it knows nothing about
the Zotero protocol or about embedding models. All sync decisions live in
``syncer.py``.

Transaction discipline (essential for the watermark/vector atomicity
guarantee):

* the connection runs in autocommit mode (``isolation_level=None``), so every
  transaction boundary is explicit;
* :meth:`SyncStore.transaction` issues ``BEGIN IMMEDIATE`` and either commits or
  rolls back as a unit;
* content-addressed embedding blobs MAY be committed in their own small
  transactions *before* the big sync transaction (they are immutable and
  referenced by content hash), but the watermark, item/collection rows,
  membership rows and item->blob links MUST be committed together in a single
  transaction by :class:`~zotero_arxiv_daily.sync.syncer.CorpusSyncer`.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

DB_FILENAME = "zotero_sync.db"
AUDIT_DIRNAME = "audit"
SCHEMA_VERSION = 2


class SyncStateError(Exception):
    """Raised when the on-disk state is incompatible or corrupt."""


SCHEMA_SQL = """
CREATE TABLE library_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    library_id TEXT NOT NULL,
    library_type TEXT NOT NULL,
    username TEXT,
    library_version INTEGER NOT NULL,
    embedding_fingerprint TEXT,
    last_sync_at TEXT
);

CREATE TABLE items (
    key TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    item_type TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    abstract TEXT NOT NULL DEFAULT '',
    added_date TEXT,
    content_hash TEXT,
    trashed INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE collections (
    key TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    parent_key TEXT
);

CREATE TABLE item_collections (
    item_key TEXT NOT NULL,
    collection_key TEXT NOT NULL,
    PRIMARY KEY (item_key, collection_key)
) WITHOUT ROWID;

CREATE INDEX idx_item_collections_collection
    ON item_collections(collection_key);

CREATE TABLE embedding_blobs (
    fingerprint TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (fingerprint, content_hash)
) WITHOUT ROWID;

CREATE TABLE item_embeddings (
    item_key TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    PRIMARY KEY (item_key, fingerprint)
) WITHOUT ROWID;

CREATE INDEX idx_item_embeddings_blob
    ON item_embeddings(fingerprint, content_hash);

CREATE TABLE sync_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    status TEXT NOT NULL,
    old_version INTEGER,
    new_version INTEGER,
    detail_json TEXT NOT NULL
);

/* Remote items present in the versions snapshot but outside the tracked
   item-type whitelist (e.g. a journalArticle changed into a book). Remembered
   so the reconcile sweep neither re-fetches them every round nor lets them
   silently re-enter the corpus; a version bump re-fetches (the type may have
   changed back). */
CREATE TABLE excluded_items (
    key TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    item_type TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

#: user_version -> SQL applied when upgrading an existing on-disk database.
_MIGRATIONS: dict[int, str] = {
    1: """
CREATE TABLE excluded_items (
    key TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    item_type TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
PRAGMA user_version = 2;
""",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def vector_to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def blob_to_vector(blob: bytes, dim: int) -> np.ndarray:
    arr = np.frombuffer(blob, dtype=np.float32)
    if arr.shape[0] != dim:
        raise SyncStateError(f"Vector blob length {arr.shape[0]} != dim {dim}")
    return arr.copy()  # copy: frombuffer returns a read-only view


class SyncStore:
    """Thin DAO over the single-file sync database."""

    def __init__(self, state_dir: str | Path):
        self.state_dir = Path(state_dir)
        self.db_path = self.state_dir / DB_FILENAME
        self.audit_dir = self.state_dir / AUDIT_DIRNAME
        self.conn = sqlite3.connect(str(self.db_path), isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    @classmethod
    def open(cls, state_dir: str | Path) -> "SyncStore":
        """Create ``state_dir`` if needed and open the store."""
        path = Path(state_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / AUDIT_DIRNAME).mkdir(exist_ok=True)
        return cls(path)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "SyncStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    # schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        row = self.conn.execute("PRAGMA user_version").fetchone()
        version = row[0]
        if version == SCHEMA_VERSION:
            return
        if version == 0:
            # executescript() implicitly commits any open transaction and runs
            # on a brand-new (empty) database, so fold the version pragma into
            # the script instead of managing a transaction around it.
            self.conn.executescript(
                SCHEMA_SQL + f"\nPRAGMA user_version = {SCHEMA_VERSION};\n"
            )
            return
        if version in _MIGRATIONS:
            # executescript() commits any open transaction; migrations are DDL
            # on an idle autocommit connection here.
            self.conn.executescript(_MIGRATIONS[version])
            if self.conn.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
                raise SyncStateError(
                    f"Migration from schema version {version} did not reach "
                    f"{SCHEMA_VERSION}"
                )
            return
        raise SyncStateError(
            f"Unsupported sync state schema version {version}; "
            f"this build supports {SCHEMA_VERSION}. Remove the state dir to rebuild."
        )

    @contextmanager
    def transaction(self, mode: str = "IMMEDIATE") -> Iterator[sqlite3.Connection]:
        """One explicit transaction. commit on success, rollback on error."""
        self.conn.execute(f"BEGIN {mode}")
        try:
            yield self.conn
        except BaseException:
            try:
                self.conn.execute("ROLLBACK")
            except sqlite3.Error:
                # The transaction may already have been torn down (e.g. a
                # disk-full COMMIT failure); never mask the original error.
                pass
            raise
        else:
            self.conn.execute("COMMIT")

    # ------------------------------------------------------------------
    # library identity / watermark
    # ------------------------------------------------------------------

    def get_library_state(self) -> Optional[dict]:
        row = self.conn.execute(
            """SELECT library_id, library_type, username,
                      library_version, embedding_fingerprint, last_sync_at
               FROM library_state WHERE id = 1"""
        ).fetchone()
        return dict(row) if row is not None else None

    def init_library(self, library_id: str, library_type: str, username: str | None) -> None:
        """Insert the singleton identity row at watermark 0. Idempotent.

        A single statement: atomic on its own on the autocommit connection and
        safe to call from inside an already-open transaction (Phase C).
        """
        self.conn.execute(
            """INSERT INTO library_state
                   (id, library_id, library_type, username,
                    library_version, embedding_fingerprint, last_sync_at)
               VALUES (1, ?, ?, ?, 0, NULL, NULL)
               ON CONFLICT(id) DO NOTHING""",
            (str(library_id), library_type, username),
        )

    def update_library_state(
        self,
        *,
        library_version: int,
        embedding_fingerprint: str,
        last_sync_at: str,
        username: str | None = None,
    ) -> None:
        """Watermark/fingerprint update — call INSIDE the big sync txn only."""
        if username is not None:
            self.conn.execute(
                "UPDATE library_state SET username = ? WHERE id = 1",
                (username,),
            )
        self.conn.execute(
            """UPDATE library_state
               SET library_version = ?, embedding_fingerprint = ?, last_sync_at = ?
               WHERE id = 1""",
            (int(library_version), embedding_fingerprint, last_sync_at),
        )

    def touch_library_state(self, last_sync_at: str) -> None:
        """Refresh last_sync_at only (no watermark/fingerprint change).

        Used by no-op rounds; call INSIDE a transaction.
        """
        self.conn.execute(
            "UPDATE library_state SET last_sync_at = ? WHERE id = 1",
            (last_sync_at,),
        )

    # ------------------------------------------------------------------
    # items
    # ------------------------------------------------------------------

    def get_items(self) -> dict[str, dict]:
        rows = self.conn.execute(
            """SELECT key, version, item_type, title, abstract, added_date,
                      content_hash, trashed
               FROM items"""
        ).fetchall()
        return {row["key"]: dict(row) for row in rows}

    def upsert_item(
        self,
        *,
        key: str,
        version: int,
        item_type: str,
        title: str,
        abstract: str,
        added_date: str | None,
        content_hash: str | None,
        trashed: int = 0,
    ) -> None:
        self.conn.execute(
            """INSERT INTO items
                   (key, version, item_type, title, abstract, added_date,
                    content_hash, trashed)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET
                   version = excluded.version,
                   item_type = excluded.item_type,
                   title = excluded.title,
                   abstract = excluded.abstract,
                   added_date = excluded.added_date,
                   content_hash = excluded.content_hash,
                   trashed = excluded.trashed""",
            (key, int(version), item_type, title, abstract, added_date,
             content_hash, int(trashed)),
        )

    def delete_item(self, key: str) -> None:
        """Hard-delete an item, its membership and embedding links."""
        self.conn.execute("DELETE FROM item_collections WHERE item_key = ?", (key,))
        self.conn.execute("DELETE FROM item_embeddings WHERE item_key = ?", (key,))
        self.conn.execute("DELETE FROM items WHERE key = ?", (key,))

    # ------------------------------------------------------------------
    # collections / membership
    # ------------------------------------------------------------------

    def get_collections(self) -> dict[str, dict]:
        rows = self.conn.execute(
            "SELECT key, version, name, parent_key FROM collections"
        ).fetchall()
        return {row["key"]: dict(row) for row in rows}

    def upsert_collection(
        self, *, key: str, version: int, name: str, parent_key: str | None
    ) -> None:
        self.conn.execute(
            """INSERT INTO collections (key, version, name, parent_key)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET
                   version = excluded.version,
                   name = excluded.name,
                   parent_key = excluded.parent_key""",
            (key, int(version), name, parent_key or None),
        )

    def delete_collection(self, key: str) -> None:
        """Delete a collection and detach every item membership to it.

        Item vectors are intentionally untouched: membership is an item
        property, not embedding content.
        """
        self.conn.execute(
            "DELETE FROM item_collections WHERE collection_key = ?", (key,)
        )
        self.conn.execute("DELETE FROM collections WHERE key = ?", (key,))

    def replace_membership(self, item_key: str, collection_keys: set[str]) -> None:
        self.conn.execute(
            "DELETE FROM item_collections WHERE item_key = ?", (item_key,)
        )
        self.conn.executemany(
            "INSERT INTO item_collections (item_key, collection_key) VALUES (?, ?)",
            [(item_key, k) for k in sorted(collection_keys)],
        )

    def get_all_membership(self) -> dict[str, set[str]]:
        result: dict[str, set[str]] = {}
        rows = self.conn.execute(
            "SELECT item_key, collection_key FROM item_collections"
        ).fetchall()
        for row in rows:
            result.setdefault(row["item_key"], set()).add(row["collection_key"])
        return result

    # ------------------------------------------------------------------
    # embedding blobs / links
    # ------------------------------------------------------------------

    def insert_blob(
        self, *, fingerprint: str, content_hash: str, dim: int, vector: np.ndarray
    ) -> bool:
        """Insert an immutable blob. Returns False if it already existed."""
        cur = self.conn.execute(
            """INSERT INTO embedding_blobs
                   (fingerprint, content_hash, dim, vector, created_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(fingerprint, content_hash) DO NOTHING""",
            (fingerprint, content_hash, int(dim), vector_to_blob(vector), utc_now_iso()),
        )
        return cur.rowcount > 0

    def blob_exists(self, fingerprint: str, content_hash: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM embedding_blobs WHERE fingerprint = ? AND content_hash = ?",
            (fingerprint, content_hash),
        ).fetchone()
        return row is not None

    def existing_blobs(self, fingerprint: str, content_hashes: set[str]) -> set[str]:
        if not content_hashes:
            return set()
        found: set[str] = set()
        # bound variables in chunks of 500 for safety on huge libraries
        keys = list(content_hashes)
        for i in range(0, len(keys), 500):
            chunk = keys[i : i + 500]
            placeholders = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                f"""SELECT content_hash FROM embedding_blobs
                    WHERE fingerprint = ? AND content_hash IN ({placeholders})""",
                [fingerprint, *chunk],
            ).fetchall()
            found.update(row["content_hash"] for row in rows)
        return found

    def link_item_embedding(
        self, *, item_key: str, fingerprint: str, content_hash: str
    ) -> None:
        self.conn.execute(
            """INSERT INTO item_embeddings (item_key, fingerprint, content_hash)
               VALUES (?, ?, ?)
               ON CONFLICT(item_key, fingerprint) DO UPDATE SET
                   content_hash = excluded.content_hash""",
            (item_key, fingerprint, content_hash),
        )

    def get_linked_hashes(self, fingerprint: str) -> dict[str, str]:
        rows = self.conn.execute(
            "SELECT item_key, content_hash FROM item_embeddings WHERE fingerprint = ?",
            (fingerprint,),
        ).fetchall()
        return {row["item_key"]: row["content_hash"] for row in rows}

    def delete_links_for_fingerprint(self, fingerprint: str) -> None:
        self.conn.execute(
            "DELETE FROM item_embeddings WHERE fingerprint = ?", (fingerprint,)
        )

    def delete_item_link(self, item_key: str, fingerprint: str) -> None:
        """Drop one item's link (used when its abstract becomes empty)."""
        self.conn.execute(
            "DELETE FROM item_embeddings WHERE item_key = ? AND fingerprint = ?",
            (item_key, fingerprint),
        )

    # ------------------------------------------------------------------
    # excluded (remote but out-of-whitelist) items
    # ------------------------------------------------------------------

    def get_excluded_items(self) -> dict[str, dict]:
        rows = self.conn.execute(
            "SELECT key, version, item_type FROM excluded_items"
        ).fetchall()
        return {r["key"]: dict(r) for r in rows}

    def upsert_excluded_item(
        self, *, key: str, version: int, item_type: str, updated_at: str
    ) -> None:
        self.conn.execute(
            """INSERT INTO excluded_items (key, version, item_type, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET
                   version = excluded.version,
                   item_type = excluded.item_type,
                   updated_at = excluded.updated_at""",
            (key, int(version), item_type, updated_at),
        )

    def delete_excluded_item(self, key: str) -> None:
        self.conn.execute("DELETE FROM excluded_items WHERE key = ?", (key,))

    def delete_excluded_missing(self, present_keys: set[str]) -> None:
        """Drop exclusions for keys that vanished from the versions snapshot."""
        if present_keys:
            placeholders = ",".join("?" for _ in present_keys)
            self.conn.execute(
                f"DELETE FROM excluded_items WHERE key NOT IN ({placeholders})",
                tuple(present_keys),
            )
        else:
            self.conn.execute("DELETE FROM excluded_items")

    def delete_blob(self, fingerprint: str, content_hash: str) -> None:
        self.conn.execute(
            "DELETE FROM embedding_blobs WHERE fingerprint = ? AND content_hash = ?",
            (fingerprint, content_hash),
        )

    def known_fingerprint_dims(self, base: str) -> dict[str, int]:
        """Map full fingerprints starting with ``base + ':dim'`` to their dim.

        Used to resolve the dimension suffix after a crash before any embedding
        call, so previously committed blobs can be reused.
        """
        prefix = base + ":dim"
        rows = self.conn.execute(
            "SELECT DISTINCT fingerprint, dim FROM embedding_blobs"
        ).fetchall()
        return {
            row["fingerprint"]: row["dim"]
            for row in rows
            if row["fingerprint"].startswith(prefix)
        }

    def unreferenced_blobs(self) -> list[tuple[str, str]]:
        rows = self.conn.execute(
            """SELECT b.fingerprint, b.content_hash
               FROM embedding_blobs b
               LEFT JOIN item_embeddings e
                 ON e.fingerprint = b.fingerprint
                AND e.content_hash = b.content_hash
               WHERE e.item_key IS NULL"""
        ).fetchall()
        return [(row["fingerprint"], row["content_hash"]) for row in rows]

    def missing_vectors(self, fingerprint: str) -> list[str]:
        """Keys of active, abstract-bearing items lacking ``fingerprint`` links.

        This is the commit-time invariant: the list MUST be empty right before
        the sync transaction commits.
        """
        rows = self.conn.execute(
            """SELECT i.key
               FROM items i
               WHERE i.trashed = 0 AND i.abstract != ''
                 AND NOT EXISTS (
                     SELECT 1 FROM item_embeddings e
                     WHERE e.item_key = i.key AND e.fingerprint = ?
                 )""",
            (fingerprint,),
        ).fetchall()
        return [row["key"] for row in rows]

    def iter_corpus_rows(self, fingerprint: str) -> list[sqlite3.Row]:
        """All corpus items with live vector blobs for ``fingerprint``."""
        return self.conn.execute(
            """SELECT i.key, i.title, i.abstract, i.added_date,
                      b.dim AS dim, b.vector AS vector
               FROM items i
               JOIN item_embeddings e ON e.item_key = i.key
               JOIN embedding_blobs b
                 ON b.fingerprint = e.fingerprint
                AND b.content_hash = e.content_hash
               WHERE i.trashed = 0 AND e.fingerprint = ?""",
            (fingerprint,),
        ).fetchall()

    # ------------------------------------------------------------------
    # audit
    # ------------------------------------------------------------------

    def insert_audit(
        self,
        *,
        started_at: str,
        finished_at: str,
        status: str,
        old_version: int | None,
        new_version: int | None,
        detail_json: str,
    ) -> int:
        cur = self.conn.execute(
            """INSERT INTO sync_audit
                   (started_at, finished_at, status, old_version, new_version,
                    detail_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (started_at, finished_at, status, old_version, new_version, detail_json),
        )
        return int(cur.lastrowid)

    def get_audit(self, audit_id: int) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM sync_audit WHERE id = ?", (audit_id,)
        ).fetchone()
        return dict(row) if row is not None else None
