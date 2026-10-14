"""Tests for the SQLite sync state store (TR-1.1 .. TR-1.5)."""

import sqlite3

import numpy as np
import pytest

from zotero_arxiv_daily.sync.state import (
    SCHEMA_VERSION,
    SyncStateError,
    SyncStore,
    blob_to_vector,
    vector_to_blob,
)

EXPECTED_TABLES = {
    "library_state",
    "items",
    "collections",
    "item_collections",
    "embedding_blobs",
    "item_embeddings",
    "sync_audit",
    "excluded_items",
}


def _open(tmp_path) -> SyncStore:
    return SyncStore.open(tmp_path / "state")


def _tables(store: SyncStore) -> set[str]:
    rows = store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {r[0] for r in rows}


def _add_item(store: SyncStore, key="I1", abstract="abstract", **kw):
    defaults = dict(
        key=key, version=1, item_type="journalArticle", title="Title",
        abstract=abstract, added_date="2026-01-01T00:00:00Z",
        content_hash="hash1",
    )
    defaults.update(kw)
    store.upsert_item(**defaults)


# TR-1.1 --------------------------------------------------------------------

def test_init_creates_schema_and_dirs(tmp_path):
    state_dir = tmp_path / "state"
    assert not state_dir.exists()
    store = SyncStore.open(state_dir)
    try:
        assert state_dir.is_dir()
        assert (state_dir / "audit").is_dir()
        assert EXPECTED_TABLES <= _tables(store)
        version = store.conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == SCHEMA_VERSION
    finally:
        store.close()


def test_init_idempotent(tmp_path):
    store = _open(tmp_path)
    store.init_library("42", "users", "u")
    store.close()
    store2 = _open(tmp_path)
    try:
        state = store2.get_library_state()
        assert state["library_id"] == "42"
        assert state["library_version"] == 0
    finally:
        store2.close()


def test_unsupported_schema_version_raises(tmp_path):
    store = _open(tmp_path)
    store.close()
    conn = sqlite3.connect(str(tmp_path / "state" / "zotero_sync.db"))
    conn.execute("PRAGMA user_version = 999")
    conn.commit()
    conn.close()
    with pytest.raises(SyncStateError, match="schema version 999"):
        _open(tmp_path)


def test_schema_v1_to_v2_migration_adds_excluded_items(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    conn = sqlite3.connect(str(state_dir / "zotero_sync.db"))
    # A pretend v1 database: pre-existing user data must survive the
    # migration, only excluded_items is new.
    conn.execute("CREATE TABLE legacy_data (k TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO legacy_data (k) VALUES ('keep-me')")
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()

    store = _open(tmp_path)
    try:
        assert store.conn.execute(
            "PRAGMA user_version"
        ).fetchone()[0] == SCHEMA_VERSION
        tables = _tables(store)
        assert "excluded_items" in tables
        assert "legacy_data" in tables
        # migrated table is usable from the normal DAO surface
        store.upsert_excluded_item(
            key="A", version=3, item_type="book", updated_at="2026-01-01T00:00:00Z"
        )
        assert store.get_excluded_items()["A"]["item_type"] == "book"
        kept = store.conn.execute(
            "SELECT k FROM legacy_data"
        ).fetchone()[0]
        assert kept == "keep-me"
    finally:
        store.close()


def test_migration_not_reaching_target_version_raises(tmp_path, monkeypatch):
    """A migration script that forgets to bump user_version is rejected."""
    import zotero_arxiv_daily.sync.state as state_mod

    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    conn = sqlite3.connect(str(state_dir / "zotero_sync.db"))
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()

    # Same DDL as the real v1->v2 migration but without the version bump.
    monkeypatch.setitem(
        state_mod._MIGRATIONS,
        1,
        "CREATE TABLE excluded_items (key TEXT PRIMARY KEY)",
    )
    with pytest.raises(SyncStateError, match="did not reach"):
        _open(tmp_path)


def test_transaction_rollback_failure_keeps_original_error(tmp_path):
    """m3: if ROLLBACK itself fails, the original exception must surface."""
    store = _open(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="original boom"):
            with store.transaction():
                # Tear the transaction down manually, then raise: the context
                # manager's ROLLBACK hits 'no transaction active'.
                store.conn.execute("ROLLBACK")
                raise RuntimeError("original boom")
        # connection remains usable in autocommit mode afterwards
        assert store.conn.execute("SELECT 1").fetchone()[0] == 1
    finally:
        store.close()


# TR-1.2 --------------------------------------------------------------------

def test_transaction_commits_metadata_watermark_and_links_together(tmp_path):
    store = _open(tmp_path)
    try:
        store.init_library("42", "users", "u")
        with store.transaction():
            _add_item(store)
            store.upsert_collection(key="C1", version=1, name="survey", parent_key=None)
            store.replace_membership("I1", {"C1"})
            store.insert_blob(fingerprint="fp1", content_hash="hash1",
                              dim=3, vector=np.array([1.0, 2.0, 3.0]))
            store.link_item_embedding(item_key="I1", fingerprint="fp1",
                                      content_hash="hash1")
            store.update_library_state(library_version=10,
                                       embedding_fingerprint="fp1",
                                       last_sync_at="now")
        assert store.get_library_state()["library_version"] == 10
        assert store.get_items()["I1"]["content_hash"] == "hash1"
        assert store.get_all_membership() == {"I1": {"C1"}}
        assert store.get_linked_hashes("fp1") == {"I1": "hash1"}
    finally:
        store.close()

    # persists after reopen
    reopened = _open(tmp_path)
    try:
        assert reopened.get_library_state()["library_version"] == 10
        assert reopened.missing_vectors("fp1") == []
    finally:
        reopened.close()


def test_transaction_rolls_back_everything_on_error(tmp_path):
    store = _open(tmp_path)
    try:
        store.init_library("42", "users", "u")
        with pytest.raises(RuntimeError, match="boom"):
            with store.transaction():
                _add_item(store, key="LATER")
                store.update_library_state(library_version=99,
                                           embedding_fingerprint="fp",
                                           last_sync_at="x")
                raise RuntimeError("boom")
        assert store.get_library_state()["library_version"] == 0
        assert "LATER" not in store.get_items()
    finally:
        store.close()


# TR-1.3 --------------------------------------------------------------------

def test_blob_survives_rollback_of_later_main_transaction(tmp_path):
    store = _open(tmp_path)
    try:
        store.init_library("42", "users", "u")
        # Phase B: blob committed in its own transaction
        with store.transaction():
            inserted = store.insert_blob(
                fingerprint="fp1", content_hash="h1", dim=2,
                vector=np.array([0.5, -0.5]))
        assert inserted is True
        # second insert with same key is a no-op
        with store.transaction():
            again = store.insert_blob(
                fingerprint="fp1", content_hash="h1", dim=2,
                vector=np.array([0.5, -0.5]))
        assert again is False
        # Phase C rolls back: watermark/item/links vanish, blob stays
        with pytest.raises(RuntimeError):
            with store.transaction():
                _add_item(store, content_hash="h1")
                store.link_item_embedding(item_key="I1", fingerprint="fp1",
                                          content_hash="h1")
                store.update_library_state(library_version=7,
                                           embedding_fingerprint="fp1",
                                           last_sync_at="t")
                raise RuntimeError("phase c failure")
        assert store.get_library_state()["library_version"] == 0
        assert store.blob_exists("fp1", "h1")
        assert store.unreferenced_blobs() == [("fp1", "h1")]
    finally:
        store.close()


# TR-1.4 --------------------------------------------------------------------

def test_missing_vectors_classifies_items(tmp_path):
    store = _open(tmp_path)
    try:
        _add_item(store, key="OK", abstract="a", content_hash="h_ok")
        _add_item(store, key="EMPTY", abstract="", content_hash=None)
        _add_item(store, key="TRASH", abstract="a", content_hash="h_t", trashed=1)
        _add_item(store, key="NEED", abstract="a", content_hash="h_need")
        with store.transaction():
            store.insert_blob(fingerprint="fp", content_hash="h_ok", dim=1,
                              vector=np.array([1.0]))
            store.link_item_embedding(item_key="OK", fingerprint="fp",
                                      content_hash="h_ok")
        # trash blob too, but trashed item is excluded from missing
        with store.transaction():
            store.insert_blob(fingerprint="fp", content_hash="h_t", dim=1,
                              vector=np.array([1.0]))
            store.link_item_embedding(item_key="TRASH", fingerprint="fp",
                                      content_hash="h_t")
        assert store.missing_vectors("fp") == ["NEED"]
    finally:
        store.close()


# TR-1.5 --------------------------------------------------------------------

def test_unreferenced_blobs_detects_deleted_item_and_old_fingerprint(tmp_path):
    store = _open(tmp_path)
    try:
        with store.transaction():
            _add_item(store, key="A", content_hash="ha")
            _add_item(store, key="B", content_hash="hb")
            for key, hash_, fp in [("A", "ha", "fp1"), ("B", "hb", "fp2")]:
                store.insert_blob(fingerprint=fp, content_hash=hash_, dim=1,
                                  vector=np.array([1.0]))
                store.link_item_embedding(item_key=key, fingerprint=fp,
                                          content_hash=hash_)
        # delete item A: item row + links gone; its immutable blob lingers and
        # is discovered for GC by unreferenced_blobs() (Phase D)
        with store.transaction():
            store.delete_item("A")
        assert store.unreferenced_blobs() == [("fp1", "ha")]
        store.delete_blob("fp1", "ha")
        assert store.unreferenced_blobs() == []
        # simulate model upgrade: links removed, old blobs orphaned
        with store.transaction():
            store.delete_links_for_fingerprint("fp2")
        assert store.unreferenced_blobs() == [("fp2", "hb")]
    finally:
        store.close()


# misc ---------------------------------------------------------------------

def test_vector_blob_roundtrip_and_dim_check(tmp_path):
    store = _open(tmp_path)
    store.close()
    v = np.array([1.5, -2.25, 3.0], dtype=np.float32)
    out = blob_to_vector(vector_to_blob(v), dim=3)
    assert np.array_equal(out, v)
    assert out.dtype == np.float32
    with pytest.raises(SyncStateError):
        blob_to_vector(vector_to_blob(v), dim=2)


def test_collection_delete_cascades_to_membership_only(tmp_path):
    store = _open(tmp_path)
    try:
        with store.transaction():
            _add_item(store, key="I1")
            store.upsert_collection(key="C1", version=1, name="a", parent_key=None)
            store.upsert_collection(key="C2", version=1, name="b", parent_key=None)
            store.replace_membership("I1", {"C1", "C2"})
            store.delete_collection("C1")
        assert store.get_all_membership() == {"I1": {"C2"}}
        assert "I1" in store.get_items()
    finally:
        store.close()


def test_context_manager_closes_store(tmp_path):
    with SyncStore.open(tmp_path / "state") as store:
        assert "library_state" in _tables(store)
    import pytest as _pytest

    with _pytest.raises(sqlite3.ProgrammingError):
        store.conn.execute("SELECT 1")


def test_existing_blobs_empty_input(tmp_path):
    store = _open(tmp_path)
    assert store.existing_blobs("fp", set()) == set()


def test_get_audit_roundtrip_and_missing(tmp_path):
    store = _open(tmp_path)
    with store.transaction():
        audit_id = store.insert_audit(
            started_at="2024-01-01T00:00:00Z",
            finished_at="2024-01-01T00:00:01Z",
            status="success",
            old_version=0,
            new_version=5,
            detail_json="{}",
        )
    row = store.get_audit(audit_id)
    assert row["status"] == "success" and row["new_version"] == 5
    assert store.get_audit(audit_id + 1000) is None
    store.close()
