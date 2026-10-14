"""CorpusSyncer orchestration tests (TR-4.1 .. TR-4.13)."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from pyzotero.errors import PyZoteroError, UserNotAuthorisedError

from zotero_arxiv_daily.sync.content import build_embedding_text, content_hash
from zotero_arxiv_daily.sync.syncer import CorpusSyncer
from zotero_arxiv_daily.sync.zotero_client import (
    SyncError,
    SyncPermissionError,
    IncrementalZoteroClient,
)

from tests.sync.conftest import (
    FakeEmbeddingProvider,
    StubPyzotero,
    embedded_texts,
    make_collection,
    make_item,
)

FP = "fake:test-model:dim4"


def _new_round(store, config, stub_kwargs, *, provider=None,
               pre_commit_callback=None):
    provider = provider or FakeEmbeddingProvider()
    stub = StubPyzotero(**(stub_kwargs or {}))
    zot_client = IncrementalZoteroClient(config, zot=stub)
    syncer = CorpusSyncer(
        config,
        embedding_provider=provider,
        zot_client=zot_client,
        store=store,
        pre_commit_callback=pre_commit_callback,
    )
    return syncer, provider, stub


def _emb(item):
    return build_embedding_text(item["data"]["title"], item["data"]["abstractNote"])


def _blob_pairs(store):
    return {
        (r[0], r[1])
        for r in store.conn.execute(
            "SELECT fingerprint, content_hash FROM embedding_blobs"
        ).fetchall()
    }


def _audit_statuses(store):
    return [
        r[0]
        for r in store.conn.execute("SELECT status FROM sync_audit ORDER BY id")
    ]


# TR-4.1 --------------------------------------------------------------------

def test_first_full_sync(make_syncer):
    a = make_item("A", 1, collections=["CHILD"])
    b = make_item("B", 1, collections=[])
    c = make_item("C", 1, abstract="", collections=["P"])
    syncer, store, _ = make_syncer(
        {
            "library_version": 5,
            "item_versions": {"A": 1, "B": 1, "C": 1},
            "collection_versions": {"P": 1, "CHILD": 1},
            "item_pages": [[a, b, c]],
            "collection_pages": [
                [
                    make_collection("P", 1, "Parent"),
                    make_collection("CHILD", 1, "Child", parent="P"),
                ]
            ],
        }
    )

    report = syncer.sync()
    state = store.get_library_state()

    assert state["library_version"] == 5
    assert state["embedding_fingerprint"] == FP
    assert set(store.get_items()) == {"A", "B", "C"}
    links = store.get_linked_hashes(FP)
    assert set(links) == {"A", "B"}
    assert store.missing_vectors(FP) == []
    assert store.get_all_membership()["A"] == {"CHILD"}
    assert report.status == "success"
    assert report.added == ["A", "B", "C"]
    assert report.embeddings_computed == 2
    assert report.embeddings_reused == 0
    assert _blob_pairs(store) == {(FP, links["A"]), (FP, links["B"])}

    papers, matrix = syncer.load_corpus()
    paths = {p.key: p.paths for p in papers}
    assert paths["A"] == ["Parent/Child"]
    assert matrix.shape == (2, 4)


# TR-4.2 --------------------------------------------------------------------

def test_incremental_add_content_metadata(make_syncer, stub_config):
    a1 = make_item("A", 1)
    b1 = make_item("B", 1)
    c1 = make_item("C", 1)
    syncer, store, _ = make_syncer(
        {
            "library_version": 5,
            "item_versions": {"A": 1, "B": 1, "C": 1},
            "item_pages": [[a1, b1, c1]],
        }
    )
    syncer.sync()

    a2 = make_item("A", 2, abstract="Abstract A — rewritten")
    b2 = make_item("B", 2)  # version bump, identical content
    d = make_item("D", 1)
    syncer2, provider2, _ = _new_round(
        store,
        stub_config,
        {
            "library_version": 6,
            "item_versions": {"A": 2, "B": 2, "C": 1, "D": 1},
            "item_pages": [[a2, b2, d]],
        },
    )
    report = syncer2.sync()

    assert report.added == ["D"]
    assert report.content_changed == ["A"]
    assert report.metadata_only == ["B"]
    assert sorted(embedded_texts(provider2)) == sorted([_emb(a2), _emb(d)])
    assert store.get_library_state()["library_version"] == 6


def test_incremental_link_hashes_stable(make_syncer, stub_config):
    """B's linked content hash is unchanged; A's is replaced."""
    a1, b1 = make_item("A", 1), make_item("B", 1)
    syncer, store, _ = make_syncer(
        {"library_version": 5, "item_versions": {"A": 1, "B": 1},
         "item_pages": [[a1, b1]]}
    )
    syncer.sync()
    before = store.get_linked_hashes(FP)

    a2 = make_item("A", 2, abstract="new abstract for A")
    b2 = make_item("B", 2)
    syncer2, _, _ = _new_round(
        store,
        stub_config,
        {"library_version": 6,
         "item_versions": {"A": 2, "B": 2},
         "item_pages": [[a2, b2]]},
    )
    syncer2.sync()
    after = store.get_linked_hashes(FP)
    assert after["B"] == before["B"]
    assert after["A"] != before["A"]


# TR-4.3 --------------------------------------------------------------------

def test_move_zero_embedding_with_paths(make_syncer, stub_config):
    a1 = make_item("A", 1, collections=["P"])
    syncer, store, _ = make_syncer(
        {
            "library_version": 5,
            "item_versions": {"A": 1},
            "collection_versions": {"P": 1, "Q": 1},
            "item_pages": [[a1]],
            "collection_pages": [
                [make_collection("P", 1, "Papers"),
                 make_collection("Q", 1, "Archive")]
            ],
        }
    )
    syncer.sync()

    a2 = make_item("A", 2, collections=["Q"])
    syncer2, provider2, _ = _new_round(
        store,
        stub_config,
        {
            "library_version": 6,
            "item_versions": {"A": 2},
            "collection_versions": {"P": 1, "Q": 1},
            "item_pages": [[a2]],
        },
    )
    report = syncer2.sync()

    assert report.embeddings_computed == 0
    assert embedded_texts(provider2) == []
    assert len(report.moved) == 1
    moved = report.moved[0]
    assert moved.key == "A"
    assert moved.added_paths == ["Archive"]
    assert moved.removed_paths == ["Papers"]
    assert store.get_all_membership()["A"] == {"Q"}
    papers, _ = syncer2.load_corpus()
    assert papers[0].paths == ["Archive"]


# TR-4.4 --------------------------------------------------------------------

def test_tombstone_and_trash_deletions_with_gc(make_syncer, stub_config):
    a, b, c = make_item("A", 1), make_item("B", 1), make_item("C", 1)
    syncer, store, _ = make_syncer(
        {"library_version": 5,
         "item_versions": {"A": 1, "B": 1, "C": 1},
         "item_pages": [[a, b, c]]}
    )
    syncer.sync()
    assert len(_blob_pairs(store)) == 3

    syncer2, _, _ = _new_round(
        store,
        stub_config,
        {
            "library_version": 6,
            "item_versions": {"C": 1},
            "item_pages": [[]],
            "deleted": {"items": ["A"], "collections": [], "searches": [], "tags": []},
            "trash_pages": [[{"key": "B"}]],
        },
    )
    report = syncer2.sync()

    assert report.tombstoned == ["A"]
    assert report.trashed == ["B"]
    assert set(store.get_items()) == {"C"}
    assert set(store.get_linked_hashes(FP)) == {"C"}
    assert store.unreferenced_blobs() == []  # Phase D GC ran


def test_reconciled_deletion(make_syncer, stub_config):
    """Vanished key with no tombstone/trash is deleted by versions reconcile."""
    a, b = make_item("A", 1), make_item("B", 1)
    syncer, store, _ = make_syncer(
        {"library_version": 5,
         "item_versions": {"A": 1, "B": 1},
         "item_pages": [[a, b]]}
    )
    syncer.sync()

    syncer2, _, _ = _new_round(
        store,
        stub_config,
        {"library_version": 6, "item_versions": {"B": 1}, "item_pages": [[]]},
    )
    report = syncer2.sync()
    assert report.reconciled == ["A"]
    assert set(store.get_items()) == {"B"}
    assert store.unreferenced_blobs() == []


# TR-4.5 --------------------------------------------------------------------

def test_pagination_interruption_then_resume(make_syncer, stub_config):
    pages = [[make_item("A", 1)], [make_item("B", 1)], [make_item("C", 1)]]
    syncer, store, _ = make_syncer(
        {
            "library_version": 5,
            "item_versions": {"A": 1, "B": 1, "C": 1},
            "item_pages": pages,
            "fail_on": {("follow", 0): PyZoteroError("page 2 network failure")},
        }
    )
    with pytest.raises(SyncError):
        syncer.sync()
    # m6: no identity row before the Phase C commit.
    assert store.get_library_state() is None
    assert store.get_items() == {}
    assert _audit_statuses(store) == ["failed"]

    syncer2, provider2, _ = _new_round(
        store,
        stub_config,
        {"library_version": 5,
         "item_versions": {"A": 1, "B": 1, "C": 1},
         "item_pages": pages},
    )
    report = syncer2.sync()
    assert report.status == "success"
    assert set(store.get_items()) == {"A", "B", "C"}
    assert store.get_library_state()["library_version"] == 5
    assert len(embedded_texts(provider2)) == 3
    assert store.missing_vectors(FP) == []


# TR-4.6 --------------------------------------------------------------------

def test_model_upgrade_switches_fingerprint(make_syncer, stub_config):
    a, b = make_item("A", 1), make_item("B", 1)
    syncer, store, _ = make_syncer(
        {"library_version": 5,
         "item_versions": {"A": 1, "B": 1},
         "item_pages": [[a, b]]}
    )
    syncer.sync()

    v2_provider = FakeEmbeddingProvider(base="fake:test-model:v2")
    syncer2, _, _ = _new_round(
        store,
        stub_config,
        {"library_version": 5,
         "item_versions": {"A": 1, "B": 1},
         "item_pages": [[]]},
        provider=v2_provider,
    )
    report = syncer2.sync()
    fp2 = "fake:test-model:v2:dim4"

    assert report.model_upgraded is True
    assert report.old_fingerprint == FP
    assert report.new_fingerprint == fp2
    assert store.get_library_state()["library_version"] == 5
    assert set(store.get_linked_hashes(fp2)) == {"A", "B"}
    assert store.get_linked_hashes(FP) == {}
    assert store.unreferenced_blobs() == []
    assert report.embeddings_computed == 2


def test_interrupted_run_reuses_blobs_with_no_prior_watermark(make_syncer, stub_config):
    """Crash before any commit: fp is rediscovered from surviving blobs."""
    a, b, c, d = (make_item(k, 1) for k in ("A", "B", "C", "D"))
    v2_kwargs = {
        "library_version": 5,
        "item_versions": {"A": 1, "B": 1, "C": 1, "D": 1},
        "item_pages": [[a, b, c, d]],
    }
    syncer, store, _ = make_syncer(
        v2_kwargs,
        provider=FakeEmbeddingProvider(base="fake:test-model:v2",
                                       batch_size=2, fail_after_calls=1),
    )
    with pytest.raises(RuntimeError):
        syncer.sync()
    fp2 = "fake:test-model:v2:dim4"
    assert len(_blob_pairs(store)) == 2

    syncer2, provider2, _ = _new_round(
        store, stub_config, v2_kwargs,
        provider=FakeEmbeddingProvider(base="fake:test-model:v2", batch_size=2),
    )
    report = syncer2.sync()
    # Only the two not-yet-stored hashes are embedded.
    assert len(embedded_texts(provider2)) == 2
    assert report.embeddings_reused == 2
    assert set(store.get_linked_hashes(fp2)) == {"A", "B", "C", "D"}
    assert store.missing_vectors(fp2) == []


# TR-4.7 --------------------------------------------------------------------

def test_phase_b_interruption_preserves_batch_blobs(make_syncer, stub_config):
    items = [make_item(k, 1) for k in ("A", "B", "C", "D")]
    kwargs = {
        "library_version": 5,
        "item_versions": {"A": 1, "B": 1, "C": 1, "D": 1},
        "item_pages": [items],
    }
    syncer, store, _ = make_syncer(
        kwargs, provider=FakeEmbeddingProvider(batch_size=2, fail_after_calls=1)
    )
    with pytest.raises(RuntimeError):
        syncer.sync()
    # m6: a first-round failure before Phase C writes no identity row at all.
    assert store.get_library_state() is None
    assert len(_blob_pairs(store)) == 2  # batch 1 committed independently

    syncer2, provider2, _ = _new_round(
        store, stub_config, kwargs,
        provider=FakeEmbeddingProvider(batch_size=2),
    )
    syncer2.sync()
    assert len(embedded_texts(provider2)) == 2
    assert store.missing_vectors(FP) == []


# TR-4.8 --------------------------------------------------------------------

def test_phase_c_callback_failure_rolls_back_everything(make_syncer, stub_config):
    a, b = make_item("A", 1), make_item("B", 1)
    kwargs = {"library_version": 5,
              "item_versions": {"A": 1, "B": 1},
              "item_pages": [[a, b]]}

    def boom(_report):
        raise RuntimeError("injected before COMMIT")

    syncer, store, _ = make_syncer(kwargs, pre_commit_callback=boom)
    with pytest.raises(RuntimeError):
        syncer.sync()

    assert store.get_library_state() is None
    assert store.get_items() == {}
    assert len(_blob_pairs(store)) == 2  # Phase B blobs survive
    assert _audit_statuses(store) == ["failed"]

    syncer2, provider2, _ = _new_round(store, stub_config, kwargs)
    syncer2.sync()
    assert embedded_texts(provider2) == []  # all blobs reused
    assert set(store.get_items()) == {"A", "B"}
    assert store.get_library_state()["library_version"] == 5


# TR-4.9 --------------------------------------------------------------------

def test_permission_failure_then_resume(make_syncer, stub_config):
    a = make_item("A", 1)
    syncer, store, _ = make_syncer(
        {"library_version": 5, "item_versions": {"A": 1}, "item_pages": [[a]]}
    )
    syncer.sync()

    syncer2, _, _ = _new_round(
        store,
        stub_config,
        {"key_info": UserNotAuthorisedError("403 revoked"),
         "library_version": 6},
    )
    with pytest.raises(SyncPermissionError):
        syncer2.sync()
    assert store.get_library_state()["library_version"] == 5
    assert set(store.get_items()) == {"A"}
    assert _audit_statuses(store) == ["success", "failed_permission"]

    d = make_item("D", 1)
    syncer3, _, _ = _new_round(
        store,
        stub_config,
        {"library_version": 6,
         "item_versions": {"A": 1, "D": 1},
         "item_pages": [[d]]},
    )
    report = syncer3.sync()
    assert report.status == "success"
    assert report.added == ["D"]
    assert store.get_library_state()["library_version"] == 6


def test_state_library_identity_mismatch_rejected(make_syncer):
    a = make_item("A", 1)
    syncer, store, _ = make_syncer(
        {"library_version": 5, "item_versions": {"A": 1}, "item_pages": [[a]]}
    )
    syncer.sync()

    # Config + key now legitimately point at a DIFFERENT library (999).
    other_config = SimpleNamespace(
        zotero=SimpleNamespace(user_id=999, api_key="other-key")
    )
    other_stub = StubPyzotero(
        key_info={
            "userID": 999,
            "username": "someone-else",
            "access": {"user": {"library": True}},
        },
        item_versions={"A": 1},
        library_version=5,
    )
    other_zot = IncrementalZoteroClient(other_config, zot=other_stub)
    syncer2 = CorpusSyncer(
        other_config,
        embedding_provider=FakeEmbeddingProvider(),
        zot_client=other_zot,
        store=store,
    )
    with pytest.raises(SyncError, match="bound to library"):
        syncer2.sync()
    # watermark untouched
    assert store.get_library_state()["library_version"] == 5


# TR-4.10 -------------------------------------------------------------------

def test_object_newer_than_snapshot_picked_up_next_round(make_syncer, stub_config):
    a = make_item("A", 5)
    b_future = make_item("B", 7)
    syncer, store, _ = make_syncer(
        {
            "library_version": 5,
            "item_versions": {"A": 5},
            "item_pages": [[a, b_future]],
        }
    )
    syncer.sync()
    assert set(store.get_items()) == {"A"}
    assert store.get_library_state()["library_version"] == 5

    syncer2, provider2, _ = _new_round(
        store,
        stub_config,
        {
            "library_version": 7,
            "item_versions": {"A": 5, "B": 7},
            "item_pages": [[make_item("B", 7)]],
        },
    )
    syncer2.sync()
    assert set(store.get_items()) == {"A", "B"}
    assert store.get_library_state()["library_version"] == 7
    assert len(embedded_texts(provider2)) == 1


def test_noop_round_short_circuits(make_syncer, stub_config):
    a = make_item("A", 5)
    kwargs = {"library_version": 5, "item_versions": {"A": 5}, "item_pages": [[a]]}
    syncer, store, stub = make_syncer(kwargs)
    syncer.sync()

    syncer2, provider2, stub2 = _new_round(store, stub_config, kwargs)
    # m7: a no-op round must still refresh last_sync_at.
    store.conn.execute(
        "UPDATE library_state SET last_sync_at = '2000-01-01T00:00:00Z'"
    )
    report = syncer2.sync()
    assert report.status == "no_changes"
    called = {c[0] for c in stub2.calls}
    assert called == {"key_info", "item_versions"}
    assert embedded_texts(provider2) == []
    assert _audit_statuses(store) == ["success", "no_changes"]
    assert store.get_library_state()["last_sync_at"] > "2000-01-01"


# TR-4.11 -------------------------------------------------------------------

def test_audit_json_files_match_db_rows(make_syncer, stub_config):
    a = make_item("A", 5)
    syncer, store, _ = make_syncer(
        {"library_version": 5, "item_versions": {"A": 5}, "item_pages": [[a]]}
    )
    syncer.sync()
    syncer2, _, _ = _new_round(
        store,
        stub_config,
        {"key_info": UserNotAuthorisedError("403"), "library_version": 6},
    )
    with pytest.raises(SyncPermissionError):
        syncer2.sync()

    db_rows = store.conn.execute(
        "SELECT status, detail_json FROM sync_audit ORDER BY id"
    ).fetchall()
    statuses = [r[0] for r in db_rows]
    assert statuses == ["success", "failed_permission"]

    files = sorted((store.audit_dir).glob("*.json"))
    assert len(files) == 2
    file_payloads = [json.loads(p.read_text()) for p in files]
    for status, detail in db_rows:
        payload = json.loads(detail)
        assert payload in file_payloads
        assert payload["status"] == status
    success = next(p for p in file_payloads if p["status"] == "success")
    for field in (
        "started_at", "finished_at", "library_id", "old_version", "new_version",
        "full_sync", "pages", "added", "moved", "tombstoned", "trashed",
        "reconciled", "embeddings_computed", "embeddings_reused",
        "model_upgraded", "counts",
    ):
        assert field in success
    failure = next(p for p in file_payloads if p["status"] == "failed_permission")
    assert "403" in failure["error"]
    assert success["library_id"] == "12345"


# TR-4.12 -------------------------------------------------------------------

def test_abstract_filled_and_collection_deleted(make_syncer, stub_config):
    c = make_item("C", 1, abstract="", collections=["P"])
    d = make_item("D", 1, collections=["P"])
    syncer, store, _ = make_syncer(
        {
            "library_version": 5,
            "item_versions": {"C": 1, "D": 1},
            "collection_versions": {"P": 1},
            "item_pages": [[c, d]],
            "collection_pages": [[make_collection("P", 1, "Papers")]],
        }
    )
    syncer.sync()
    assert set(store.get_linked_hashes(FP)) == {"D"}

    c2 = make_item("C", 2, abstract="abstract added later", collections=["P"])
    syncer2, provider2, _ = _new_round(
        store,
        stub_config,
        {
            "library_version": 6,
            "item_versions": {"C": 2, "D": 1},
            # collection P is gone remotely
            "collection_versions": {},
            "item_pages": [[c2]],
        },
    )
    report = syncer2.sync()

    assert sorted(embedded_texts(provider2)) == [_emb(c2)]
    moved_keys = {m.key for m in report.moved}
    assert moved_keys == {"C", "D"}
    for m in report.moved:
        assert m.removed_paths == ["Papers"]
        assert m.added_paths == []
    assert store.get_all_membership() == {}
    assert store.get_collections() == {}
    papers, _ = syncer2.load_corpus()
    assert {p.key for p in papers} == {"C", "D"}
    assert all(p.paths == [] for p in papers)
    assert store.missing_vectors(FP) == []


def test_all_empty_abstract_library_commits_bare_fp(make_syncer):
    """First sync with no embeddable items publishes the bare fingerprint."""
    a = make_item("A", 1, title="No abstract", abstract="")
    syncer, store, _ = make_syncer(
        {"library_version": 5, "item_versions": {"A": 1}, "item_pages": [[a]]}
    )
    report = syncer.sync()
    assert report.status == "success"
    assert report.embeddings_computed == 0
    assert store.get_library_state()["embedding_fingerprint"] == "fake:test-model"
    assert store.get_linked_hashes(FP) == {}
    papers, matrix = syncer.load_corpus()
    assert papers == []
    assert matrix.shape == (0, 0)
    assert _audit_statuses(store) == ["success"]


def test_unknown_collection_tombstone_ignored(make_syncer):
    a = make_item("A", 1)
    syncer, store, _ = make_syncer(
        {
            "library_version": 5,
            "item_versions": {"A": 1},
            "item_pages": [[a]],
            "deleted": {"collections": ["GHOST"], "items": []},
        }
    )
    report = syncer.sync()
    assert report.status == "success"
    assert report.deleted_collections == []


def test_inconsistent_dim_across_batches_aborts(make_syncer):
    class DriftingProvider(FakeEmbeddingProvider):
        def embedding_batch_size(self):
            return 1

        def embed_texts(self, texts):
            import numpy as _np

            self.calls.append(list(texts))
            dim = 4 if len(self.calls) == 1 else 5
            return _np.ones((len(texts), dim), dtype=_np.float32)

    a, b = make_item("A", 1), make_item("B", 1, abstract="second distinct abstract")
    syncer, store, _ = make_syncer(
        {"library_version": 5,
         "item_versions": {"A": 1, "B": 1},
         "item_pages": [[a, b]]},
        provider=DriftingProvider(),
    )
    with pytest.raises(SyncError, match="Inconsistent embedding dimensions"):
        syncer.sync()
    assert store.get_library_state() is None
    # first batch blob may be persisted, but no links / watermark published
    assert store.get_linked_hashes(FP) == {}
    assert _audit_statuses(store) == ["failed"]


def test_gc_failure_is_best_effort(make_syncer, stub_config, monkeypatch):
    a, b = make_item("A", 1), make_item("B", 1)
    syncer, store, _ = make_syncer(
        {"library_version": 5,
         "item_versions": {"A": 1, "B": 1},
         "item_pages": [[a, b]]}
    )
    syncer.sync()
    # make the NEXT sync's GC blow up without affecting the commit
    def boom(self):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(type(store), "unreferenced_blobs", boom)
    a2 = make_item("A", 6, abstract="changed again")
    syncer2, _, _ = _new_round(
        store,
        stub_config,
        {"library_version": 6,
         "item_versions": {"A": 6, "B": 1},
         "item_pages": [[a2]]},
    )
    report = syncer2.sync()
    assert report.status == "success"
    assert store.get_library_state()["library_version"] == 6


def test_audit_file_write_failure_is_best_effort(make_syncer, monkeypatch):
    import zotero_arxiv_daily.sync.syncer as syncer_mod

    a = make_item("A", 1)
    syncer, store, _ = make_syncer(
        {"library_version": 5, "item_versions": {"A": 1}, "item_pages": [[a]]}
    )

    def boom(path, text):
        raise OSError("audit dir unwritable")

    monkeypatch.setattr(syncer_mod, "write_atomic", boom)
    report = syncer.sync()
    assert report.status == "success"
    assert _audit_statuses(store) == ["success"]


def test_parse_date_fallbacks(make_syncer):
    from datetime import datetime

    a = make_item("A", 1, added="2024-06-01T12:30:00+00:00")  # isoformat
    b = make_item("B", 1, abstract="other body", added="garbage")  # unparseable
    c = make_item("C", 1, abstract="third body", added=None)  # missing
    syncer, _, _ = make_syncer(
        {"library_version": 5,
         "item_versions": {"A": 1, "B": 1, "C": 1},
         "item_pages": [[a, b, c]]}
    )
    syncer.sync()
    papers, _ = syncer.load_corpus()
    by_key = {p.key: p for p in papers}
    assert by_key["A"].added_date == datetime(2024, 6, 1, 12, 30)
    assert by_key["B"].added_date == datetime.min
    assert by_key["C"].added_date == datetime.min


# FR-5 step 6 drift + protocol guards --------------------------------------

def test_drift_fallback_fetches_missing_change(make_syncer, stub_config):
    """versions map shows B at v6 but since pages omit it: fetch by key."""
    a, b = make_item("A", 1), make_item("B", 1)
    syncer, store, _ = make_syncer(
        {"library_version": 5,
         "item_versions": {"A": 1, "B": 1},
         "item_pages": [[a, b]]}
    )
    syncer.sync()

    b2 = make_item("B", 6, abstract="rewritten via drift fetch")
    syncer2, provider2, stub2 = _new_round(
        store,
        stub_config,
        {
            "library_version": 6,
            "item_versions": {"A": 1, "B": 6},
            "item_pages": [[]],  # since page conspicuously omits B
            "item_key_rows": [b2],
        },
    )
    report = syncer2.sync()
    assert report.content_changed == ["B"]
    assert sorted(embedded_texts(provider2)) == [_emb(b2)]
    item_calls = [c for c in stub2.calls if c[0] == "items"]
    assert any("itemKey" in c[1] for c in item_calls)


def test_malformed_item_payload_raises(make_syncer):
    syncer, store, _ = make_syncer(
        {"library_version": 5,
         "item_versions": {"A": 1},
         "item_pages": [[{"key": "A"}]]}  # no "data" dict
    )
    with pytest.raises(SyncError):
        syncer.sync()
    # m6: no identity row before the Phase C commit.
    assert store.get_library_state() is None
    assert _audit_statuses(store) == ["failed"]


def test_embed_shape_mismatch_aborts(make_syncer):
    class BrokenProvider(FakeEmbeddingProvider):
        def embed_texts(self, texts):
            self.calls.append(list(texts))
            import numpy as _np
            return _np.ones((len(texts) + 1, self.dim), dtype=_np.float32)

    a, b = make_item("A", 1), make_item("B", 1)
    syncer, store, _ = make_syncer(
        {"library_version": 5,
         "item_versions": {"A": 1, "B": 1},
         "item_pages": [[a, b]]},
        provider=BrokenProvider(),
    )
    with pytest.raises(SyncError, match="embed_texts returned shape"):
        syncer.sync()
    assert store.get_library_state() is None
    assert _audit_statuses(store) == ["failed"]


def test_load_corpus_before_sync_raises(make_syncer):
    syncer, _, _ = make_syncer(
        {"library_version": 5, "item_versions": {}, "item_pages": [[]]}
    )
    with pytest.raises(SyncError, match="No synced corpus"):
        syncer.load_corpus()


# TR-4.13 -------------------------------------------------------------------

def test_load_corpus_rows_align_with_matrix(make_syncer):
    a, b = make_item("A", 1), make_item("B", 1, abstract="distinct abstract text")
    syncer, store, _ = make_syncer(
        {"library_version": 5,
         "item_versions": {"A": 1, "B": 1},
         "item_pages": [[a, b]]}
    )
    syncer.sync()

    papers, matrix = syncer.load_corpus()
    assert matrix.shape == (2, 4)
    links = store.get_linked_hashes(FP)
    for i, paper in enumerate(papers):
        blob_row = store.conn.execute(
            "SELECT dim, vector FROM embedding_blobs WHERE fingerprint = ? AND content_hash = ?",
            (FP, links[paper.key]),
        ).fetchone()
        decoded = np.frombuffer(blob_row["vector"], dtype=np.float32)
        assert np.array_equal(matrix[i], decoded)
        assert paper.key in {"A", "B"}

# defensive guards ----------------------------------------------------------

def test_item_in_versions_but_newer_than_lmv_skipped(make_syncer):
    """Versions map racing ahead: payload v7 with snapshot V=5 waits."""
    a = make_item("A", 5)
    b_future = make_item("B", 7)
    syncer, store, _ = make_syncer(
        {
            "library_version": 5,
            "item_versions": {"A": 5, "B": 7},  # contradictory snapshot race
            "item_pages": [[a, b_future]],
        }
    )
    report = syncer.sync()
    assert set(store.get_items()) == {"A"}
    assert report.added == ["A"]
    assert store.get_library_state()["library_version"] == 5


def test_payload_no_newer_than_local_ignored(make_syncer, stub_config):
    """A repeated same-version payload must not re-embed or change anything."""
    a = make_item("A", 1)
    syncer, store, _ = make_syncer(
        {"library_version": 5, "item_versions": {"A": 1}, "item_pages": [[a]]}
    )
    syncer.sync()
    syncer2, provider2, _ = _new_round(
        store,
        stub_config,
        {
            "library_version": 6,
            "item_versions": {"A": 1},
            "item_pages": [[make_item("A", 1, abstract="stale? ignored")]],
        },
    )
    report = syncer2.sync()
    assert report.content_changed == []
    assert embedded_texts(provider2) == []
    assert store.get_items()["A"]["abstract"] == "Abstract A"
    assert store.get_library_state()["library_version"] == 6


def test_collection_payload_not_in_versions_ignored(make_syncer):
    a = make_item("A", 1, collections=[])
    ghost = make_collection("GHOST", 3, "Ghost")
    syncer, store, _ = make_syncer(
        {
            "library_version": 5,
            "item_versions": {"A": 1},
            "item_pages": [[a]],
            "collection_versions": {},
            "collection_pages": [[ghost]],
        }
    )
    report = syncer.sync()
    assert report.status == "success"
    assert "GHOST" not in store.get_collections()


def test_malformed_collection_payload_raises(make_syncer):
    a = make_item("A", 1, collections=[])
    syncer, store, _ = make_syncer(
        {
            "library_version": 5,
            "item_versions": {"A": 1},
            "item_pages": [[a]],
            "collection_versions": {"C": 1},
            "collection_pages": [[{"key": "C", "data": {"name": "C"}}]],
        }
    )
    with pytest.raises(SyncError, match="Malformed collection payload"):
        syncer.sync()
    assert _audit_statuses(store) == ["failed"]


def test_collection_newer_than_snapshot_skipped(make_syncer):
    a = make_item("A", 1, collections=[])
    future_coll = make_collection("C", 7, "Future")
    syncer, store, _ = make_syncer(
        {
            "library_version": 5,
            "item_versions": {"A": 1},
            "item_pages": [[a]],
            "collection_versions": {"C": 7},
            "collection_pages": [[future_coll]],
        }
    )
    report = syncer.sync()
    assert report.status == "success"
    assert "C" not in store.get_collections()


def test_upgrade_crash_recovers_dim_from_surviving_blob(make_syncer, stub_config):
    """Crashed v2 run left a blob; resume prefers the old dimension (493-497)."""
    a, b = make_item("A", 1), make_item("B", 1)
    syncer, store, _ = make_syncer(
        {"library_version": 5,
         "item_versions": {"A": 1, "B": 1},
         "item_pages": [[a, b]]}
    )
    syncer.sync()

    fp2 = "fake:test-model:v2:dim4"
    a_hash = content_hash("Title A", "Abstract A")
    with store.transaction():
        assert store.insert_blob(
            fingerprint=fp2,
            content_hash=a_hash,
            dim=4,
            vector=np.ones(4, dtype=np.float32),
        )

    v2 = FakeEmbeddingProvider(base="fake:test-model:v2")
    syncer2, _, _ = _new_round(
        store,
        stub_config,
        {"library_version": 5,
         "item_versions": {"A": 1, "B": 1},
         "item_pages": [[]]},
        provider=v2,
    )
    report = syncer2.sync()
    assert report.model_upgraded is True
    assert report.new_fingerprint == fp2
    assert report.embeddings_computed == 1  # only B paid for
    assert embedded_texts(v2) == [build_embedding_text("Title B", "Abstract B")]
    assert set(store.get_linked_hashes(fp2)) == {"A", "B"}


def test_load_corpus_collection_path_fallbacks(make_syncer):
    a = make_item("A", 1, collections=["CHILD"])
    child = make_collection("CHILD", 1, "Child")
    syncer, store, _ = make_syncer(
        {"library_version": 5,
         "item_versions": {"A": 1},
         "item_pages": [[a]],
         "collection_versions": {"CHILD": 1},
         "collection_pages": [[child]]}
    )
    syncer.sync()

    # empty-name collection falls back to its key; missing collection -> key
    with store.transaction():
        store.upsert_collection(key="EMPTY", version=1, name="", parent_key=None)
        store.replace_membership("A", {"CHILD", "EMPTY", "GHOST"})

    papers, matrix = syncer.load_corpus()
    assert matrix.shape == (1, 4)
    assert papers[0].paths == ["Child", "EMPTY", "GHOST"]


def test_stale_collection_payload_ignored(make_syncer, stub_config):
    a = make_item("A", 1, collections=["C"])
    coll = make_collection("C", 1, "See")
    kwargs = {
        "library_version": 5,
        "item_versions": {"A": 1},
        "item_pages": [[a]],
        "collection_versions": {"C": 1},
        "collection_pages": [[coll]],
    }
    syncer, store, _ = make_syncer(kwargs)
    syncer.sync()

    same_coll_again = make_collection("C", 1, "See renamed? ignored")
    syncer2, _, _ = _new_round(
        store,
        stub_config,
        {
            "library_version": 6,
            "item_versions": {"A": 1},
            "item_pages": [[a]],
            "collection_versions": {"C": 1},
            "collection_pages": [[same_coll_again]],
        },
    )
    report = syncer2.sync()
    assert report.status == "success"
    assert store.get_collections()["C"]["name"] == "See"
    assert store.get_library_state()["library_version"] == 6


def test_missing_vector_blocks_watermark_commit(make_syncer, monkeypatch):
    """Headline invariant: blobs without links must never publish a watermark."""
    a = make_item("A", 1)
    syncer, store, _ = make_syncer(
        {"library_version": 5, "item_versions": {"A": 1}, "item_pages": [[a]]}
    )
    # Sabotage the link write: the blob exists but the item never points to it.
    monkeypatch.setattr(type(store), "link_item_embedding", lambda *a, **kw: None)
    with pytest.raises(SyncError, match="Refusing to commit sync"):
        syncer.sync()
    # m6: no identity row before the Phase C commit.
    assert store.get_library_state() is None
    assert _audit_statuses(store) == ["failed"]
    # the blob may remain (Phase B committed it), but nothing was announced
    assert store.get_linked_hashes(FP) == {}


def test_upgrade_crash_whole_batch_reused_partial_resume(make_syncer, stub_config):
    """A fully pre-existing batch is skipped (551); only missing C is paid."""
    a, b, c = make_item("A", 1), make_item("B", 1), make_item("C", 1)
    syncer, store, _ = make_syncer(
        {"library_version": 5,
         "item_versions": {"A": 1, "B": 1, "C": 1},
         "item_pages": [[a, b, c]]}
    )
    syncer.sync()

    fp2 = "fake:test-model:v2:dim4"
    with store.transaction():
        for title, abstract in (("Title A", "Abstract A"),
                                ("Title B", "Abstract B")):
            store.insert_blob(
                fingerprint=fp2,
                content_hash=content_hash(title, abstract),
                dim=4,
                vector=np.ones(4, dtype=np.float32),
            )

    v2 = FakeEmbeddingProvider(base="fake:test-model:v2", batch_size=2)
    syncer2, _, _ = _new_round(
        store,
        stub_config,
        {"library_version": 5,
         "item_versions": {"A": 1, "B": 1, "C": 1},
         "item_pages": [[]]},
        provider=v2,
    )
    report = syncer2.sync()
    assert report.model_upgraded is True
    assert report.embeddings_computed == 1
    assert embedded_texts(v2) == [build_embedding_text("Title C", "Abstract C")]
    assert store.get_library_state()["embedding_fingerprint"] == fp2
    assert set(store.get_linked_hashes(fp2)) == {"A", "B", "C"}


def test_lazy_provider_construction_and_close(tmp_path, monkeypatch):
    state_dir = str(tmp_path / "lazy_state")
    cfg = SimpleNamespace(
        executor=SimpleNamespace(reranker="api"),
        zotero=SimpleNamespace(
            user_id=12345,
            api_key="stub-key",
            state_dir=state_dir,
            get=lambda key, default=None: state_dir if key == "state_dir" else default,
        ),
    )
    import zotero_arxiv_daily.sync.syncer as syncer_mod

    monkeypatch.setattr(
        syncer_mod,
        "get_reranker_cls",
        lambda name: (lambda config: FakeEmbeddingProvider()),
    )
    syncer = CorpusSyncer(cfg)
    provider = syncer.provider()
    assert syncer.provider() is provider  # cached
    assert syncer.store() is syncer.store()  # store lazily opened and cached
    assert syncer.zot() is syncer.zot()  # lazy Zotero client cached
    syncer.close()
    assert syncer._store is None
    syncer.close()  # idempotent no-op


# M1: itemType whitelist drift (journalArticle -> book -> journalArticle) ----

def test_item_changes_to_noneligible_type_leaves_and_can_return(
    make_syncer, stub_config
):
    a1 = make_item("A", 1)
    syncer, store, _ = make_syncer(
        {"library_version": 5, "item_versions": {"A": 1}, "item_pages": [[a1]]}
    )
    syncer.sync()
    assert set(store.get_items()) == {"A"}

    # Round 2: A became a book. The server-side itemType filter hides it from
    # the since=5 pages, but the versions map still carries A:2 and the
    # by-key reconcile fetch reveals the new type.
    a2_book = make_item("A", 2, item_type="book")
    syncer2, provider2, stub2 = _new_round(
        store,
        stub_config,
        {
            "library_version": 6,
            "item_versions": {"A": 2},
            "item_pages": [[]],
            "item_key_rows": [a2_book],
        },
    )
    report2 = syncer2.sync()
    assert set(store.get_items()) == set()
    excluded = store.get_excluded_items()
    assert excluded["A"]["version"] == 2
    assert excluded["A"]["item_type"] == "book"
    assert "A" in report2.reconciled
    assert embedded_texts(provider2) == []  # books never reach the embedder

    # Round 3: no remote change -> cheap no-op; the excluded key must not be
    # re-fetched by key every single round.
    syncer3, provider3, stub3 = _new_round(
        store,
        stub_config,
        {
            "library_version": 6,
            "item_versions": {"A": 2},
            "item_pages": [[]],
            "item_key_rows": [a2_book],
        },
    )
    report3 = syncer3.sync()
    assert report3.status == "no_changes"
    assert [c for c in stub3.calls if "itemKey" in c[1]] == []
    assert set(store.get_items()) == set()

    # Round 4: back to an eligible type -> excluded-drift fetch readmits it.
    a3 = make_item("A", 3)
    syncer4, provider4, stub4 = _new_round(
        store,
        stub_config,
        {
            "library_version": 7,
            "item_versions": {"A": 3},
            "item_pages": [[]],
            "item_key_rows": [a3],
        },
    )
    report4 = syncer4.sync()
    assert report4.added == ["A"]
    assert set(store.get_items()) == {"A"}
    assert store.get_excluded_items() == {}
    assert embedded_texts(provider4) == [_emb(a3)]


def test_nontracked_noneligible_item_is_excluded_quietly(make_syncer):
    """A never-tracked book discovered via the version sweep is remembered but
    never enters the corpus."""
    book = make_item("X", 1, item_type="book")
    syncer, store, stub = make_syncer(
        {
            "library_version": 5,
            "item_versions": {"X": 1},
            "item_pages": [[]],
            "item_key_rows": [book],
        }
    )
    report = syncer.sync()
    assert store.get_items() == {}
    assert store.get_excluded_items()["X"]["item_type"] == "book"
    assert report.added == []
    assert report.reconciled == []
    assert report.status == "success"


# M3: keys present in the versions snapshot but missed by the since pages ----

def test_item_missing_from_since_pages_swept_from_versions(make_syncer, stub_config):
    b = make_item("B", 1)
    syncer, store, _ = make_syncer(
        {"library_version": 5, "item_versions": {"B": 1}, "item_pages": [[b]]}
    )
    syncer.sync()

    # A is created (v2) while paging; the live since=5 window does not surface
    # it, but the versions snapshot taken at V=6 does.
    a = make_item("A", 2)
    syncer2, provider2, stub2 = _new_round(
        store,
        stub_config,
        {
            "library_version": 6,
            "item_versions": {"A": 2, "B": 1},
            "item_pages": [[]],
            "item_key_rows": [a],
        },
    )
    report2 = syncer2.sync()
    assert report2.added == ["A"]
    assert set(store.get_items()) == {"A", "B"}
    key_calls = [c for c in stub2.calls if "itemKey" in c[1]]
    assert any("A" in c[1]["itemKey"] for c in key_calls)
    assert embedded_texts(provider2) == [_emb(a)]


def test_missing_key_newer_than_snapshot_is_not_fetched(make_syncer):
    """A version-snapshot key above V waits for the next round (AC-15)."""
    a = make_item("A", 7)
    syncer, store, stub = make_syncer(
        {
            "library_version": 6,
            "item_versions": {"A": 7},
            "item_pages": [[]],
            "item_key_rows": [a],
        }
    )
    syncer.sync()
    assert store.get_items() == {}
    assert [c for c in stub.calls if "itemKey" in c[1]] == []
    assert store.get_library_state()["library_version"] == 6


def test_collection_missing_from_since_pages_swept_from_versions(make_syncer):
    d = make_collection("D", 1, "Dyn")
    syncer, store, stub = make_syncer(
        {
            "library_version": 5,
            "item_versions": {},
            "collection_versions": {"D": 1},
            "collection_pages": [[]],
            "collection_key_rows": [d],
        }
    )
    syncer.sync()
    assert "D" in store.get_collections()
    coll_calls = [
        c for c in stub.calls
        if c[0] == "collections" and "collectionKey" in c[1]
    ]
    assert any("D" in c[1]["collectionKey"] for c in coll_calls)


# m1: per-endpoint request counters land on the audit report ----------------

def test_report_endpoints_called_counts(make_syncer):
    a = make_item("A", 1)
    syncer, _, _ = make_syncer(
        {"library_version": 5, "item_versions": {"A": 1}, "item_pages": [[a]]}
    )
    report = syncer.sync()
    assert isinstance(report.endpoints_called, dict)
    for endpoint in ("key_info", "item_versions", "collection_versions",
                     "items", "deleted", "trash"):
        assert report.endpoints_called.get(endpoint) == 1, endpoint


# M3 (collections): tracked collection drift missed by the since pages -------

def test_tracked_collection_drift_fetched_by_key(make_syncer, stub_config):
    c1 = make_collection("C", 1, "See old")
    syncer, store, _ = make_syncer(
        {
            "library_version": 5,
            "item_versions": {},
            "collection_versions": {"C": 1},
            "collection_pages": [[c1]],
        }
    )
    syncer.sync()

    # C renamed to v2 but the since pages do not surface it.
    c2 = make_collection("C", 2, "See new")
    syncer2, _, stub2 = _new_round(
        store,
        stub_config,
        {
            "library_version": 6,
            "item_versions": {},
            "collection_versions": {"C": 2},
            "collection_pages": [[]],
            "collection_key_rows": [c2],
        },
    )
    syncer2.sync()
    assert store.get_collections()["C"]["name"] == "See new"
    coll_calls = [
        c for c in stub2.calls
        if c[0] == "collections" and "collectionKey" in c[1]
    ]
    assert any("C" in c[1]["collectionKey"] for c in coll_calls)


# m8: a by-key payload with a bad version field is a protocol error ---------

def test_item_with_malformed_version_raises(make_syncer):
    bad = {"key": "A", "version": "oops", "data": {"itemType": "journalArticle"}}
    syncer, store, _ = make_syncer(
        {
            "library_version": 5,
            "item_versions": {"A": 1},
            "item_pages": [[]],
            "item_key_rows": [bad],
        }
    )
    with pytest.raises(SyncError, match="Malformed item payload"):
        syncer.sync()
    assert store.get_library_state() is None
