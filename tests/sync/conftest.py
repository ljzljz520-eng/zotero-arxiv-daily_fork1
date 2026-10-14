"""Shared fixtures for sync tests: a scriptable stub of the pyzotero client.

No network, no unittest.mock — the stub records calls and serves scripted
responses, including multi-page Link:next sequences and injected failures.
"""

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from zotero_arxiv_daily.sync.state import SyncStore
from zotero_arxiv_daily.sync.syncer import CorpusSyncer
from zotero_arxiv_daily.sync.zotero_client import IncrementalZoteroClient


class FakeEmbeddingProvider:
    """Deterministic embed_texts stub with call recording and failure injection.

    Vectors are derived from a sha256 seed of the text so the same text always
    maps to the same vector across runs (cache reuse tests rely on this).
    """

    def __init__(self, *, dim=4, base="fake:test-model", batch_size=64,
                 fail_after_calls=None):
        self.dim = dim
        self._base = base
        self._batch_size = batch_size
        self._fail_after_calls = fail_after_calls
        self.calls: list[list[str]] = []

    def embed_texts(self, texts):
        self.calls.append(list(texts))
        if (
            self._fail_after_calls is not None
            and len(self.calls) > self._fail_after_calls
        ):
            raise RuntimeError("injected embedding failure")
        rows = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
            rows.append(np.random.RandomState(seed).rand(self.dim))
        if not rows:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)

    def embedding_model_name(self):
        return "test-model"

    def embedding_fingerprint_base(self):
        return self._base

    def embedding_batch_size(self):
        return self._batch_size


class _Headers(dict):
    """Case-insensitive header dict (mirrors httpx.Headers get() semantics)."""

    def __init__(self, mapping):
        super().__init__({str(k).lower(): v for k, v in mapping.items()})

    def get(self, key, default=None):
        return super().get(str(key).lower(), default)


DEFAULT_KEY_INFO = {
    "userID": 12345,
    "username": "stub-user",
    "access": {"user": {"library": True, "files": False, "notes": False, "write": False}},
}


class StubPyzotero:
    """Scriptable replacement for pyzotero.Zotero used in sync tests."""

    def __init__(
        self,
        *,
        key_info=DEFAULT_KEY_INFO,
        item_versions=None,
        collection_versions=None,
        item_pages=None,
        collection_pages=None,
        deleted=None,
        trash_pages=None,
        item_key_rows=None,
        collection_key_rows=None,
        library_version=100,
        fail_on=None,
    ):
        self._key_info = key_info
        self._item_versions = dict(item_versions or {})
        self._collection_versions = dict(collection_versions or {})
        self._item_pages = list(item_pages or [])
        self._item_key_rows = list(item_key_rows or [])
        self._collection_pages = list(collection_pages or [])
        self._collection_key_rows = list(collection_key_rows or [])
        self._deleted = deleted
        self._trash_pages = list(trash_pages or [])
        self._fail_on = dict(fail_on or {})
        self.request = SimpleNamespace(
            headers=_Headers({"last-modified-version": str(library_version)})
        )
        self.links: dict[str, str] = {}
        self.calls: list[tuple[str, dict]] = []
        self._follow_queue: list = []
        self._follow_kind: str | None = None
        self._call_index: dict[str, int] = {}

    def _maybe_fail(self, kind):
        idx = self._call_index.get(kind, 0)
        self._call_index[kind] = idx + 1
        exc = self._fail_on.get((kind, idx))
        if exc is not None:
            raise exc

    def key_info(self, **kwargs):
        self.calls.append(("key_info", kwargs))
        self._maybe_fail("key_info")
        if isinstance(self._key_info, Exception):
            raise self._key_info
        return self._key_info

    def item_versions(self, **kwargs):
        self.calls.append(("item_versions", kwargs))
        self._maybe_fail("item_versions")
        return dict(self._item_versions)

    def collection_versions(self, **kwargs):
        self.calls.append(("collection_versions", kwargs))
        self._maybe_fail("collection_versions")
        return dict(self._collection_versions)

    def _start_pages(self, kind, pages):
        if not pages:
            pages = [[]]
        self._follow_kind = kind
        self._follow_queue = list(pages[1:])
        self.links = {"next": f"/stub/{kind}?page=1"} if self._follow_queue else {}
        return list(pages[0])

    def items(self, **kwargs):
        self.calls.append(("items", kwargs))
        self._maybe_fail("items")
        if "itemKey" in kwargs:  # reconcile fallback fetch
            wanted = set(kwargs["itemKey"].split(","))
            return [r for r in self._item_key_rows if r.get("key") in wanted]
        return self._start_pages("items", self._item_pages)

    def collections(self, **kwargs):
        self.calls.append(("collections", kwargs))
        self._maybe_fail("collections")
        if "collectionKey" in kwargs:  # reconcile fallback fetch
            wanted = set(kwargs["collectionKey"].split(","))
            return [
                r for r in self._collection_key_rows if r.get("key") in wanted
            ]
        return self._start_pages("collections", self._collection_pages)

    def trash(self, **kwargs):
        self.calls.append(("trash", kwargs))
        self._maybe_fail("trash")
        return self._start_pages("trash", self._trash_pages)

    def follow(self):
        self.calls.append(("follow", {}))
        self._maybe_fail("follow")
        page = self._follow_queue.pop(0)
        self.links = {"next": "/stub/next"} if self._follow_queue else {}
        return list(page)

    def deleted(self, **kwargs):
        self.calls.append(("deleted", kwargs))
        self._maybe_fail("deleted")
        if isinstance(self._deleted, Exception):
            raise self._deleted
        if self._deleted is None:
            return {"items": [], "collections": [], "searches": [], "tags": []}
        return self._deleted

    def set_library_version_header(self, value):
        self.request = SimpleNamespace(
            headers=_Headers({"last-modified-version": str(value)})
        )

    def drop_library_version_header(self):
        self.request = SimpleNamespace(headers=_Headers({}))


@pytest.fixture()
def make_stub_zot():
    """Factory: ``stub = make_stub_zot(item_pages=[...])``."""
    def _factory(**kwargs):
        return StubPyzotero(**kwargs)
    return _factory


@pytest.fixture()
def stub_config():
    return SimpleNamespace(
        zotero=SimpleNamespace(user_id=12345, api_key="stub-key")
    )


def make_item(key, version, *, title=None, abstract=None, collections=None,
              item_type="journalArticle", added="2026-01-01T00:00:00Z"):
    return {
        "key": key,
        "version": version,
        "data": {
            "title": title if title is not None else f"Title {key}",
            "abstractNote": abstract if abstract is not None else f"Abstract {key}",
            "dateAdded": added,
            "collections": list(collections or []),
            "itemType": item_type,
        },
    }


def make_collection(key, version, name, parent=False):
    return {
        "key": key,
        "version": version,
        "data": {"name": name, "parentCollection": parent},
    }


@pytest.fixture()
def make_syncer(tmp_path, stub_config, make_stub_zot):
    """Factory wiring a CorpusSyncer to stub client/provider + tmp store."""
    def _factory(stub_kwargs=None, *, provider=None, pre_commit_callback=None):
        store = SyncStore.open(tmp_path / ".zotero_sync")
        stub = make_stub_zot(**(stub_kwargs or {}))
        zot_client = IncrementalZoteroClient(stub_config, zot=stub)
        syncer = CorpusSyncer(
            stub_config,
            embedding_provider=provider or FakeEmbeddingProvider(),
            zot_client=zot_client,
            store=store,
            pre_commit_callback=pre_commit_callback,
        )
        return syncer, store, stub

    return _factory


def embedded_texts(provider):
    return [text for batch in provider.calls for text in batch]
