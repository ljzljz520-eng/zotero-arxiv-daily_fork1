"""Tests for IncrementalZoteroClient (TR-3.1 .. TR-3.4)."""

import pytest
from pyzotero.errors import (
    PyZoteroError,
    ResourceNotFoundError,
    UserNotAuthorisedError,
)

from zotero_arxiv_daily.sync.zotero_client import (
    ITEM_TYPES_FILTER,
    IncrementalZoteroClient,
    SyncPermissionError,
    SyncProtocolError,
)


def _item(key, version, title=None, abstract=None, collections=None):
    return {
        "key": key,
        "version": version,
        "data": {
            "title": title or f"Title {key}",
            "abstractNote": abstract or f"Abstract {key}",
            "dateAdded": "2026-01-01T00:00:00Z",
            "collections": collections or [],
            "itemType": "journalArticle",
        },
    }


def test_default_client_constructs_pyzotero_lazily_and_caches(stub_config, monkeypatch):
    """The real pyzotero.Zotero wiring is built once, on first use."""
    import zotero_arxiv_daily.sync.zotero_client as client_mod

    built = []

    class _FakeZotero:
        def __init__(self, *args, **kwargs):
            built.append((args, kwargs))

    monkeypatch.setattr(client_mod.zotero, "Zotero", _FakeZotero)
    client = IncrementalZoteroClient(stub_config)  # no injected zot
    first = client._client()
    second = client._client()
    assert isinstance(first, _FakeZotero)
    assert second is first
    assert len(built) == 1
    args, kwargs = built[0]
    assert args == ("12345", "user", "stub-key")
    assert kwargs == {}


# TR-3.1 --------------------------------------------------------------------

def test_changed_items_params_and_parsing(stub_config, make_stub_zot):
    stub = make_stub_zot(
        item_pages=[[_item("A", 1)], [_item("B", 2), _item("C", 3)]],
    )
    client = IncrementalZoteroClient(stub_config, zot=stub)
    items, pages = client.changed_items(7)

    assert pages == 2
    assert [it["key"] for it in items] == ["A", "B", "C"]
    item_calls = [c for c in stub.calls if c[0] == "items"]
    assert item_calls[0][1] == {
        "since": 7,
        "itemType": ITEM_TYPES_FILTER,
        "limit": 100,
        "includeTrashed": 0,
    }
    assert ("follow", {}) in stub.calls


def test_library_versions_snapshot_reads_lmv(stub_config, make_stub_zot):
    stub = make_stub_zot(item_versions={"A": 1, "B": 3}, library_version=42)
    client = IncrementalZoteroClient(stub_config, zot=stub)
    versions, v = client.library_versions_snapshot()
    assert versions == {"A": 1, "B": 3}
    assert v == 42
    assert ("item_versions", {}) in stub.calls


def test_collection_versions_snapshot(stub_config, make_stub_zot):
    stub = make_stub_zot(collection_versions={"C1": 9})
    client = IncrementalZoteroClient(stub_config, zot=stub)
    assert client.collection_versions_snapshot() == {"C1": 9}


def test_versions_payload_not_a_dict(stub_config, make_stub_zot):
    stub = make_stub_zot()
    stub._item_versions = None
    # bypass the dict coercion in the constructor helper
    stub.item_versions = lambda **kw: ["not", "a", "mapping"]
    client = IncrementalZoteroClient(stub_config, zot=stub)
    with pytest.raises(SyncProtocolError, match="versions mapping"):
        client.library_versions_snapshot()


def test_versions_payload_non_integer_value(stub_config, make_stub_zot):
    stub = make_stub_zot()
    stub.item_versions = lambda **kw: {"A": "v-one"}
    client = IncrementalZoteroClient(stub_config, zot=stub)
    with pytest.raises(SyncProtocolError, match="Malformed versions"):
        client.library_versions_snapshot()


# TR-3.2 --------------------------------------------------------------------

@pytest.mark.parametrize(
    "exc",
    [UserNotAuthorisedError("403"), ResourceNotFoundError("404")],
)
def test_check_key_http_auth_errors(stub_config, make_stub_zot, exc):
    stub = make_stub_zot(key_info=exc)
    client = IncrementalZoteroClient(stub_config, zot=stub)
    with pytest.raises(SyncPermissionError, match="denied access"):
        client.check_key()


def test_check_key_wraps_other_pyzotero_error(stub_config, make_stub_zot):
    from zotero_arxiv_daily.sync.zotero_client import SyncError

    stub = make_stub_zot(key_info=PyZoteroError("boom"))
    client = IncrementalZoteroClient(stub_config, zot=stub)
    with pytest.raises(SyncError):
        client.check_key()


def test_check_key_user_id_mismatch(stub_config, make_stub_zot):
    stub = make_stub_zot(
        key_info={"userID": 999, "username": "x",
                  "access": {"user": {"library": True}}}
    )
    client = IncrementalZoteroClient(stub_config, zot=stub)
    with pytest.raises(SyncPermissionError, match="user 999"):
        client.check_key()


def test_check_key_library_access_false(stub_config, make_stub_zot):
    stub = make_stub_zot(
        key_info={"userID": 12345, "username": "x",
                  "access": {"user": {"library": False}}}
    )
    client = IncrementalZoteroClient(stub_config, zot=stub)
    with pytest.raises(SyncPermissionError, match="library access"):
        client.check_key()


def test_check_key_malformed_payload(stub_config, make_stub_zot):
    stub = make_stub_zot(key_info={"username": "no-user-id"})
    client = IncrementalZoteroClient(stub_config, zot=stub)
    with pytest.raises(SyncProtocolError):
        client.check_key()


def test_check_key_success_identity(stub_config, make_stub_zot):
    stub = make_stub_zot()
    client = IncrementalZoteroClient(stub_config, zot=stub)
    identity = client.check_key()
    assert identity.user_id == 12345
    assert identity.username == "stub-user"
    assert identity.library_access is True


# TR-3.3 --------------------------------------------------------------------

def test_pagination_walks_two_follows(stub_config, make_stub_zot):
    stub = make_stub_zot(
        item_pages=[[_item("A", 1)], [_item("B", 2)], [_item("C", 3)]],
    )
    client = IncrementalZoteroClient(stub_config, zot=stub)
    items, pages = client.changed_items(0)
    assert pages == 3
    assert [it["key"] for it in items] == ["A", "B", "C"]
    assert stub.calls.count(("follow", {})) == 2


def test_missing_lmv_header_raises(stub_config, make_stub_zot):
    stub = make_stub_zot(item_versions={"A": 1})
    stub.drop_library_version_header()
    client = IncrementalZoteroClient(stub_config, zot=stub)
    with pytest.raises(SyncProtocolError, match="Last-Modified-Version"):
        client.library_versions_snapshot()


def test_invalid_lmv_header_raises(stub_config, make_stub_zot):
    stub = make_stub_zot(item_versions={"A": 1})
    stub.set_library_version_header("not-an-int")
    client = IncrementalZoteroClient(stub_config, zot=stub)
    with pytest.raises(SyncProtocolError):
        client.library_versions_snapshot()


def test_data_endpoint_403_maps_to_permission_error(stub_config, make_stub_zot):
    stub = make_stub_zot(
        item_versions={"A": 1},
        fail_on={("item_versions", 0): UserNotAuthorisedError("revoked")},
    )
    client = IncrementalZoteroClient(stub_config, zot=stub)
    with pytest.raises(SyncPermissionError):
        client.library_versions_snapshot()


# TR-3.4 --------------------------------------------------------------------

def test_deleted_since_parses_categories(stub_config, make_stub_zot):
    stub = make_stub_zot(deleted={
        "items": ["I1", "I2"], "collections": ["C9"],
        "searches": [], "tags": [],
    })
    client = IncrementalZoteroClient(stub_config, zot=stub)
    result = client.deleted_since(5)
    assert result["items"] == ["I1", "I2"]
    assert result["collections"] == ["C9"]
    deleted_call = [c for c in stub.calls if c[0] == "deleted"][0]
    assert deleted_call[1] == {"since": 5}


def test_deleted_since_malformed(stub_config, make_stub_zot):
    stub = make_stub_zot(deleted=["not", "a", "dict"])
    client = IncrementalZoteroClient(stub_config, zot=stub)
    with pytest.raises(SyncProtocolError):
        client.deleted_since(1)


def test_trashed_keys_paginates(stub_config, make_stub_zot):
    stub = make_stub_zot(trash_pages=[[{"key": "T1"}], [{"key": "T2"}, {"key": "T3"}]])
    client = IncrementalZoteroClient(stub_config, zot=stub)
    keys, pages = client.trashed_keys(3)
    assert keys == ["T1", "T2", "T3"]
    assert pages == 2


def test_trashed_keys_malformed_row(stub_config, make_stub_zot):
    stub = make_stub_zot(trash_pages=[[{"no_key": True}]])
    client = IncrementalZoteroClient(stub_config, zot=stub)
    with pytest.raises(SyncProtocolError):
        client.trashed_keys(0)
