"""Narrow incremental-sync facade over pyzotero.

Only the read endpoints needed by the sync protocol are exposed:

* key identity / permissions
* full ``format=versions`` snapshots (items + collections)
* ``?since=`` paginated item/collection payloads
* deletion tombstones and trash

The facade's only protocol-specific jobs are request parameter shape,
pagination walking, ``Last-Modified-Version`` extraction and HTTP error
classification. All decisions about what to do with the responses live in
:mod:`zotero_arxiv_daily.sync.syncer`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyzotero import zotero
from pyzotero.errors import (
    PyZoteroError,
    ResourceNotFoundError,
    UserNotAuthorisedError,
)

#: Eligible Zotero item types, mirroring the previous full-fetch behaviour.
ITEM_TYPES_FILTER = "conferencePaper || journalArticle || preprint"
#: Whitelist used when evaluating payloads fetched by key (no server-side
#: itemType filter applies to ``itemKey=`` requests).
ALLOWED_ITEM_TYPES = frozenset({"conferencePaper", "journalArticle", "preprint"})
PAGE_LIMIT = 100
_LMV_HEADER = "last-modified-version"


class SyncError(Exception):
    """Base class for sync failures."""


class SyncPermissionError(SyncError):
    """API key is invalid, lacks library access, or targets another library."""


class SyncProtocolError(SyncError):
    """Remote response is missing required fields/headers or is malformed."""


@dataclass(frozen=True)
class KeyIdentity:
    user_id: int
    username: str | None
    library_access: bool


class IncrementalZoteroClient:
    def __init__(self, config: Any, zot: Any | None = None, library_type: str = "user"):
        self._config = config
        self._zot = zot
        self.library_id = str(config.zotero.user_id)
        self.api_key = config.zotero.api_key
        # pyzotero wants "user"/"group"; its API path becomes "users"/"groups"
        self.library_type = library_type
        #: logical endpoint -> number of HTTP requests issued this sync
        self.request_counts: dict[str, int] = {}

    def _bump(self, endpoint: str, count: int = 1) -> None:
        self.request_counts[endpoint] = self.request_counts.get(endpoint, 0) + count

    # ------------------------------------------------------------------
    # pyzotero wiring / error mapping
    # ------------------------------------------------------------------

    def _client(self) -> Any:
        if self._zot is None:
            self._zot = zotero.Zotero(self.library_id, self.library_type, self.api_key)
        return self._zot

    @staticmethod
    def _call(description: str, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (UserNotAuthorisedError, ResourceNotFoundError) as exc:
            raise SyncPermissionError(
                f"Zotero API denied access while {description}: {exc}"
            ) from exc
        except PyZoteroError as exc:
            raise SyncError(f"Zotero API error while {description}: {exc}") from exc

    @staticmethod
    def _read_library_version(z: Any) -> int:
        request = getattr(z, "request", None)
        headers = getattr(request, "headers", None)
        raw = headers.get(_LMV_HEADER) if headers is not None else None
        if raw is None:
            raise SyncProtocolError(
                "Missing Last-Modified-Version header in Zotero response"
            )
        try:
            return int(raw)
        except (TypeError, ValueError) as exc:
            raise SyncProtocolError(
                f"Invalid Last-Modified-Version header: {raw!r}"
            ) from exc

    def _paginate(self, first_page: Any, endpoint: str) -> tuple[list[Any], int]:
        """Walk Link: rel="next" pages. Returns (flattened rows, page count)."""
        z = self._client()
        rows = list(first_page)
        pages = 1
        while getattr(z, "links", None) and z.links.get("next"):
            rows.extend(self._call("following next result page", z.follow))
            pages += 1
            self._bump(endpoint)
        return rows, pages

    # ------------------------------------------------------------------
    # endpoints
    # ------------------------------------------------------------------

    def check_key(self) -> KeyIdentity:
        z = self._client()
        self._bump("key_info")
        info = self._call("checking API key permissions", z.key_info)
        try:
            user_id = int(info["userID"])
            username = info.get("username")
            library_access = bool(
                info.get("access", {}).get("user", {}).get("library", False)
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise SyncProtocolError(f"Malformed key info response: {info!r}") from exc

        if user_id != int(self.library_id):
            raise SyncPermissionError(
                f"API key belongs to Zotero user {user_id} but config.zotero."
                f"user_id is {self.library_id}; refusing to mix libraries in "
                "one state directory"
            )
        if not library_access:
            raise SyncPermissionError(
                "API key has no user library access (access.user.library is "
                "false); a read-enabled key is required"
            )
        return KeyIdentity(user_id=user_id, username=username, library_access=True)

    def library_versions_snapshot(self) -> tuple[dict[str, int], int]:
        """Full item key→version map plus the snapshot library version V."""
        z = self._client()
        self._bump("item_versions")
        versions = self._call("fetching item version snapshot", z.item_versions)
        version = self._read_library_version(z)
        return self._coerce_versions(versions), version

    def collection_versions_snapshot(self) -> dict[str, int]:
        z = self._client()
        self._bump("collection_versions")
        versions = self._call(
            "fetching collection version snapshot", z.collection_versions
        )
        return self._coerce_versions(versions)

    def changed_items(self, since: int) -> tuple[list[dict], int]:
        z = self._client()
        self._bump("items")
        first = self._call(
            f"fetching changed items since {since}",
            z.items,
            since=int(since),
            itemType=ITEM_TYPES_FILTER,
            limit=PAGE_LIMIT,
            includeTrashed=0,
        )
        return self._paginate(first, "items")

    def items_by_keys(self, keys: list[str]) -> list[dict]:
        """Fetch full item payloads by explicit keys (chunks of 50).

        Used as the reconcile fallback when the versions map shows drift that
        a ``since=`` page did not include. No ``itemType`` filter is sent: the
        caller must check ``data.itemType`` against ALLOWED_ITEM_TYPES, since
        an item may have changed into a non-eligible type.
        """
        z = self._client()
        rows: list[dict] = []
        for i in range(0, len(keys), 50):
            chunk = keys[i : i + 50]
            self._bump("items_by_keys")
            page = self._call(
                f"fetching {len(chunk)} items by key",
                z.items,
                itemKey=",".join(chunk),
                limit=50,
            )
            rows.extend(page)
        return rows

    def changed_collections(self, since: int) -> tuple[list[dict], int]:
        z = self._client()
        self._bump("collections")
        first = self._call(
            f"fetching changed collections since {since}",
            z.collections,
            since=int(since),
            limit=PAGE_LIMIT,
        )
        return self._paginate(first, "collections")

    def collections_by_keys(self, keys: list[str]) -> list[dict]:
        """Fetch full collection payloads by explicit keys (chunks of 50)."""
        z = self._client()
        rows: list[dict] = []
        for i in range(0, len(keys), 50):
            chunk = keys[i : i + 50]
            self._bump("collections_by_keys")
            page = self._call(
                f"fetching {len(chunk)} collections by key",
                z.collections,
                collectionKey=",".join(chunk),
                limit=50,
            )
            rows.extend(page)
        return rows

    def deleted_since(self, since: int) -> dict[str, list[str]]:
        z = self._client()
        self._bump("deleted")
        payload = self._call(
            f"fetching deletions since {since}", z.deleted, since=int(since)
        )
        if not isinstance(payload, dict):
            raise SyncProtocolError(f"Malformed /deleted payload: {payload!r}")
        return {
            "items": list(payload.get("items", [])),
            "collections": list(payload.get("collections", [])),
            "searches": list(payload.get("searches", [])),
            "tags": list(payload.get("tags", [])),
        }

    def trashed_keys(self, since: int) -> tuple[list[str], int]:
        z = self._client()
        self._bump("trash")
        first = self._call(
            f"fetching trashed items since {since}",
            z.trash,
            since=int(since),
            limit=PAGE_LIMIT,
        )
        rows, pages = self._paginate(first, "trash")
        keys = []
        for row in rows:
            try:
                keys.append(row["key"])
            except (KeyError, TypeError) as exc:
                raise SyncProtocolError(f"Malformed trash item: {row!r}") from exc
        return keys, pages

    @staticmethod
    def _coerce_versions(payload: Any) -> dict[str, int]:
        if not isinstance(payload, dict):
            raise SyncProtocolError(f"Expected versions mapping, got {payload!r}")
        try:
            return {str(key): int(ver) for key, ver in payload.items()}
        except (TypeError, ValueError) as exc:
            raise SyncProtocolError(f"Malformed versions mapping: {payload!r}") from exc
