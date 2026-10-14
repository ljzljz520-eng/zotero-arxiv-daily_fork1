"""Incremental corpus sync orchestration (FR-3 .. FR-15).

Four phases, enforced by code structure (see NFR-3):

* **Phase A — pull**: every remote page is fetched into memory. Any failure
  leaves the database untouched.
* **Phase B — blob prefetch**: missing content-addressed vectors are embedded
  in batches and each batch is committed in its own small transaction. Blobs
  are immutable, so committing them early is harmless; a crash here means the
  next run reuses them and pays for nothing twice.
* **Phase C — atomic commit**: items, collections, membership, deletions,
  embedding links, the watermark V, the new fingerprint and the success audit
  row are all written in ONE ``BEGIN IMMEDIATE`` transaction. Right before
  COMMIT, :meth:`SyncStore.missing_vectors` must be empty; otherwise the whole
  transaction rolls back and the watermark stays at W.
* **Phase D — post-commit**: unreferenced blob GC, atomic audit JSON file and
  a human-readable log line. Failures of these derived operations never roll
  back committed state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from loguru import logger

from ..protocol import CorpusPaper
from ..reranker import get_reranker_cls
from ..reranker.base import BaseReranker
from .audit import (
    STATUS_FAILED,
    STATUS_FAILED_PERMISSION,
    STATUS_NO_CHANGES,
    STATUS_SUCCESS,
    AuditReport,
    MovedItem,
    write_atomic,
)
from .content import build_embedding_text, content_hash, normalize_text
from .state import SyncStore, blob_to_vector, utc_now_iso
from .zotero_client import (
    ALLOWED_ITEM_TYPES,
    SyncError,
    SyncPermissionError,
    SyncProtocolError,
    IncrementalZoteroClient,
)

_DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


@dataclass
class _ItemState:
    """Projected post-sync state of one tracked item."""

    key: str
    version: int
    item_type: str
    title: str
    abstract: str
    added_date: str | None
    content_hash: str
    collections: set[str] = field(default_factory=set)


class CorpusSyncer:
    def __init__(
        self,
        config: Any,
        embedding_provider: BaseReranker | None = None,
        zot_client: IncrementalZoteroClient | None = None,
        store: SyncStore | None = None,
        *,
        pre_commit_callback: Callable[[AuditReport], None] | None = None,
    ):
        self.config = config
        self._provider = embedding_provider
        self._zot = zot_client
        self._store = store
        self._owns_store = store is None
        self._pre_commit_callback = pre_commit_callback

    # ------------------------------------------------------------------
    # lazy dependencies
    # ------------------------------------------------------------------

    def store(self) -> SyncStore:
        if self._store is None:
            state_dir = self.config.zotero.get("state_dir", ".zotero_sync")
            self._store = SyncStore.open(state_dir)
            self._owns_store = True
        return self._store

    def zot(self) -> IncrementalZoteroClient:
        if self._zot is None:
            self._zot = IncrementalZoteroClient(self.config)
        return self._zot

    def provider(self) -> BaseReranker:
        if self._provider is None:
            cls = get_reranker_cls(self.config.executor.reranker)
            self._provider = cls(self.config)
        return self._provider

    def close(self) -> None:
        if self._owns_store and self._store is not None:
            self._store.close()
            self._store = None

    # ------------------------------------------------------------------
    # entry point
    # ------------------------------------------------------------------

    def sync(self) -> AuditReport:
        store = self.store()
        report = AuditReport(
            started_at=utc_now_iso(),
            library_id=self.zot().library_id,
            library_type="users",
        )
        try:
            return self._sync(store, report)
        except SyncPermissionError as exc:
            self._record_failure(store, report, STATUS_FAILED_PERMISSION, exc)
            raise
        except Exception as exc:
            self._record_failure(store, report, STATUS_FAILED, exc)
            raise

    # ------------------------------------------------------------------
    # sync core
    # ------------------------------------------------------------------

    def _sync(self, store: SyncStore, report: AuditReport) -> AuditReport:
        zot = self.zot()
        provider = self.provider()

        # Step 1: identity / permission check (FR-4)
        identity = zot.check_key()
        report.username = identity.username

        state = store.get_library_state()
        first_sync = state is None
        if state is not None and state["library_id"] != str(identity.user_id):
            # FR-3: never silently mix two libraries in one state directory.
            raise SyncError(
                f"Refusing to sync: state directory is bound to library "
                f"{state['library_id']} but the API key belongs to "
                f"{identity.user_id}. Point zotero.state_dir at a fresh "
                "directory to sync the other library."
            )

        # The identity row itself is written inside the Phase C transaction on
        # a first sync, so a Phase A failure leaves no state behind at all.
        watermark = 0 if first_sync else int(state["library_version"])
        stored_fp: str | None = None if first_sync else state["embedding_fingerprint"]
        report.old_version = watermark
        report.full_sync = first_sync
        report.old_fingerprint = stored_fp

        # Step 2: full versions snapshot + snapshot library version V (FR-5)
        versions, snapshot_v = zot.library_versions_snapshot()

        base = provider.embedding_fingerprint_base()
        stored_base = (
            BaseReranker.split_fingerprint(stored_fp)[0] if stored_fp else None
        )
        model_upgraded = stored_base is not None and stored_base != base
        report.model_upgraded = model_upgraded

        if snapshot_v == watermark and not model_upgraded and stored_fp is not None:
            # No remote change and same embedding model: this is the cheap
            # no-op round (NFR-5) — only key_info + versions were requested.
            report.new_version = snapshot_v
            report.new_fingerprint = stored_fp
            report.endpoints_called = dict(zot.request_counts)
            return self._finish_no_changes(store, report)

        # ---- Phase A: pull everything into memory (FR-5 steps 3-5) ----
        changed_rows, item_pages = zot.changed_items(watermark)
        changed_coll_rows, coll_pages = zot.changed_collections(watermark)
        deleted_payload = zot.deleted_since(watermark)
        trashed_keys, trash_pages = zot.trashed_keys(watermark)
        coll_versions = zot.collection_versions_snapshot()
        report.new_version = snapshot_v
        report.pages = {
            "items": item_pages,
            "collections": coll_pages,
            "trash": trash_pages,
        }
        # Snapshot now (so a later failure audit shows where we got stuck);
        # refreshed right before the Phase C commit with reconcile calls.
        report.endpoints_called = dict(zot.request_counts)

        local_items = store.get_items()
        local_colls = store.get_collections()
        local_membership = store.get_all_membership()

        # ---------- collections projection / reconciliation (FR-6) ----
        deleted_colls = set(local_colls) - set(coll_versions)
        coll_tombstones = set(deleted_payload["collections"])
        # tombstones for collections we never tracked are not audited
        deleted_colls_audited = sorted(deleted_colls)
        report.deleted_collections = deleted_colls_audited
        if coll_tombstones - deleted_colls:
            # Remote tombstone for a key we do not know: nothing to delete, but
            # versions reconciliation remains the source of truth.
            pass

        new_colls: dict[str, dict] = {
            key: dict(row) for key, row in local_colls.items() if key not in deleted_colls
        }
        changed_coll_map: dict[str, dict] = {}
        for row in changed_coll_rows:
            key = row.get("key")
            if key:
                changed_coll_map[key] = row

        # Reconcile collections the since pages missed: locally tracked with a
        # versions mismatch (drift), or present in the snapshot but unknown
        # locally (pagination-window gap, cf. the item sweep below).
        coll_fetch: set[str] = set()
        for key, local_row in local_colls.items():
            remote_version = coll_versions.get(key)
            if (
                key not in deleted_colls
                and remote_version is not None
                and remote_version != int(local_row["version"])
                and key not in changed_coll_map
            ):
                coll_fetch.add(key)
        for key, remote_version in coll_versions.items():
            if (
                remote_version <= snapshot_v
                and key not in new_colls
                and key not in changed_coll_map
            ):
                coll_fetch.add(key)
        if coll_fetch:
            for row in zot.collections_by_keys(sorted(coll_fetch)):
                key = row.get("key")
                if key:
                    changed_coll_map[key] = row

        for key, row in changed_coll_map.items():
            data = row.get("data") or {}
            if not key or key not in coll_versions:
                # absent from the versions snapshot: deleted concurrently
                continue
            try:
                version = int(row["version"])
            except (KeyError, TypeError, ValueError) as exc:
                raise SyncProtocolError(f"Malformed collection payload: {row!r}") from exc
            if version > snapshot_v:
                # FR-8/AC-15: objects newer than the snapshot wait for the
                # next since=snapshot_v round.
                continue
            existing = new_colls.get(key)
            if existing is not None and version <= int(existing["version"]):
                continue  # no newer collection data to apply
            parent = data.get("parentCollection")
            new_colls[key] = {
                "version": version,
                "name": normalize_text(data.get("name")),
                "parent_key": parent if parent else None,
            }

        # ---------- item deletion classification (FR-7) -----------------
        tracked = set(local_items)
        tombstoned = set(deleted_payload["items"]) & tracked
        trashed = set(trashed_keys) & tracked - tombstoned
        reconciled = (tracked - set(versions)) - tombstoned - trashed
        deleted_items = tombstoned | trashed | reconciled
        report.tombstoned = sorted(tombstoned)
        report.trashed = sorted(trashed)
        report.reconciled = sorted(reconciled)

        # ---------- changed item projection (FR-7) ----------------------
        changed_map: dict[str, dict] = {}
        for row in changed_rows:
            key = row.get("key")
            if not key or not isinstance(row.get("data"), dict):
                raise SyncProtocolError(f"Malformed item payload: {row!r}")
            changed_map[key] = row

        # Reconcile by explicit key fetch (FR-5 step 6 + pagination safety):
        #  * drift: tracked item whose versions version disagrees with local
        #    but the since pages did not surface it;
        #  * excluded drift: a previously out-of-whitelist item whose version
        #    moved (it may have changed back to an eligible type);
        #  * missing key: present in the versions snapshot at v<=V but neither
        #    tracked, excluded nor carried by a since page (a concurrent write
        #    shifted the live offset window while paging).
        local_excluded = store.get_excluded_items()
        fetch_keys: set[str] = set()
        for key, local_row in local_items.items():
            if key in deleted_items:
                continue
            remote_version = versions.get(key)
            if (
                remote_version is not None
                and remote_version != int(local_row["version"])
                and key not in changed_map
            ):
                fetch_keys.add(key)
        for key, excluded_row in local_excluded.items():
            remote_version = versions.get(key)
            if (
                remote_version is not None
                and remote_version != int(excluded_row["version"])
                and key not in changed_map
            ):
                fetch_keys.add(key)
        for key, remote_version in versions.items():
            if (
                remote_version <= snapshot_v
                and key not in local_items
                and key not in local_excluded
                and key not in changed_map
            ):
                fetch_keys.add(key)
        if fetch_keys:
            for row in zot.items_by_keys(sorted(fetch_keys)):
                key = row.get("key")
                if key and isinstance(row.get("data"), dict):
                    changed_map[key] = row

        projected: dict[str, _ItemState] = {}
        active_text: dict[str, tuple[str, str]] = {}
        # membership overrides for items not carried in any payload (e.g.
        # membership stripped because a collection was deleted)
        membership_override: dict[str, set[str]] = {}
        # tracked items whose current type is outside the whitelist (M1):
        # leave the corpus like a deletion, and remember them in
        # excluded_items so they are not re-fetched every round.
        type_excluded: set[str] = set()
        excluded_upserts: dict[str, tuple[int, str]] = {}
        excluded_removals: set[str] = set()

        for key in sorted(changed_map):
            if key in deleted_items or key not in versions:
                # Gone from the authoritative snapshot, or already classified.
                continue
            row = changed_map[key]
            try:
                version = int(row["version"])
            except (KeyError, TypeError, ValueError) as exc:
                raise SyncProtocolError(f"Malformed item payload: {row!r}") from exc
            if version > snapshot_v:
                # AC-15: leave >V objects to the next round.
                continue
            local_for_guard = local_items.get(key)
            if local_for_guard is not None and version <= int(
                local_for_guard["version"]
            ):
                # Never downgrade to a payload no newer than what is stored
                # (a correct server never sends one via since=W anyway).
                continue
            data = row["data"]
            item_type = str(data.get("itemType", ""))
            if item_type not in ALLOWED_ITEM_TYPES:
                # itemKey fetches carry no itemType filter; the versions
                # snapshot can surface an item that changed into a book etc.
                excluded_upserts[key] = (version, item_type)
                if local_for_guard is not None:
                    type_excluded.add(key)
                continue
            if key in local_excluded:
                # Eligible again: drop the exclusion and (re-)admit below.
                excluded_removals.add(key)
            title = normalize_text(data.get("title"))
            abstract = normalize_text(data.get("abstractNote"))
            cols = set(data.get("collections") or []) - deleted_colls
            chash = content_hash(title, abstract)
            state_row = _ItemState(
                key=key,
                version=version,
                item_type=item_type,
                title=title,
                abstract=abstract,
                added_date=data.get("dateAdded"),
                content_hash=chash,
                collections=cols,
            )
            projected[key] = state_row
            if abstract:
                active_text[key] = (chash, build_embedding_text(title, abstract))

            local_row = local_items.get(key)
            if local_row is None:
                report.added.append(key)
                continue

            # Compare against the RAW old membership: a carried payload that
            # drops a concurrently-deleted collection is still a membership
            # change for this item and must be audited as a move.
            old_cols = local_membership.get(key, set())
            hash_changed = chash != (local_row["content_hash"] or "")
            if cols != old_cols:
                report.moved.append(
                    MovedItem(
                        key=key,
                        added_paths=sorted(
                            self._collection_path(k, new_colls) for k in cols - old_cols
                        ),
                        removed_paths=sorted(
                            self._collection_path(k, local_colls)
                            for k in old_cols - cols
                        ),
                    )
                )
            if hash_changed:
                report.content_changed.append(key)
            elif version != int(local_row["version"]) and cols == old_cols:
                report.metadata_only.append(key)

        if type_excluded:
            # Audited under "reconciled": from the corpus perspective the item
            # has been removed by reconciliation against the remote snapshot.
            deleted_items |= type_excluded
            report.reconciled = sorted(set(report.reconciled) | type_excluded)

        # Membership-only effects of collection deletion on untouched items.
        for key, old_cols in local_membership.items():
            if key in projected or key in deleted_items:
                continue
            reduced = old_cols - deleted_colls
            if reduced != old_cols:
                membership_override[key] = reduced
                report.moved.append(
                    MovedItem(
                        key=key,
                        added_paths=[],
                        removed_paths=sorted(
                            self._collection_path(k, local_colls)
                            for k in old_cols - reduced
                        ),
                    )
                )

        # Active items that survive unchanged still need valid links (FR-12);
        # they are cheap idempotent upserts and make the commit-time invariant
        # self-healing.
        for key, local_row in local_items.items():
            if key in deleted_items or key in projected or not local_row["abstract"]:
                continue
            if local_row["trashed"]:  # pragma: no cover - trashed subset deleted
                continue
            active_text[key] = (
                local_row["content_hash"],
                build_embedding_text(local_row["title"], local_row["abstract"]),
            )

        # ---- Phase B: content-addressed blob prefetch (FR-11/FR-12) ----
        target_fp, computed = self._embed_blobs(
            store,
            provider=provider,
            base=base,
            stored_fp=stored_fp,
            model_upgraded=model_upgraded,
            active_text=active_text,
            report=report,
        )
        report.embeddings_computed = computed
        report.embeddings_reused = max(0, len(active_text) - computed)
        report.new_fingerprint = target_fp

        # ---- Phase C: single atomic transaction (FR-13) ----------------
        finished_at = utc_now_iso()
        report.finished_at = finished_at
        report.status = STATUS_SUCCESS
        report.endpoints_called = dict(zot.request_counts)
        detail_json = report.to_json()

        with store.transaction():
            if first_sync:
                store.init_library(
                    str(identity.user_id), "users", identity.username
                )
            for key in sorted(deleted_colls):
                store.delete_collection(key)
            for key, row in new_colls.items():
                store.upsert_collection(
                    key=key,
                    version=row["version"],
                    name=row["name"],
                    parent_key=row["parent_key"],
                )

            for key in sorted(deleted_items):
                store.delete_item(key)

            for key in sorted(projected):
                item = projected[key]
                store.upsert_item(
                    key=item.key,
                    version=item.version,
                    item_type=item.item_type,
                    title=item.title,
                    abstract=item.abstract,
                    added_date=item.added_date,
                    content_hash=item.content_hash,
                )
                store.replace_membership(item.key, item.collections)
                if not item.abstract:
                    # Abstract removed (or always-empty first sync): no vector
                    # link may remain, otherwise it would leak into the corpus.
                    store.delete_item_link(item.key, target_fp)

            for key, cols in membership_override.items():
                store.replace_membership(key, cols)

            # Out-of-whitelist item bookkeeping (M1).
            for key, (ex_version, ex_type) in excluded_upserts.items():
                store.upsert_excluded_item(
                    key=key,
                    version=ex_version,
                    item_type=ex_type,
                    updated_at=finished_at,
                )
            for key in excluded_removals:
                store.delete_excluded_item(key)
            store.delete_excluded_missing(set(versions))

            if model_upgraded and stored_fp is not None:
                # Old links are dropped inside the same transaction; old blobs
                # become unreferenced and are GC'd in Phase D.
                store.delete_links_for_fingerprint(stored_fp)

            for key, (chash, _text) in active_text.items():
                store.link_item_embedding(
                    item_key=key, fingerprint=target_fp, content_hash=chash
                )

            # Commit-time invariant: every active abstract-bearing item must
            # have a vector link for the fingerprint we are about to publish.
            missing = store.missing_vectors(target_fp)
            if missing:
                raise SyncError(
                    "Refusing to commit sync: "
                    f"{len(missing)} active items lack embedding vectors "
                    f"for fingerprint {target_fp!r}: {sorted(missing)[:10]}"
                )

            if self._pre_commit_callback is not None:
                self._pre_commit_callback(report)

            store.update_library_state(
                library_version=snapshot_v,
                embedding_fingerprint=target_fp,
                last_sync_at=finished_at,
                username=identity.username,
            )
            store.insert_audit(
                started_at=report.started_at,
                finished_at=finished_at,
                status=STATUS_SUCCESS,
                old_version=watermark,
                new_version=snapshot_v,
                detail_json=detail_json,
            )

        # ---- Phase D: GC + audit artifacts (best-effort) ---------------
        # The watermark is already committed here; a post-commit failure must
        # never retroactively produce a "failed" audit contradicting state.
        try:
            self._garbage_collect(store)
            self._write_audit_file(store, report, detail_json)
            logger.success(report.human_summary())
        except Exception:  # pragma: no cover - defensive around log/IO
            logger.opt(exception=True).warning(
                "Post-commit phase failed, but sync is already committed at "
                f"library version {snapshot_v}"
            )
        return report

    # ------------------------------------------------------------------
    # Phase B helpers
    # ------------------------------------------------------------------

    def _resolve_target_fp(
        self,
        store: SyncStore,
        *,
        base: str,
        stored_fp: str | None,
        model_upgraded: bool,
    ) -> str | None:
        """Fingerprint usable before the first embedding call, if discoverable.

        Returns None when the dimension is genuinely unknown (first ever run
        with no committed blobs).
        """
        if stored_fp is not None and not model_upgraded:
            return stored_fp
        known = store.known_fingerprint_dims(base)
        if not known:
            return None
        if stored_fp is not None:
            _, old_dim = BaseReranker.split_fingerprint(stored_fp)
            if old_dim is not None:
                preferred = BaseReranker.resolve_fingerprint(base, old_dim)
                if preferred in known:
                    return preferred
        # Deterministic fallback if several dims somehow coexist.
        return sorted(known)[0]

    def _embed_blobs(
        self,
        store: SyncStore,
        *,
        provider: BaseReranker,
        base: str,
        stored_fp: str | None,
        model_upgraded: bool,
        active_text: dict[str, tuple[str, str]],
        report: AuditReport,
    ) -> tuple[str, int]:
        """Embed all missing hashes in per-batch independent transactions.

        Returns ``(target_fingerprint, number_of_texts_actually_embedded)``.
        """
        # Unique content hashes only — identical text across items is embedded
        # once and linked many times.
        hash_to_text: dict[str, str] = {}
        for chash, text in active_text.values():
            hash_to_text.setdefault(chash, text)

        target_fp = self._resolve_target_fp(
            store, base=base, stored_fp=stored_fp, model_upgraded=model_upgraded
        )

        if not hash_to_text:
            # No vectors at all: publish the bare base; the invariant query
            # trivially holds.
            return target_fp if target_fp is not None else base, 0

        existing: set[str] = (
            store.existing_blobs(target_fp, set(hash_to_text)) if target_fp else set()
        )
        missing_hashes = [h for h in hash_to_text if h not in existing]

        batch_size = max(1, int(provider.embedding_batch_size()))
        computed = 0
        dim: int | None = (
            BaseReranker.split_fingerprint(target_fp)[1] if target_fp else None
        )

        for start in range(0, len(missing_hashes), batch_size):
            batch_hashes = missing_hashes[start : start + batch_size]
            # Recheck per batch: fp may only become known after the first
            # provider response, and earlier batches of THIS run may already
            # have populated blobs.
            if target_fp is not None:
                already = store.existing_blobs(target_fp, set(batch_hashes))
                batch_hashes = [h for h in batch_hashes if h not in already]
                if not batch_hashes:  # pragma: no cover - defensive recheck
                    continue

            batch_texts = [hash_to_text[h] for h in batch_hashes]
            vectors = provider.embed_texts(batch_texts)
            vectors = np.asarray(vectors, dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[0] != len(batch_texts):
                raise SyncProtocolError(
                    f"embed_texts returned shape {getattr(vectors, 'shape', None)} "
                    f"for {len(batch_texts)} texts"
                )

            if dim is None:
                dim = int(vectors.shape[1])
                target_fp = BaseReranker.resolve_fingerprint(base, dim)
                # First-ever fp resolution: blobs for the same hashes may exist
                # under it from a crashed previous run.
                already = store.existing_blobs(target_fp, set(batch_hashes))
                keep = [
                    (h, v) for h, v in zip(batch_hashes, vectors) if h not in already
                ]
            else:
                if int(vectors.shape[1]) != dim:
                    raise SyncProtocolError(
                        f"Inconsistent embedding dimensions: saw {dim}, "
                        f"got {vectors.shape[1]}. The served model appears to "
                        "have changed under the same model name — bump "
                        "reranker.embedding_model_version in the config to "
                        "rebuild the embedding cache with the new model."
                    )
                keep = list(zip(batch_hashes, vectors))

            if keep:
                with store.transaction():
                    for chash, vector in keep:
                        store.insert_blob(
                            fingerprint=target_fp,
                            content_hash=chash,
                            dim=dim,
                            vector=vector,
                        )
                computed += len(keep)

        if target_fp is None:  # pragma: no cover - guarded by shape check above
            # Defensive: missing_hashes was non-empty, so a response resolved
            # the dimension; reaching here means provider returned nothing.
            raise SyncError("Embedding phase ended without a fingerprint")
        return target_fp, computed

    # ------------------------------------------------------------------
    # collection paths / no-op / failure artifacts
    # ------------------------------------------------------------------

    @staticmethod
    def _collection_path(key: str, colls: dict[str, dict]) -> str:
        """Resolve a collection key to a ``Parent/Child`` path.

        Defensive against missing parents and cycles: walks the parent chain
        with the names available; falls back to the collection key when even
        the collection row itself is absent.
        """
        parts: list[str] = []
        seen: set[str] = set()
        current = key
        while current and current not in seen:
            seen.add(current)
            row = colls.get(current)
            if row is None:
                break
            parts.append(row["name"] or current)
            current = row.get("parent_key")
        if not parts:
            return key
        return "/".join(reversed(parts))

    def _finish_no_changes(
        self, store: SyncStore, report: AuditReport
    ) -> AuditReport:
        finished_at = utc_now_iso()
        report.finished_at = finished_at
        report.status = STATUS_NO_CHANGES
        detail_json = report.to_json()
        with store.transaction():
            store.touch_library_state(finished_at)
            store.insert_audit(
                started_at=report.started_at,
                finished_at=finished_at,
                status=STATUS_NO_CHANGES,
                old_version=report.old_version,
                new_version=report.new_version,
                detail_json=detail_json,
            )
        self._write_audit_file(store, report, detail_json)
        logger.info(report.human_summary())
        return report

    def _record_failure(
        self,
        store: SyncStore | None,
        report: AuditReport,
        status: str,
        exc: Exception,
    ) -> None:
        """Best-effort failure audit; never masks the original exception."""
        try:
            report.status = status
            report.error = f"{type(exc).__name__}: {exc}"
            report.finished_at = utc_now_iso()
            detail_json = report.to_json()
            if store is not None:
                with store.transaction():
                    store.insert_audit(
                        started_at=report.started_at,
                        finished_at=report.finished_at,
                        status=status,
                        old_version=report.old_version,
                        new_version=report.new_version,
                        detail_json=detail_json,
                    )
                self._write_audit_file(store, report, detail_json)
            logger.error(report.human_summary())
        except Exception:  # pragma: no cover - audit failure must not mask
            logger.opt(exception=True).error("Failed to persist failure audit")

    def _garbage_collect(self, store: SyncStore) -> None:
        try:
            orphans = store.unreferenced_blobs()
            for fingerprint, chash in orphans:
                store.delete_blob(fingerprint, chash)
            if orphans:
                logger.info(f"Garbage collected {len(orphans)} embedding blobs")
        except Exception:
            logger.opt(exception=True).warning("Embedding blob GC failed")

    def _write_audit_file(
        self, store: SyncStore, report: AuditReport, detail_json: str
    ) -> None:
        try:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            path = store.audit_dir / f"{stamp}-{report.status}.json"
            write_atomic(path, detail_json)
        except Exception:
            logger.opt(exception=True).warning("Failed to write audit JSON file")

    # ------------------------------------------------------------------
    # corpus loading for the executor (FR-16)
    # ------------------------------------------------------------------

    def load_corpus(self) -> tuple[list[CorpusPaper], np.ndarray]:
        store = self.store()
        state = store.get_library_state()
        if state is None or state.get("embedding_fingerprint") is None:
            raise SyncError("No synced corpus; run sync() before load_corpus()")
        fingerprint = state["embedding_fingerprint"]
        rows = store.iter_corpus_rows(fingerprint)
        membership = store.get_all_membership()
        colls = store.get_collections()

        papers: list[CorpusPaper] = []
        vectors: list[np.ndarray] = []
        for row in rows:
            paths = sorted(
                {
                    self._collection_path(col_key, colls)
                    for col_key in membership.get(row["key"], set())
                }
            )
            papers.append(
                CorpusPaper(
                    title=row["title"],
                    abstract=row["abstract"],
                    added_date=self._parse_date(row["added_date"]),
                    paths=paths,
                    key=row["key"],
                )
            )
            vectors.append(blob_to_vector(row["vector"], row["dim"]))

        if vectors:
            matrix = np.stack(vectors).astype(np.float32)
        else:
            matrix = np.empty((0, 0), dtype=np.float32)
        return papers, matrix

    @staticmethod
    def _parse_date(raw: str | None) -> datetime:
        if not raw:
            return datetime.min
        try:
            return datetime.strptime(raw, _DATE_FORMAT)
        except ValueError:
            try:
                return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(
                    tzinfo=None
                )
            except ValueError:
                return datetime.min
