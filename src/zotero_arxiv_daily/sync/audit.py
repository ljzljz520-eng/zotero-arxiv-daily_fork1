"""Per-run sync audit report: counters, key lists and JSON serialization."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

STATUS_SUCCESS = "success"
STATUS_NO_CHANGES = "no_changes"
STATUS_FAILED = "failed"
STATUS_FAILED_PERMISSION = "failed_permission"


@dataclass
class MovedItem:
    key: str
    added_paths: list[str] = field(default_factory=list)
    removed_paths: list[str] = field(default_factory=list)


@dataclass
class AuditReport:
    started_at: str
    library_id: str
    library_type: str = "users"
    username: str | None = None
    old_version: int | None = None
    new_version: int | None = None
    status: str = "running"
    full_sync: bool = False
    # fingerprints / model upgrade
    old_fingerprint: str | None = None
    new_fingerprint: str | None = None
    model_upgraded: bool = False
    # remote request bookkeeping
    pages: dict = field(default_factory=dict)
    endpoints_called: dict = field(default_factory=dict)
    # item changes (key lists)
    added: list[str] = field(default_factory=list)
    content_changed: list[str] = field(default_factory=list)
    metadata_only: list[str] = field(default_factory=list)
    moved: list[MovedItem] = field(default_factory=list)
    tombstoned: list[str] = field(default_factory=list)
    trashed: list[str] = field(default_factory=list)
    reconciled: list[str] = field(default_factory=list)
    deleted_collections: list[str] = field(default_factory=list)
    # embeddings
    embeddings_computed: int = 0
    embeddings_reused: int = 0
    # timing / errors
    finished_at: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["counts"] = {
            "added": len(self.added),
            "content_changed": len(self.content_changed),
            "metadata_only": len(self.metadata_only),
            "moved": len(self.moved),
            "tombstoned": len(self.tombstoned),
            "trashed": len(self.trashed),
            "reconciled": len(self.reconciled),
            "deleted_collections": len(self.deleted_collections),
            "embeddings_computed": self.embeddings_computed,
            "embeddings_reused": self.embeddings_reused,
        }
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)

    def human_summary(self) -> str:
        return (
            f"Zotero sync {self.status}: library {self.library_id} "
            f"v{self.old_version}->v{self.new_version} | "
            f"added={len(self.added)} content_changed={len(self.content_changed)} "
            f"metadata_only={len(self.metadata_only)} moved={len(self.moved)} "
            f"tombstoned={len(self.tombstoned)} trashed={len(self.trashed)} "
            f"reconciled={len(self.reconciled)} "
            f"deleted_collections={len(self.deleted_collections)} | "
            f"embeddings computed={self.embeddings_computed} "
            f"reused={self.embeddings_reused}"
            + (f" | model upgraded {self.old_fingerprint} -> {self.new_fingerprint}"
               if self.model_upgraded else "")
            + (f" | error: {self.error}" if self.error else "")
        )


def write_atomic(path: str | Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (temp file + os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)
