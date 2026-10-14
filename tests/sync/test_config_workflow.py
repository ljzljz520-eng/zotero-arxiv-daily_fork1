"""TR-6: config defaults, GitHub Actions cache wiring and .gitignore entry."""

from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = str(_REPO_ROOT / "config")


def _compose_state_dir(monkeypatch, env_value=None):
    GlobalHydra.instance().clear()
    if env_value is None:
        monkeypatch.delenv("ZOTERO_STATE_DIR", raising=False)
    else:
        monkeypatch.setenv("ZOTERO_STATE_DIR", env_value)
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        cfg = compose(config_name="default")
    return cfg


def test_state_dir_defaults_to_repo_relative_dir(monkeypatch):
    cfg = _compose_state_dir(monkeypatch)
    assert cfg.zotero.state_dir == ".zotero_sync"
    assert cfg.reranker.embedding_model_version is None


def test_state_dir_env_override(monkeypatch):
    cfg = _compose_state_dir(
        monkeypatch, env_value="/tmp/custom-zotero-state"
    )
    assert cfg.zotero.state_dir == "/tmp/custom-zotero-state"


def test_main_workflow_caches_state_dir_with_matching_env():
    workflow = yaml.safe_load(
        (_REPO_ROOT / ".github/workflows/main.yml").read_text()
    )
    job = workflow["jobs"]["calculate-and-send"]
    assert job["env"]["ZOTERO_STATE_DIR"].endswith("/.zotero_sync")

    cache_steps = [
        step for step in job["steps"]
        if str(step.get("uses", "")).startswith("actions/cache@")
    ]
    assert len(cache_steps) == 1
    cache = cache_steps[0]
    assert cache["with"]["path"] == ".zotero_sync"
    key = cache["with"]["key"]
    assert "zotero-sync-" in key
    # The key must rotate per run: an exact static hit would make
    # actions/cache skip the post-step save, freezing the watermark.
    assert "${{ github.run_id }}" in key
    restore_keys = cache["with"]["restore-keys"]
    if isinstance(restore_keys, str):
        restore_keys = restore_keys.split()
    # newest snapshot for the same account first, then OS-wide fallback
    assert any("${{ secrets.ZOTERO_ID }}" in k for k in restore_keys)
    assert any(k.rstrip("-").endswith("${{ runner.os }}") for k in restore_keys)

    run_step = next(step for step in job["steps"] if step.get("name") == "Run script")
    # The state dir variable is declared at job level and visible to the run.
    assert job["env"]["ZOTERO_STATE_DIR"]


def test_gitignore_ignores_state_dir():
    text = (_REPO_ROOT / ".gitignore").read_text()
    assert ".zotero_sync/" in text
