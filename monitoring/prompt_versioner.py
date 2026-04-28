"""
monitoring/prompt_versioner.py
Prompt version management and CI regression gating.

Maintains prompts/versions.json as the authoritative record of every
prompt version, its SHA-256 hash, which eval report validated it, and
whether CI passed.

Key operations:
  - hash_current()      — compute SHA-256 of the active prompt file
  - verify_integrity()  — confirm the file on disk matches the stored hash
  - register_version()  — add a new version entry after prompt changes
  - mark_ci_result()    — record pass/fail against a version
  - check_for_changes() — fail CI if prompt changed without a new version entry

Usage:
  from monitoring.prompt_versioner import PromptVersioner
  pv = PromptVersioner()
  pv.check_for_changes()   # raises if hash mismatch — use in CI

CLI:
  python cli.py prompt-version status
  python cli.py prompt-version register --version answer_v2 --description "..."
  python cli.py prompt-version verify
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Any

import config

VERSIONS_PATH = config.PROMPTS_DIR / "versions.json"


def _sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


class PromptVersioner:
    """Manages the prompt version manifest at prompts/versions.json."""

    def __init__(self):
        self._path = VERSIONS_PATH
        self._data = self._load()

    def _load(self) -> Dict:
        if self._path.exists():
            return json.loads(self._path.read_text(encoding="utf-8"))
        # Bootstrap empty manifest
        return {
            "current": "answer_v1",
            "versions": {},
            "schema_version": 1,
        }

    def _save(self) -> None:
        self._path.write_text(
            json.dumps(self._data, indent=2),
            encoding="utf-8",
        )

    def _prompt_path(self, version: Optional[str] = None) -> Path:
        v = version or self._data.get("current", "answer_v1")
        entry = self._data["versions"].get(v, {})
        fname = entry.get("file", f"{v}.txt")
        return config.PROMPTS_DIR / fname

    # ── Core operations ────────────────────────────────────────────────

    def hash_current(self) -> str:
        """Return the SHA-256 of the currently active prompt file."""
        p = self._prompt_path()
        if not p.exists():
            raise FileNotFoundError(f"Prompt file not found: {p}")
        return _sha256(p)

    def stored_hash(self, version: Optional[str] = None) -> Optional[str]:
        """Return the stored SHA-256 for a version, or None if not recorded."""
        v = version or self._data.get("current", "answer_v1")
        return self._data["versions"].get(v, {}).get("sha256") or None

    def verify_integrity(self, version: Optional[str] = None) -> bool:
        """
        Confirm the prompt file on disk matches the stored SHA-256.
        Returns True if they match, False if the file was changed without
        a version bump.
        """
        stored = self.stored_hash(version)
        if not stored:
            return True   # no hash stored yet — assume first-time setup
        actual = _sha256(self._prompt_path(version))
        return actual == stored

    def check_for_changes(self) -> None:
        """
        Fail if the current prompt file's SHA-256 doesn't match the manifest.
        Call this in CI to catch unreg istered prompt changes.
        Raises RuntimeError with a clear message.
        """
        v = self._data.get("current", "answer_v1")
        stored = self.stored_hash(v)
        if not stored:
            # First run — register current hash silently
            self._backfill_hash(v)
            return

        actual = _sha256(self._prompt_path(v))
        if actual != stored:
            raise RuntimeError(
                f"\n[PROMPT INTEGRITY FAILURE]\n"
                f"  Version   : {v}\n"
                f"  File      : {self._prompt_path(v)}\n"
                f"  Stored    : {stored[:16]}…\n"
                f"  On-disk   : {actual[:16]}…\n\n"
                f"The prompt file was modified without registering a new version.\n"
                f"Run:  python cli.py prompt-version register "
                f"--version <name> --description '<what changed>'\n"
                f"Then update config.py / generator.py to use the new version name."
            )

    def _backfill_hash(self, version: str) -> None:
        """Set the hash for a version that exists in the manifest but has no hash yet."""
        p = self._prompt_path(version)
        if not p.exists():
            return
        entry = self._data["versions"].setdefault(version, {})
        entry["sha256"] = _sha256(p)
        if not entry.get("file"):
            entry["file"] = f"{version}.txt"
        self._save()

    def register_version(
        self,
        version: str,
        description: str,
        changelog: str = "",
        set_current: bool = True,
    ) -> None:
        """
        Register a new prompt version in the manifest.
        Computes and stores the SHA-256 of the corresponding .txt file.
        Optionally sets it as the current active version.
        """
        p = config.PROMPTS_DIR / f"{version}.txt"
        if not p.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {p}\n"
                f"Create prompts/{version}.txt first."
            )
        self._data["versions"][version] = {
            "file":        f"{version}.txt",
            "sha256":      _sha256(p),
            "created":     datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "description": description,
            "changelog":   changelog,
            "eval_report": None,
            "ci_passed":   None,
        }
        if set_current:
            self._data["current"] = version
        self._save()

    def mark_ci_result(
        self,
        version: Optional[str],
        passed: bool,
        report_path: Optional[str] = None,
    ) -> None:
        """Record the CI pass/fail result against a version."""
        v = version or self._data.get("current", "answer_v1")
        entry = self._data["versions"].setdefault(v, {})
        entry["ci_passed"]   = passed
        entry["eval_report"] = report_path or entry.get("eval_report")
        self._save()

    def status(self) -> Dict[str, Any]:
        """Return a summary dict of current prompt version health."""
        v       = self._data.get("current", "answer_v1")
        entry   = self._data["versions"].get(v, {})
        stored  = entry.get("sha256", "")
        p       = self._prompt_path(v)
        actual  = _sha256(p) if p.exists() else ""
        intact  = (stored == actual) if stored else None

        return {
            "current_version": v,
            "file":            str(p),
            "stored_hash":     stored[:16] + "…" if stored else "not recorded",
            "actual_hash":     actual[:16] + "…" if actual else "file missing",
            "integrity_ok":    intact,
            "description":     entry.get("description", ""),
            "created":         entry.get("created", ""),
            "ci_passed":       entry.get("ci_passed"),
            "eval_report":     entry.get("eval_report"),
            "all_versions":    list(self._data["versions"].keys()),
        }

    def ensure_hashes_populated(self) -> None:
        """
        Backfill SHA-256 for any version entries that have an empty hash.
        Safe to call on first run.
        """
        for v in list(self._data["versions"].keys()):
            if not self._data["versions"][v].get("sha256"):
                self._backfill_hash(v)
