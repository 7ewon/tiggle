"""정규화된 문장 -> TaskSpec 캐시. 같은 명령은 LLM을 다시 부르지 않고 항상 같은 결과를 낸다.

파일은 사람이 읽고 고칠 수 있는 JSON이다. LLM이 틀리게 해석한 항목은 spec을 직접 고치면
그대로 '정답'으로 쓰인다 (source를 "manual"로 바꿔 두면 구분하기 쉽다).
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import FleetConfig
from .schema import SCHEMA_VERSION, TaskSpec


class SpecCache:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self._entries: dict[str, dict] = {}
        if self.path and self.path.exists():
            self._entries = json.loads(self.path.read_text(encoding="utf-8"))

    @staticmethod
    def key(norm: str, fleet: FleetConfig) -> str:
        return f"{SCHEMA_VERSION}|{fleet.signature}|{norm}"

    def get(self, norm: str, fleet: FleetConfig) -> Optional[TaskSpec]:
        entry = self._entries.get(self.key(norm, fleet))
        return TaskSpec.model_validate(entry["spec"]) if entry else None

    def put(self, norm: str, fleet: FleetConfig, spec: TaskSpec, *, utterance: str, source: str) -> None:
        self._entries[self.key(norm, fleet)] = {
            "utterance": utterance,
            "source": source,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "spec": spec.model_dump(mode="json"),
        }
        self._save()

    def _save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(self._entries, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp, self.path)
