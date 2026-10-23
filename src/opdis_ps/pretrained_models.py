"""Resolve and validate checkpoints from ``paper-pretrained-models-v1``."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

PACKAGE_ID = "paper-pretrained-models-v1"
SCHEMA_NAME = "ecgi.paper_pretrained_models"
FAMILY_WEIGHTS = {"prior": "ema", "transformer": "raw", "palette": "ema"}
METHOD_FAMILIES = {
    "transformer": "transformer",
    "palette": "palette",
    "dps": "prior",
    "ddrm": "prior",
    "opdis-ps": "prior",
    "opdis-ps-marginal": "prior",
    "opdis-ps-marginal-isotropic": "prior",
}
SAFE_MODEL_ID = re.compile(r"^(prior|transformer|palette)-s[0-4]$")


@lru_cache(maxsize=64)
def _sha256_file(path: Path, size_bytes: int, mtime_ns: int) -> str:
    del size_bytes, mtime_ns
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file once per path, size, and modification time in this process."""
    path = path.resolve()
    stat = path.stat()
    return _sha256_file(path, stat.st_size, stat.st_mtime_ns)


def normalize_replicate(replicate: int | str) -> str:
    value = f"s{replicate}" if isinstance(replicate, int) else str(replicate)
    if not re.fullmatch(r"s[0-4]", value):
        raise ValueError("replicate must be one of 0..4 or s0..s4")
    return value


@dataclass(frozen=True)
class ResolvedModel:
    model_id: str
    family: str
    replicate: str
    weights_kind: str
    path: Path
    size_bytes: int
    sha256: str
    package_manifest_sha256: str


class PretrainedModels:
    """Read the package manifest and verify a requested inference checkpoint."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing pretrained-model manifest: {manifest_path}")
        self.manifest_sha256 = sha256_file(manifest_path)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_name") != SCHEMA_NAME:
            raise ValueError(f"unexpected pretrained-model schema in {manifest_path}")
        if self.manifest.get("schema_version") != 1:
            raise ValueError(f"unsupported pretrained-model schema version in {manifest_path}")
        if self.manifest.get("package_id") != PACKAGE_ID:
            raise ValueError(f"unexpected pretrained-model package ID in {manifest_path}")
        records = self.manifest.get("models")
        if not isinstance(records, list):
            raise ValueError("pretrained-model manifest has no model list")
        self._records: dict[str, Mapping[str, Any]] = {}
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError("pretrained-model manifest contains an invalid model record")
            model_id = str(record.get("model_id", ""))
            if not SAFE_MODEL_ID.fullmatch(model_id) or model_id in self._records:
                raise ValueError(f"invalid or duplicate model ID {model_id!r}")
            self._records[model_id] = record

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(self._records)

    def resolve(self, family: str, replicate: int | str = 0) -> ResolvedModel:
        if family not in FAMILY_WEIGHTS:
            raise ValueError(f"unknown model family {family!r}")
        replicate_id = normalize_replicate(replicate)
        model_id = f"{family}-{replicate_id}"
        try:
            record = self._records[model_id]
        except KeyError as error:
            raise ValueError(f"package does not contain {model_id}") from error
        if record.get("family") != family or record.get("replicate") != replicate_id:
            raise ValueError(f"{model_id}: inconsistent family or replicate metadata")
        expected_weights = FAMILY_WEIGHTS[family]
        if record.get("weights_kind") != expected_weights:
            raise ValueError(f"{model_id}: unexpected weight selection")
        file_record = record.get("file")
        if not isinstance(file_record, Mapping):
            raise ValueError(f"{model_id}: missing file metadata")
        relative = str(file_record.get("path", ""))
        if relative != f"{family}/{model_id}.pt":
            raise ValueError(f"{model_id}: unexpected checkpoint path {relative!r}")
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root) or not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"{model_id}: missing or unsafe checkpoint path {path}")
        size_bytes = int(file_record.get("size_bytes", -1))
        expected_sha256 = str(file_record.get("sha256", ""))
        if path.stat().st_size != size_bytes:
            raise ValueError(f"{model_id}: checkpoint size differs from manifest")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"{model_id}: checkpoint SHA-256 differs from manifest")
        return ResolvedModel(
            model_id=model_id,
            family=family,
            replicate=replicate_id,
            weights_kind=expected_weights,
            path=path,
            size_bytes=size_bytes,
            sha256=actual_sha256,
            package_manifest_sha256=self.manifest_sha256,
        )

    def for_method(self, method: str, replicate: int | str = 0) -> ResolvedModel | None:
        family = METHOD_FAMILIES.get(method)
        return None if family is None else self.resolve(family, replicate)


def resolve_method_model(
    method: str,
    models_root: str | Path | None,
    replicate: int | str = 0,
) -> ResolvedModel | None:
    family = METHOD_FAMILIES.get(method)
    if family is None:
        return None
    if models_root is None:
        raise ValueError(
            f"{method} requires --models pointing to an extracted {PACKAGE_ID} package"
        )
    return PretrainedModels(models_root).resolve(family, replicate)
