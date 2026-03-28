from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable

import pyarrow.parquet as pq


__version__ = "4.0.0"


class DownloadMode(Enum):
    REUSE_DATASET_IF_EXISTS = "reuse_dataset_if_exists"
    REUSE_CACHE_IF_EXISTS = "reuse_cache_if_exists"
    FORCE_REDOWNLOAD = "force_redownload"


class Split(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


config = SimpleNamespace(HF_DATASETS_TRUST_REMOTE_CODE=False)


def _dataset_root() -> Path:
    return Path(os.environ.get("LMEVAL_LOCAL_DATASETS_ROOT", "/root/datasets"))


def _iter_parquet_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches():
            rows.extend(batch.to_pylist())
    return rows


@dataclass
class Dataset:
    _rows: list[dict[str, Any]]
    split: str | None = None

    def __iter__(self):
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, item):
        if isinstance(item, str):
            return [row[item] for row in self._rows]
        if isinstance(item, slice):
            return Dataset(self._rows[item], split=self.split)
        if isinstance(item, list):
            return Dataset([self._rows[i] for i in item], split=self.split)
        return self._rows[item]

    @property
    def features(self) -> dict[str, Any]:
        if not self._rows:
            return {}
        return {key: None for key in self._rows[0].keys()}

    def map(self, function: Callable[[dict[str, Any]], dict[str, Any]]) -> "Dataset":
        mapped_rows: list[dict[str, Any]] = []
        for row in self._rows:
            updated = function(dict(row))
            if isinstance(updated, dict):
                merged = dict(row)
                merged.update(updated)
                mapped_rows.append(merged)
            else:
                mapped_rows.append(dict(row))
        return Dataset(mapped_rows, split=self.split)

    def filter(self, function: Callable[[dict[str, Any]], bool]) -> "Dataset":
        return Dataset([row for row in self._rows if function(row)], split=self.split)

    @classmethod
    def from_list(cls, rows: list[dict[str, Any]], split: str | Split | None = None):
        return cls(list(rows), split=split.value if isinstance(split, Split) else split)

    @classmethod
    def from_dict(cls, data: dict[str, list[Any]], split: str | Split | None = None):
        if not data:
            return cls([], split=split.value if isinstance(split, Split) else split)
        keys = list(data.keys())
        length = len(data[keys[0]])
        rows = [{key: data[key][idx] for key in keys} for idx in range(length)]
        return cls(rows, split=split.value if isinstance(split, Split) else split)


class DatasetDict(dict):
    pass


def _load_arc(name: str) -> DatasetDict:
    root = _dataset_root() / "ai2_arc" / name
    return DatasetDict(
        {
            "train": Dataset(_iter_parquet_rows(sorted(root.glob("train-*.parquet"))), split="train"),
            "validation": Dataset(
                _iter_parquet_rows(sorted(root.glob("validation-*.parquet"))), split="validation"
            ),
            "test": Dataset(_iter_parquet_rows(sorted(root.glob("test-*.parquet"))), split="test"),
        }
    )


def _load_hellaswag() -> DatasetDict:
    root = _dataset_root() / "hellaswag" / "data"
    return DatasetDict(
        {
            "train": Dataset(_iter_parquet_rows(sorted(root.glob("train-*.parquet"))), split="train"),
            "validation": Dataset(
                _iter_parquet_rows(sorted(root.glob("validation-*.parquet"))), split="validation"
            ),
            "test": Dataset(_iter_parquet_rows(sorted(root.glob("test-*.parquet"))), split="test"),
        }
    )


def _load_ceval(name: str) -> DatasetDict:
    root = _dataset_root() / "ceval" / name
    return DatasetDict(
        {
            "dev": Dataset(_iter_parquet_rows(sorted(root.glob("dev-*.parquet"))), split="dev"),
            "val": Dataset(_iter_parquet_rows(sorted(root.glob("val-*.parquet"))), split="val"),
            "test": Dataset(_iter_parquet_rows(sorted(root.glob("test-*.parquet"))), split="test"),
        }
    )


def _load_json_dataset(data_files, split: str | None):
    if isinstance(data_files, dict):
        files = data_files.get(split or "train") or next(iter(data_files.values()))
    else:
        files = data_files
    if isinstance(files, (str, Path)):
        files = [files]
    rows: list[dict[str, Any]] = []
    for file in files:
        path = Path(str(file))
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            rows.extend(payload)
        else:
            rows.append(payload)
    dataset = Dataset(rows, split=split or "train")
    if split is not None:
        return dataset
    return DatasetDict({"train": dataset})


def load_dataset(
    path: str,
    name: str | None = None,
    split: str | None = None,
    data_files=None,
    **_: Any,
):
    if path == "allenai/ai2_arc":
        dataset = _load_arc(name or "ARC-Challenge")
    elif path == "Rowan/hellaswag":
        dataset = _load_hellaswag()
    elif path == "ceval/ceval-exam":
        if not name:
            raise ValueError("ceval/ceval-exam requires a dataset name")
        dataset = _load_ceval(name)
    elif path == "json":
        return _load_json_dataset(data_files=data_files, split=split)
    else:
        raise NotImplementedError(
            f"Local datasets shim does not support dataset path {path!r}"
        )

    if split is not None:
        return dataset[split]
    return dataset


__all__ = [
    "Dataset",
    "DatasetDict",
    "DownloadMode",
    "Split",
    "config",
    "load_dataset",
]
