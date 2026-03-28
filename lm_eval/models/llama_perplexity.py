from __future__ import annotations

import logging
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm


eval_logger = logging.getLogger(__name__)

_THIRD_PARTY_DIR = Path(__file__).resolve().parents[3]
_DEFAULT_LLAMA_PERPLEXITY_BIN = (
    _THIRD_PARTY_DIR / "llama.cpp" / "build" / "bin" / "llama-perplexity"
)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"cannot interpret {value!r} as bool")


def _as_int(value, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, str) and value.strip().lower() == "auto":
        return default
    return int(value)


def _serialize_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<I", len(encoded)) + encoded


def _serialize_answers(answers: list[str], labels: list[int]) -> bytes:
    payload = bytearray()
    payload.extend(struct.pack("<I", len(answers)))
    for answer in answers:
        payload.extend(_serialize_string(answer))
    for label in labels:
        payload.extend(struct.pack("<i", label))
    return bytes(payload)


def _serialize_task(context: str, continuations: list[str]) -> bytes:
    payload = bytearray()
    payload.extend(_serialize_string(context))
    payload.extend(_serialize_answers(continuations, [1] + [0] * (len(continuations) - 1)))
    payload.extend(_serialize_answers([], []))
    return bytes(payload)


def _build_payload(tasks: list[tuple[str, list[str]]]) -> bytes:
    blobs = [_serialize_task(context, continuations) for context, continuations in tasks]
    header_size = 4 + 4 * len(blobs)
    offsets: list[int] = []
    cursor = header_size
    for blob in blobs:
        offsets.append(cursor)
        cursor += len(blob)

    payload = bytearray()
    payload.extend(struct.pack("<I", len(blobs)))
    for offset in offsets:
        payload.extend(struct.pack("<I", offset))
    for blob in blobs:
        payload.extend(blob)
    return bytes(payload)


@dataclass
class _UniqueRequest:
    req: Instance
    context: str
    continuation: str
    positions: list[int] = field(default_factory=list)


@dataclass
class _GroupedTask:
    context: str
    entries: list[_UniqueRequest] = field(default_factory=list)


@register_model("llama-perplexity")
class LlamaPerplexityLM(LM):
    def __init__(
        self,
        model_path: str | None = None,
        model: str | None = None,
        llama_perplexity_bin: str | None = None,
        threads: int = 8,
        threads_batch: int | None = None,
        ctx_size: int = 4096,
        batch_size: int = 2048,
        ubatch_size: int = 512,
        n_parallel: int = 8,
        gpu_layers: int = 0,
        flash_attn: bool = False,
        no_mmap: bool = False,
        max_tasks_per_call: int = 256,
        **_: object,
    ) -> None:
        super().__init__()
        self.model_path = model_path or model
        if not self.model_path:
            raise ValueError("must pass `model_path` (or `model`) to use llama-perplexity")

        if llama_perplexity_bin:
            self.llama_perplexity_bin = Path(llama_perplexity_bin)
        else:
            discovered = shutil.which("llama-perplexity")
            self.llama_perplexity_bin = (
                Path(discovered) if discovered else _DEFAULT_LLAMA_PERPLEXITY_BIN
            )

        if not self.llama_perplexity_bin.is_file():
            raise FileNotFoundError(
                f"llama-perplexity binary not found at {self.llama_perplexity_bin}"
            )

        self.threads = _as_int(threads, 8)
        self.threads_batch = _as_int(threads_batch, self.threads)
        self.ctx_size = _as_int(ctx_size, 4096)
        self.batch_size = _as_int(batch_size, 2048)
        self.ubatch_size = _as_int(ubatch_size, 512)
        self.n_parallel = _as_int(n_parallel, 8)
        self.gpu_layers = _as_int(gpu_layers, 0)
        self.flash_attn = _as_bool(flash_attn)
        self.no_mmap = _as_bool(no_mmap)
        self.max_tasks_per_call = _as_int(max_tasks_per_call, 256)
        if self.n_parallel <= 0:
            raise ValueError("n_parallel must be > 0")
        if self.max_tasks_per_call <= 0:
            raise ValueError("max_tasks_per_call must be > 0")

    @property
    def tokenizer_name(self) -> str:
        return Path(self.model_path).name

    def get_model_info(self) -> dict[str, str]:
        return {
            "backend": "llama-perplexity",
            "model_path": self.model_path,
            "llama_perplexity_bin": str(self.llama_perplexity_bin),
            "n_parallel": str(self.n_parallel),
        }

    def _group_requests(self, requests: list[Instance]) -> list[_GroupedTask]:
        unique_by_identity: dict[int, _UniqueRequest] = {}
        for position, req in enumerate(requests):
            req_id = id(req)
            if req_id in unique_by_identity:
                unique_by_identity[req_id].positions.append(position)
                continue

            if len(req.args) != 2:
                raise NotImplementedError(
                    "llama-perplexity backend currently supports text-only loglikelihood requests"
                )

            context, continuation = req.args
            if not isinstance(context, str) or not isinstance(continuation, str):
                raise NotImplementedError(
                    "llama-perplexity backend currently supports text-only loglikelihood requests"
                )
            if continuation == "":
                raise ValueError("llama-perplexity backend received an empty continuation")

            unique_by_identity[req_id] = _UniqueRequest(
                req=req,
                context=context,
                continuation=continuation,
                positions=[position],
            )

        grouped: dict[tuple[str | None, int | None, str], _GroupedTask] = {}
        for entry in unique_by_identity.values():
            key = (entry.req.task_name, entry.req.doc_id, entry.context)
            if key not in grouped:
                grouped[key] = _GroupedTask(context=entry.context)
            grouped[key].entries.append(entry)

        ordered_groups = list(grouped.values())
        for group in ordered_groups:
            group.entries.sort(key=lambda item: item.req.idx)
            if len(group.entries) < 2:
                task_name = group.entries[0].req.task_name or "<unknown>"
                doc_id = group.entries[0].req.doc_id
                raise NotImplementedError(
                    "llama-perplexity backend currently supports grouped multiple-choice style "
                    f"loglikelihood requests only; got a singleton request for task={task_name}, doc_id={doc_id}. "
                    "Tasks with one continuation or multiple_input contexts are not supported yet."
                )

        return ordered_groups

    def _run_multiple_choice(self, grouped_tasks: list[_GroupedTask]) -> dict[tuple[int, int], float]:
        prompt_payload = _build_payload(
            [
                (group.context, [entry.continuation for entry in group.entries])
                for group in grouped_tasks
            ]
        )

        with tempfile.NamedTemporaryFile(
            prefix="lm-eval-llama-perplexity-",
            suffix=".bin",
            delete=False,
            dir="/tmp",
        ) as prompt_file:
            prompt_file.write(prompt_payload)
            prompt_path = Path(prompt_file.name)

        with tempfile.NamedTemporaryFile(
            prefix="lm-eval-llama-perplexity-",
            suffix=".tsv",
            delete=False,
            dir="/tmp",
        ) as output_file:
            output_path = Path(output_file.name)

        command = [
            str(self.llama_perplexity_bin),
            "-m",
            self.model_path,
            "-bf",
            str(prompt_path),
            "--multiple-choice",
            "-o",
            str(output_path),
            "-n",
            "0",
            "-t",
            str(self.threads),
            "-tb",
            str(self.threads_batch),
            "-c",
            str(self.ctx_size),
            "-b",
            str(self.batch_size),
            "-ub",
            str(self.ubatch_size),
            "-np",
            str(self.n_parallel),
            "-ngl",
            str(self.gpu_layers),
        ]
        if self.flash_attn:
            command.append("-fa")
        if self.no_mmap:
            command.append("--no-mmap")

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "llama-perplexity failed with exit code "
                    f"{completed.returncode}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
                )

            scores: dict[tuple[int, int], float] = {}
            for raw_line in output_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                task_idx_str, choice_idx_str, sum_logprob_str, _avg_str, _count_str = line.split(
                    "\t"
                )
                scores[(int(task_idx_str), int(choice_idx_str))] = float(sum_logprob_str)

            missing_keys = [
                (task_index, choice_index)
                for task_index, group in enumerate(grouped_tasks)
                for choice_index, _entry in enumerate(group.entries)
                if (task_index, choice_index) not in scores
            ]
            if missing_keys:
                first_missing_task, first_missing_choice = missing_keys[0]
                task = grouped_tasks[first_missing_task]
                task_name = task.entries[0].req.task_name or "<unknown>"
                doc_id = task.entries[0].req.doc_id
                stdout_tail = completed.stdout[-4000:]
                stderr_tail = completed.stderr[-4000:]
                raise RuntimeError(
                    "llama-perplexity returned incomplete multiple-choice scores: "
                    f"parsed {len(scores)} scores for "
                    f"{sum(len(group.entries) for group in grouped_tasks)} expected choices. "
                    f"First missing key: task={first_missing_task}, choice={first_missing_choice}, "
                    f"lm_eval_task={task_name}, doc_id={doc_id}.\n"
                    f"stdout tail:\n{stdout_tail}\n"
                    f"stderr tail:\n{stderr_tail}"
                )

            return scores
        finally:
            prompt_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    def loglikelihood(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[tuple[float, bool]]:
        if not requests:
            return []

        grouped_tasks = self._group_requests(requests)
        scores: dict[tuple[int, int], float] = {}
        chunk_starts = range(0, len(grouped_tasks), self.max_tasks_per_call)
        chunk_iter = tqdm(
            chunk_starts,
            total=(len(grouped_tasks) + self.max_tasks_per_call - 1) // self.max_tasks_per_call,
            desc="llama-perplexity chunks",
            unit="chunk",
            disable=disable_tqdm,
        )
        for chunk_start in chunk_iter:
            chunk = grouped_tasks[chunk_start : chunk_start + self.max_tasks_per_call]
            chunk_scores = self._run_multiple_choice(chunk)
            for (task_index, choice_index), score in chunk_scores.items():
                scores[(chunk_start + task_index, choice_index)] = score

        results: list[tuple[float, bool] | None] = [None] * len(requests)
        for task_index, group in enumerate(grouped_tasks):
            for choice_index, entry in enumerate(group.entries):
                score_key = (task_index, choice_index)
                if score_key not in scores:
                    raise RuntimeError(
                        f"missing llama-perplexity score for task {task_index}, choice {choice_index}"
                    )
                result = (scores[score_key], False)
                for position in entry.positions:
                    results[position] = result

        if any(result is None for result in results):
            raise RuntimeError("failed to populate all loglikelihood results")

        return [(result[0], result[1]) for result in results if result is not None]

    def generate_until(self, requests: list[Instance]) -> list[str]:
        raise NotImplementedError(
            "llama-perplexity backend only supports loglikelihood requests"
        )

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        raise NotImplementedError(
            "llama-perplexity backend only supports loglikelihood requests"
        )
