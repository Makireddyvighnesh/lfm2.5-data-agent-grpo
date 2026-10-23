"""Durable JSON artifacts and append-only rollout records, without retokenization."""

import gzip
import json
import os
import threading
import uuid
from pathlib import Path

_append_lock = threading.Lock()


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json(path, value, *, compressed=False):
    """Publish one complete file atomically; a partial write never replaces an artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("wb") as raw:
            if compressed:
                with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=3, mtime=0) as stream:
                    stream.write(json.dumps(value, allow_nan=False, separators=(",", ":")).encode())
            else:
                raw.write((json.dumps(value, allow_nan=False, indent=2) + "\n").encode())
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(tmp, path)
        sync_directory(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def append_jsonl(path, value):
    """Serialize concurrent workers' small metric rows; retain failed/uncommitted attempts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n"
    with _append_lock:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())


def save_rollout(output, row, task, result=None, trace=None):
    """Archive captures immediately, before an optimizer update can fail."""
    output = Path(output)
    filename = f"rollouts/{row['rollout_id']}.json.gz"
    row["capture_file"] = filename
    sequences = [
        {"input_ids": list(s.input_ids), "completion_mask": list(s.completion_mask),
         "old_log_probs": list(s.old_log_probs), "rollout_id": str(s.rollout_id)}
        for s in row.get("sequences", [])
    ]
    metrics = {k: v for k, v in row.items() if k != "sequences"}
    bundle = {
        "schema_version": 1, "metrics": metrics,
        "task": {k: task[k] for k in ("name", "instruction", "difficulty") if k in task},
        "result": result, "training_trace": trace, "training_sequences": sequences,
    }
    write_json(output / filename, bundle, compressed=True)
    append_jsonl(output / "rollout_metrics.jsonl", {"event": "rollout_finished", **metrics})
    return metrics
