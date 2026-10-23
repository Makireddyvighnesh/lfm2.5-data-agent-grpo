"""Download every task's input files once into a host cache, laid out exactly like pull_bucket.py stages them.

serve_harbor.py bind-mounts <cache>/<BUCKET_PREFIX> read-only at /home/user/input, so the in-sandbox data pull
(an HF-bucket listing + download per rollout, 64 at once) finds the files and skips. Idempotent: complete
prefixes are skipped; a prefix is marked complete only after all its files arrive.
usage: python prefetch_data.py [--cache /home/ubuntu/STT/LLMPerfLab/MiMo_rl/data_cache]
"""
import argparse
import json
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import download_bucket_files, list_bucket_tree

DA = Path(__file__).resolve().parent
SMOL = DA.parent / "SmolDataEnvs"
TASK_DIRS = [SMOL / "prepared/datasets/train/tasks", SMOL / "prepared/datasets/test/tasks",
             DA / "prepared/datasets/eval/tasks"]
DEFAULT_CACHE = DA.parent / "data_cache"


def prefixes():
    out = set()
    for d in TASK_DIRS:
        for t in d.iterdir():
            env = tomllib.loads((t / "task.toml").read_text())["environment"]["env"]
            out.add((env["HF_BUCKET"], env["BUCKET_PREFIX"].rstrip("/")))
    return sorted(out)


def fetch(cache, bucket, prefix):
    dest = cache / prefix
    if (dest / ".complete").exists():
        return prefix, "cached", 0
    dest.mkdir(parents=True, exist_ok=True)
    targets = [(it.path, str(dest / Path(it.path).name))
               for it in list_bucket_tree(bucket, prefix=prefix + "/", recursive=True)
               if getattr(it, "type", None) == "file"]
    if not targets:
        return prefix, "EMPTY", 0
    download_bucket_files(bucket, files=targets)
    size = sum(Path(t).stat().st_size for _, t in targets)
    (dest / ".complete").write_text(json.dumps({"files": len(targets), "bytes": size}))
    return prefix, "downloaded", size


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    a = p.parse_args()
    a.cache.mkdir(parents=True, exist_ok=True)
    jobs = prefixes()
    print(f"{len(jobs)} distinct datasets -> {a.cache}", flush=True)
    done = total = 0
    with ThreadPoolExecutor(8) as pool:
        for prefix, status, size in pool.map(lambda j: fetch(a.cache, *j), jobs):
            done += 1
            total += size
            if status != "cached":
                print(f"[{done}/{len(jobs)}] {status} {prefix} ({size / 1e6:.1f} MB)", flush=True)
    print(f"finished: {done} datasets, {total / 1e9:.2f} GB downloaded this run", flush=True)


if __name__ == "__main__":
    main()
