"""Input-file size per task (train + test), from the HF bucket listing; written to analysis/task_file_sizes.json."""
import json, tomllib, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from huggingface_hub import list_bucket_tree

SMOL = Path(__file__).resolve().parents[2] / "SmolDataEnvs"
OUT = Path(__file__).resolve().parent / "task_file_sizes.json"


def prefix_size(bucket, prefix):
    files = [it for it in list_bucket_tree(bucket, prefix=prefix.rstrip("/") + "/", recursive=True)
             if getattr(it, "type", None) == "file"]
    return {"bytes": sum(int(getattr(f, "size", 0) or 0) for f in files), "files": len(files)}


def main():
    tasks = []
    for split in ("train", "test"):
        for d in sorted((SMOL / "prepared/datasets" / split / "tasks").iterdir()):
            env = tomllib.loads((d / "task.toml").read_text())["environment"]["env"]
            meta = tomllib.loads((d / "task.toml").read_text())["metadata"]
            tasks.append({"split": split, "name": d.name, "difficulty": meta["difficulty_tier"],
                          "bucket": env["HF_BUCKET"], "prefix": env["BUCKET_PREFIX"]})
    prefixes = sorted({(t["bucket"], t["prefix"]) for t in tasks})
    with ThreadPoolExecutor(16) as pool:
        sizes = dict(zip(prefixes, pool.map(lambda bp: prefix_size(*bp), prefixes)))
    for t in tasks:
        t.update(sizes[(t["bucket"], t["prefix"])])
    OUT.write_text(json.dumps(tasks, indent=1))
    print(f"{len(tasks)} tasks, {len(prefixes)} distinct datasets -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
