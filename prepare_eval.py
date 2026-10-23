"""Materialize the 144-task Data Agent eval split the same way SmolDataEnvs' prepare.py does train/test.

Output: prepared/datasets/eval/tasks/<task>/ + prepared/eval_manifest.json. Same per-task edits as the
tutorial (4 GiB memory, 1 CPU, TAR_OPTIONS) and the same pinned grader, so scores are comparable with
the SmolDataEnvs test baselines. Also reports overlap with the SmolDataEnvs train/test tasks.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import tomli_w
import tomllib
from huggingface_hub import snapshot_download

HERE = Path(__file__).resolve().parent
SMOL = HERE.parent / "SmolDataEnvs"
REPO = "FineEnvs/data-agent-harbor-eval"
REVISION = "4987835c9fa72da209290a3dad908284a4e9ab86"
GRADER_SHA256 = "16f906e9248fa8b7e1141939e425e88e59d52d65e98e360bbacef3720a43629f"  # tutorial prepare.py
# The eval split on the Hub still ships the pre-fix grader: identical grade() ladder, but --json emits
# "tool_efficiency": null, Harbor rejects the whole reward dict, and every CORRECT answer fails to grade.
# Replace exactly that version with the pinned corrected grader the SmolDataEnvs test split uses.
HUB_EVAL_GRADER_SHA256 = "bbcbf4e52964404159a4abd3f2eb8c19c449f7dfbc4f9aab62e4fd7ac22ac84a"
FIXED_GRADER = SMOL / "prepared/datasets/test/tasks/0000_324_324293_qa_2/tests/grader.py"
TEST_SH_SHA256 = {"53ca029279599011577756ed55634853abb45e081f0948d8c0a16a4e0e26e410",  # reward.txt
                  "d881dee179a01e7425ab4cfa89a172dd67b4259e534730fd139a3c12a7d6bddf"}  # reward.json
FILES = ("instruction.md", "task.toml", "environment/Dockerfile", "environment/pull_bucket.py",
         "tests/grader.py", "tests/test.sh")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=HERE / "prepared")
    a = p.parse_args()

    source = Path(snapshot_download(REPO, repo_type="dataset", revision=REVISION,
                                    allow_patterns=["tasks/*/*", "tasks/*/*/*"], max_workers=16))
    if sha256(FIXED_GRADER) != GRADER_SHA256:
        raise ValueError(f"Pinned grader changed: {FIXED_GRADER}")
    dest = a.output / "datasets" / "eval" / "tasks"
    tasks = []
    for folder in sorted((source / "tasks").iterdir()):
        if sha256(folder / "tests/grader.py") not in (GRADER_SHA256, HUB_EVAL_GRADER_SHA256):
            raise ValueError(f"Unknown grader version: {folder.name}")
        if sha256(folder / "tests/test.sh") not in TEST_SH_SHA256:
            raise ValueError(f"Unknown test.sh version: {folder.name}")
        target = dest / folder.name
        for rel in FILES:
            (target / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(FIXED_GRADER if rel == "tests/grader.py" else folder / rel, target / rel)
        spec = tomllib.loads((target / "task.toml").read_text())
        spec["environment"]["memory_mb"] = 4096
        spec["environment"]["cpus"] = 1
        spec["environment"].setdefault("env", {})["TAR_OPTIONS"] = "--no-same-owner"
        (target / "task.toml").write_text(tomli_w.dumps(spec))
        meta = spec["metadata"]
        tasks.append({
            "name": folder.name,
            "source": meta["source_row_id"],
            "notebook": meta["source_row_id"].split("_qa_")[0],
            "difficulty": meta["difficulty_tier"],
            "instruction_sha256": sha256(target / "instruction.md"),
        })
    if len(tasks) != 144 or len({t["instruction_sha256"] for t in tasks}) != len(tasks):
        raise ValueError(f"Expected 144 tasks with unique instructions, got {len(tasks)}")

    overlap = {}
    for split in ("train", "test"):
        other = json.loads((SMOL / "prepared" / f"{split}_manifest.json").read_text())["tasks"]
        overlap[split] = {
            "tasks": len({t["source"] for t in tasks} & {t["source"] for t in other}),
            "notebooks": len({t["notebook"] for t in tasks} & {t.get("notebook") for t in other}),
        }
    manifest = {"source_dataset": REPO, "source_revision": REVISION, "grader_sha256": GRADER_SHA256,
                "task_count": len(tasks), "overlap_with_smoldataenvs": overlap, "tasks": tasks}
    (a.output / "eval_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    counts = {d: sum(t["difficulty"] == d for t in tasks) for d in ("easy", "medium", "hard")}
    print(json.dumps({"tasks": len(tasks), "difficulty": counts, "overlap": overlap}))


if __name__ == "__main__":
    main()
