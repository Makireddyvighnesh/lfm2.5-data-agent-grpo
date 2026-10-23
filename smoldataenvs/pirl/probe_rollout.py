"""One Harbor rollout with full diagnostics (M3): status, reward, ATIF match, tokens, phase timings."""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pirl.common import DATA, MODEL, TEMPERATURE, TOP_K, TOP_P


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--harness", default="pi")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--server", default="http://127.0.0.1:8200")
    p.add_argument("--vllm", default="http://127.0.0.1:8000")
    a = p.parse_args()

    from harbor_env import HarborEnv
    from harbor_env.harness import HarborSession
    from smoldataenv_harbor.tasks import load_tasks

    tasks = load_tasks(DATA, a.split)
    split = httpx.get(a.server + "/smoldataenv/splits").json()[a.split]
    t0 = time.time()
    s = HarborSession(
        env=HarborEnv(a.server, message_timeout_s=1800), owns_env=True, split=split,
        task_index=a.index, instruction=tasks[a.index]["instruction"], harness=a.harness,
        sandbox="docker", llm_url=a.vllm, model=a.model, reward_key="correctness,reward",
        sampling={"temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K},
        agent_step_limit=17, agent_timeout_sec=600,
    )
    try:
        s.wait_for_completion(timeout_s=1500)
        r = s.result
        out = {
            "elapsed_s": round(time.time() - t0, 1), "ok": r.ok, "reward": r.reward,
            "atif": r.atif, "n_turns": r.n_turns, "n_roots": r.n_roots,
            "n_trainable_tokens": r.n_trainable_tokens, "capture_level": r.capture_level,
            "phase_timings": r.phase_timings, "findings": r.findings, "error": r.error,
            "trial": r.trial_name,
        }
        try:
            tr = s.fetch_training_trace()
            out["turns"] = [(len(t.prompt_token_ids), len(t.completion_token_ids)) for t in tr.turns]
        except Exception as e:  # noqa: BLE001
            out["trace_error"] = str(e)
        print(json.dumps(out, indent=1, default=str))
    finally:
        s.close()


if __name__ == "__main__":
    main()
