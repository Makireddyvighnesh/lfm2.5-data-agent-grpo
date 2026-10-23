"""Show where turn k+1's prompt stops extending turn k's prompt + completion (prefix-chain breaks)."""
import sys, json, httpx
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pirl.common import DATA, MODEL, tokenizer_for
from harbor_env import HarborEnv
from harbor_env.harness import HarborSession
from smoldataenv_harbor.tasks import load_tasks

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 3
tasks = load_tasks(DATA, "train")
split = httpx.get("http://127.0.0.1:8200/smoldataenv/splits").json()["train"]
s = HarborSession(env=HarborEnv("http://127.0.0.1:8200", message_timeout_s=1800), owns_env=True, split=split,
    task_index=idx, instruction=tasks[idx]["instruction"], harness="pi", sandbox="docker",
    llm_url="http://127.0.0.1:8000", model=MODEL, reward_key="correctness,reward",
    sampling={"temperature": 1.0, "top_p": 1.0, "top_k": -1}, agent_step_limit=17, agent_timeout_sec=600)
tok = tokenizer_for()
try:
    s.wait_for_completion(timeout_s=1500)
    turns = [t for t in s.result.turns if not t.discarded]
    for k in range(len(turns) - 1):
        a = list(turns[k].prompt_token_ids) + list(turns[k].completion_token_ids)
        b = list(turns[k + 1].prompt_token_ids)
        n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        print(f"turn {k}->{k+1}: prev={len(a)} next={len(b)} common={n}")
        if n < len(a):
            print("   PREV :", repr(tok.decode(a[max(0, n-20):n+40])))
            print("   NEXT :", repr(tok.decode(b[max(0, n-20):n+40])))
finally:
    s.close()
