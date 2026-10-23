"""Do test250 failures belong to problem categories p7 saw (and solved) during training?"""
import json, re, math, tomllib, collections, sys
from pathlib import Path
DA = Path(__file__).resolve().parents[1]
TRAIN = DA / "prepared_v32/datasets/train/tasks"
TEST = DA.parent / "SmolDataEnvs/prepared/datasets/test/tasks"

def meta(d):
    t = tomllib.loads((d / "task.toml").read_text())
    q = (d / "instruction.md").read_text()
    return dict(q=t["task"]["description"] or q, mode=t["metadata"].get("reward_mode_initial"),
                diff=t["metadata"].get("difficulty"), ds=t["metadata"].get("kaggle_dataset_name"))

QTYPES = [  # first match wins
    ("ml_metric", r"\b(accuracy|r2|r\^2|r-squared|rmse|mse|mae|f1|auc|roc|precision|recall|score of the model|model|predict|regression|classifier|cluster)"),
    ("correlation", r"correlat"),
    ("percent_ratio", r"(percent|%|proportion|ratio|rate\b|fraction|share)"),
    ("count", r"(how many|number of|count)"),
    ("aggregate", r"\b(mean|average|median|sum|total|std|standard deviation|variance|maximum|minimum|max|min|highest|lowest|largest|smallest)\b"),
    ("which_label", r"^(which|what (is|are|was|were) the (name|most|top|category|type))|\bwhich\b"),
    ("shape_schema", r"(shape|columns|rows|dtype|missing|null|unique)"),
]
def qtype(q):
    ql = q.lower()
    for name, pat in QTYPES:
        if re.search(pat, ql): return name
    return "other"

train = {d.name: meta(d) for d in TRAIN.iterdir() if d.is_dir()}
test = {d.name: meta(d) for d in TEST.iterdir() if d.is_dir()}

# p7 training outcomes per task (all visits)
tr = collections.defaultdict(list)
for line in open(DA / "runs/p7_think_bash/rollouts.jsonl"):
    r = json.loads(line)
    if r.get("ok") and r.get("model_calls", 0) > 0: tr[r["task"]].append(r["correct"])
tr_rate = {k: sum(v) / len(v) for k, v in tr.items()}
# p7 test250 outcomes
te = {}
for line in open(DA / "runs/test250_p7u350_r1/rollouts.jsonl"):
    r = json.loads(line); te[r["task"]] = r["correctness"]
print(f"train tasks with p7 rollouts: {len(tr_rate)}  ({sum(len(v) for v in tr.values())} rollouts); test250 graded: {len(te)}, failed: {sum(1 for v in te.values() if v < 1)}")

def table(key, label):
    print(f"\n== {label} ==")
    print(f"{'category':22s} {'#train':>6s} {'train solve%':>12s} {'#test':>5s} {'test pass%':>10s} {'#test fail':>10s}")
    cats = collections.defaultdict(lambda: [[], []])
    for k, m in train.items():
        if k in tr_rate: cats[key(m)][0].append(tr_rate[k])
    for k, m in test.items():
        if k in te: cats[key(m)][1].append(te[k])
    for c, (a, b) in sorted(cats.items(), key=lambda x: -len(x[1][1])):
        print(f"{str(c):22s} {len(a):6d} {100*sum(a)/max(len(a),1):11.0f}% {len(b):5d} {100*sum(b)/max(len(b),1):9.0f}% {sum(1 for x in b if x<1):10d}")

table(lambda m: m["mode"], "answer type (reward_mode_initial)")
table(lambda m: qtype(m["q"]), "question type (keyword rules)")
table(lambda m: (m["diff"], m["mode"]), "difficulty x answer type")

# same Kaggle dataset
by_ds = collections.defaultdict(list)
for k, m in train.items():
    if k in tr_rate: by_ds[m["ds"]].append(tr_rate[k])
print("\n== same Kaggle dataset in train ==")
for grp in ("pass", "fail"):
    ks = [k for k in te if (te[k] >= 1) == (grp == "pass")]
    hit = [k for k in ks if test[k]["ds"] in by_ds]
    rates = [sum(by_ds[test[k]["ds"]]) / len(by_ds[test[k]["ds"]]) for k in hit]
    print(f"test {grp}: {len(ks)} tasks, {len(hit)} share a Kaggle dataset with train; mean train solve% on those train tasks: {100*sum(rates)/max(len(rates),1):.0f}%")

# nearest similar training questions (TF-IDF cosine on question text)
tok = lambda s: re.findall(r"[a-z]+", s.lower())
docs = {k: collections.Counter(tok(m["q"])) for k, m in train.items() if k in tr_rate}
df = collections.Counter(w for c in docs.values() for w in c)
N = len(docs)
def vec(c): 
    v = {w: n * math.log((N + 1) / (df.get(w, 0) + 1)) for w, n in c.items()}
    s = math.sqrt(sum(x * x for x in v.values())) or 1; return {w: x / s for w, x in v.items()}
tv = {k: vec(c) for k, c in docs.items()}
def topk(q, k=5):
    v = vec(collections.Counter(tok(q)))
    sims = sorted(((sum(v[w] * t.get(w, 0) for w in v), kk) for kk, t in tv.items()), reverse=True)[:k]
    return sims
print("\n== 5 most similar training questions (TF-IDF) ==")
out = {}
for grp in ("pass", "fail"):
    ks = [k for k in te if (te[k] >= 1) == (grp == "pass")]
    sim, solve, never = [], [], 0
    for k in ks:
        nn = topk(test[k]["q"]); out[k] = [(round(s, 3), kk, round(tr_rate[kk], 2)) for s, kk in nn]
        sim.append(sum(s for s, _ in nn) / 5); r = [tr_rate[kk] for _, kk in nn]; solve.append(sum(r) / 5)
        never += all(x == 0 for x in r)
    print(f"test {grp}: mean similarity {sum(sim)/len(sim):.2f}; p7 train solve% on those neighbours {100*sum(solve)/len(solve):.0f}%; tasks whose 5 neighbours were never solved: {never}")
json.dump({"test_neighbours": out}, open(DA / "analysis/test250_train_neighbours.json", "w"), indent=1)
