from collections import Counter

from pirl.curriculum import PromptSampler


def test_regret_values():
    s = PromptSampler(range(10))
    s.update(0, 0, 4, 1)
    s.update(1, 4, 4, 1)
    s.update(2, 1, 4, 1)
    s.update(3, 2, 4, 1)
    assert s.regret(0) == 0.0  # never solved: no signal yet
    assert s.regret(1) == 0.0  # mastered
    assert s.regret(2) == 0.75  # barely solvable: most to learn
    assert s.regret(3) == 0.5


def test_dead_prompts_get_more_rollouts_and_are_never_dropped():
    s = PromptSampler(range(20), base_n=4, hard_n=8, new_frac=0.0, seed=1)
    for i in range(20):
        s.update(i, 0 if i < 10 else 2, 4, 1)
    assert s.n_rollouts(0) == 8 and s.n_rollouts(15) == 4 and s.n_rollouts(99) == 4
    picks = Counter(i for _ in range(2000) for i in s.next_batch(4, 2))
    assert all(picks[i] > 0 for i in range(10))  # floor keeps dead prompts in play
    assert sum(picks[i] for i in range(10, 20)) > 3 * sum(picks[i] for i in range(10))  # frontier preferred


def test_staleness_raises_priority_of_long_unvisited_prompts():
    s = PromptSampler(range(2))
    s.update(0, 0, 4, 1)
    s.update(1, 0, 4, 99)
    assert s.weight(0, 100) > s.weight(1, 100)


def test_batch_mixes_new_and_seen_without_duplicates():
    s = PromptSampler(range(100), new_frac=0.5, seed=0)
    first = s.next_batch(8, 1)
    assert first == list(range(8))  # nothing seen yet: fixed order
    for i in first:
        s.update(i, 1, 4, 1)
    second = s.next_batch(8, 2)
    assert len(set(second)) == 8
    assert sum(i >= 8 for i in second) == 4  # half new prompts, in order
    assert [i for i in second if i >= 8] == [8, 9, 10, 11]


def test_rebuild_from_rollout_rows_matches_updates():
    rows = [{"task": "a", "step": 1, "reward": 1.1, "correct": 1.0}, {"task": "a", "step": 1, "reward": 0.0, "correct": 0.0},
            {"task": "a", "step": 1, "reward": None, "correct": None}, {"task": "b", "step": 2, "reward": 0.0, "correct": 0.0},
            {"task": "a", "step": 3, "reward": 0.0, "correct": 0.0}]
    s = PromptSampler(range(2))
    s.rebuild(rows, {"a": 0, "b": 1})
    assert s.stats[0] == {"s": 0, "n": 1, "step": 3}  # last visit wins
    assert s.stats[1] == {"s": 0, "n": 1, "step": 2}
    assert s.summary() == {"prompts_seen": 2, "dead": 2, "mastered": 0, "frontier": 0}
