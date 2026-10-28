"""Exact finite checks for the mixed-certificate extension.

Run: python mixed_portfolio_verification.py
All resource and financial arithmetic uses integers on a quarter-unit grid.
The matching oracle uses SciPy's linear assignment routine; a separate full
enumeration checks every welfare value, menu price, and counterfactual.
No company data, saved solutions, or random library state is used.
"""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

P = Path(__file__).resolve().parent
SCALE = 4


class Generator:
    def __init__(self, seed=20261019):
        self.z = seed

    def next(self, n):
        self.z = (1664525 * self.z + 1013904223) % 2**32
        return self.z * n // 2**32


def matching(units, cap, weights, canonical=False):
    """Integer-capacity optional assignment, with zero-value rejection.

    Keep raw capacities for all slack tests. Only the number of slot copies
    in the assignment matrix is truncated to the number of unit applicants.
    """
    ids = list(units)
    cap = np.asarray(cap, dtype=int)
    if np.any(cap < 0):
        return None
    if not ids:
        return 0, {}
    slots = [t for t, c in enumerate(cap) for _ in range(min(int(c), len(ids)))]
    cost = np.full((len(ids), len(slots) + len(ids)), 10**7, dtype=np.int64)
    cost[:, len(slots):] = 0
    for row, i in enumerate(ids):
        for col, t in enumerate(slots):
            if t in weights[i]:
                cost[row, col] = -weights[i][t]
    rows, cols = linear_sum_assignment(cost)
    ans = {i: None for i in ids}
    val = 0
    for row, col in zip(rows, cols):
        if col < len(slots):
            i, t = ids[int(row)], slots[int(col)]
            assert t in weights[i]
            ans[i] = t
            val += weights[i][t]
    if not canonical:
        return int(val), ans
    # Resolve ties by the fixed order rejection, period 0, period 1, ... .
    residual = cap.copy()
    target = int(val)
    selected = {}
    for pos, i in enumerate(ids):
        for t in [None] + sorted(weights[i]):
            if t is not None and residual[t] == 0:
                continue
            trial = residual.copy()
            own = 0 if t is None else weights[i][t]
            if t is not None:
                trial[t] -= 1
            rest = matching(ids[pos + 1:], trial, weights)
            if own + rest[0] == target:
                selected[i] = t
                residual = trial
                target -= own
                break
        else:
            raise AssertionError("canonical optimum not found")
    return int(val), selected


def branch_data(ids, unit_ids, profiles, weights, C):
    units = [i for i in ids if i in unit_ids]
    bundles = [i for i in ids if i not in unit_ids]
    records = []
    for modes in itertools.product(*(range(len(profiles[i])) for i in bundles)):
        load = sum((profiles[i][m][0] for i, m in zip(bundles, modes)), np.zeros(len(C), dtype=int))
        if np.any(load > C):
            continue
        residual = (C - load) // SCALE
        fixed = sum(profiles[i][m][1] for i, m in zip(bundles, modes))
        unit_w, assignment = matching(units, residual, weights, canonical=True)
        records.append({"modes": modes, "bundles": bundles, "load": load,
                        "residual": residual, "fixed": fixed, "unit_w": unit_w,
                        "assignment": assignment, "value": fixed + unit_w})
    assert records
    return records


def mixed(ids, unit_ids, profiles, weights, C):
    records = branch_data(ids, unit_ids, profiles, weights, C)
    best_value = max(r["value"] for r in records)
    best_mode = None
    for r in records:
        if r["value"] != best_value:
            continue
        choices = dict(zip(r["bundles"], r["modes"]))
        choices.update({i: 0 if t is None else t + 1 for i, t in r["assignment"].items()})
        mode = tuple(choices[i] for i in ids)
        if best_mode is None or mode < best_mode:
            best_mode = mode
    return best_value, best_mode, records


def exhaustive(ids, profiles, C):
    best = -1
    chosen = None
    for modes in itertools.product(*(range(len(profiles[i])) for i in ids)):
        load = np.zeros(len(C), dtype=int)
        value = 0
        for i, m in zip(ids, modes):
            a, w = profiles[i][m]
            load += a
            value += w
        if np.all(load <= C) and value > best:
            best, chosen = value, modes
    return int(best), chosen


def displacement(record, weights, t):
    cap, assigned = record["residual"], record["assignment"]
    T = len(cap)
    arcs = []
    occupancy = np.zeros(T, dtype=int)
    for i, old in assigned.items():
        if old is None:
            continue
        occupancy[old] += 1
        arcs.append((old, T, weights[i][old]))
        for new in weights[i]:
            if new != old:
                arcs.append((old, new, weights[i][old] - weights[i][new]))
    arcs.extend((s, T, 0) for s in range(T) if occupancy[s] < cap[s])
    dist = [10**10] * (T + 1)
    dist[t] = 0
    for _ in range(T):
        changed = False
        for u, v, length in arcs:
            if dist[u] < 10**10 and dist[v] > dist[u] + length:
                dist[v] = dist[u] + length
                changed = True
        if not changed:
            break
    assert dist[T] < 10**10 and dist[T] >= 0
    return dist[T]


def instance(g, case):
    T, u, k = 3, case % 5, 1 + (case // 5) % 2
    unit_ids = set(range(u))
    profiles, weights, financial = {}, {}, {}
    for i in range(u + k):
        b, c = 2 + g.next(17), g.next(6)
        financial[i] = (b, c)
        pp = [(np.zeros(T, dtype=int), 0)]
        if i in unit_ids:
            weights[i] = {}
            for t in range(T):
                a = np.zeros(T, dtype=int)
                a[t] = SCALE
                if g.next(4) != 0:
                    w = SCALE * b - c * SCALE * (t + 1)
                    weights[i][t] = w
                    pp.append((a, w))
                else:
                    # An unavailable mode has a load exceeding every capacity.
                    pp.append((np.full(T, 1000, dtype=int), -10**6))
        else:
            for _ in range(2):
                a = np.zeros(T, dtype=int)
                for _ in range(SCALE):
                    a[g.next(T)] += 1
                w = SCALE * b - c * int(np.dot(np.arange(1, T + 1), a))
                pp.append((a, w))
        profiles[i] = pp
    C = np.array([2 + g.next(13) for _ in range(T)], dtype=int)
    return list(range(u + k)), unit_ids, profiles, weights, financial, C


def run():
    g = Generator()
    counts = {"instances": 0, "welfare_comparisons": 0, "canonical_allocation_comparisons": 0,
              "exclusion_comparisons": 0, "menu_price_comparisons": 0,
              "displacement_comparisons": 0, "pruning_comparisons": 0,
              "payment_checks": 0, "deviation_checks": 0, "counterexample_checks": 0}
    max_price = 0
    for case in range(240):
        ids, units, profiles, weights, financial, C = instance(g, case)
        full, chosen, _ = mixed(ids, units, profiles, weights, C)
        brute, brute_mode = exhaustive(ids, profiles, C)
        assert (full, chosen) == (brute, brute_mode)
        counts["instances"] += 1
        counts["welfare_comparisons"] += 1
        counts["canonical_allocation_comparisons"] += 1
        for i in ids:
            other = [j for j in ids if j != i]
            wo, _, branches = mixed(other, units, profiles, weights, C)
            assert wo == exhaustive(other, profiles, C)[0]
            counts["exclusion_comparisons"] += 1
            own = profiles[i][chosen[ids.index(i)]][1]
            pay = wo - full + own
            assert pay >= 0 and own - pay >= 0
            if chosen[ids.index(i)] == 0:
                assert pay == 0
            counts["payment_checks"] += 1
            for mode, (a, _) in enumerate(profiles[i][1:], 1):
                if np.any(a > C):
                    continue
                residual_best = exhaustive(other, profiles, C - a)[0]
                true_price = wo - residual_best
                terms = []
                for branch in branches:
                    if np.any(branch["load"] > C - a):
                        continue
                    newcap = (C - a - branch["load"]) // SCALE
                    uw = matching([j for j in other if j in units], newcap, weights)[0]
                    gap = wo - branch["value"]
                    loss = branch["unit_w"] - uw
                    assert gap >= 0 and loss >= 0
                    terms.append((gap, loss))
                    if i in units:
                        assert loss == displacement(branch, weights, mode - 1)
                        counts["displacement_comparisons"] += 1
                assert min(gap + loss for gap, loss in terms) == true_price
                counts["menu_price_comparisons"] += 1
                # Any feasible branch supplies an upper bound for gap pruning.
                h = terms[0][0] + terms[0][1]
                pruned = [(gap, loss) for gap, loss in terms if gap <= h]
                assert min(gap + loss for gap, loss in pruned) == true_price
                counts["pruning_comparisons"] += 1
                max_price = max(max_price, true_price)
            # Test both private financial coordinates on the first 30 cases.
            # Recompute full allocations and each report-independent exclusion.
            if case < 30:
                truthful_u = full - wo
                b_true, c_true = financial[i]
                for b_report, c_report in itertools.product([0, 5, 20], [0, 2, 7]):
                    changed = dict(profiles)
                    changed_w = {j: dict(w) for j, w in weights.items()}
                    new = [profiles[i][0]]
                    for a, _ in profiles[i][1:]:
                        q4 = int(np.dot(np.arange(1, len(C) + 1), a))
                        new.append((a, SCALE * b_report - c_report * q4))
                    changed[i] = new
                    if i in units:
                        changed_w[i] = {t: new[t + 1][1] for t in weights[i]}
                    wr, mr, _ = mixed(ids, units, changed, changed_w, C)
                    m = mr[ids.index(i)]
                    own_report = new[m][1]
                    true_own = profiles[i][m][1]
                    p = wo - wr + own_report
                    assert true_own - p <= truthful_u
                    counts["deviation_checks"] += 1
    # Sharp frozen-portfolio error, including the price switch at M = epsilon.
    examples = []
    for M in [1, 2, 4, 8, 16, 64, 256, 1024]:
        for eps in [1, 2, 3]:
            for capA in [1, 2, 3]:
                C = np.array([capA * SCALE, SCALE])
                profiles = {0: [(np.zeros(2, dtype=int), 0), (np.array([4, 0]), SCALE * M)],
                            1: [(np.zeros(2, dtype=int), 0), (np.array([2, 2]), SCALE * eps)]}
                weights = {0: {0: SCALE * M}}
                value, _, branches = mixed([0, 1], {0}, profiles, weights, C)
                a = np.array([SCALE, 0])
                residual = exhaustive([0, 1], profiles, C - a)[0]
                price = value - residual
                assert price == (SCALE * min(M, eps) if capA == 2 else
                                 SCALE * max(M, eps) if capA == 1 else 0)
                if capA == 2:
                    on = next(b for b in branches if b["modes"] == (1,))
                    frozen = on["unit_w"] - matching([0], (C - a - on["load"]) // SCALE, weights)[0]
                    assert frozen == SCALE * M
                    examples.append({"M": M, "epsilon": eps, "exact_price": price // SCALE,
                                     "frozen_price": frozen // SCALE})
                counts["counterexample_checks"] += 1
    result = {"protocol": "LCG seed 20261019; 240 cases; T=3; u=case mod 5; k=1+floor(case/5) mod 2; quarter-unit loads and welfare; canonical mode order",
              "counts": counts, "maximum_menu_price": max_price / SCALE,
              "maximum_absolute_difference": 0,
              "all_physical_and_financial_arithmetic_exact": True,
              "counterexample": examples,
              "program_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (P / "mixed_portfolio_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "counterexample"}, indent=2))


if __name__ == "__main__":
    run()
