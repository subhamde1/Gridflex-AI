"""Independent exact audit of the fractional-capacity protection bound.

This program imports no clearing, matching, or payment implementation from
the other replication programs. Its small optional assignments and public
bundled portfolios are completely enumerated. All inequalities use Fraction.
"""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
from collections import Counter
import json
import random

P = Path(__file__).resolve().parent
ZERO = F(0)


def dot(a, b):
    return sum((x*y for x, y in zip(a, b)), ZERO)


def audit_case(unit_edges, bundle_menus, C, Cp, a):
    T = len(C)
    unit_usage = {}
    for z in product(*[[-1]+list(e) for e in unit_edges]):
        usage = tuple(z.count(t) for t in range(T))
        value = sum((unit_edges[i][t] for i, t in enumerate(z) if t >= 0), ZERO)
        unit_usage[usage] = max(unit_usage.get(usage, value), value)

    def V(r):
        assert all(int(x) == x and x >= 0 for x in r)
        return max(w for use, w in unit_usage.items() if all(x <= y for x, y in zip(use, r)))

    portfolios = []
    for z in product(*[range(len(ms)) for ms in bundle_menus]):
        load = tuple(sum((bundle_menus[i][m][0][t] for i, m in enumerate(z)), ZERO) for t in range(T))
        value = sum((bundle_menus[i][m][1] for i, m in enumerate(z)), ZERO)
        portfolios.append((z, load, value))

    def conditional(port, h):
        _, load, value = port
        if any(x > y for x, y in zip(load, h)):
            return None
        r = tuple((x-y).__floor__() for x, y in zip(h, load))
        return value+V(r), r

    states = [tuple(x-y for x, y in zip(C, a)), Cp, C,
              tuple(x-y for x, y in zip(Cp, a))]
    W = [max(q[0] for p in portfolios if (q := conditional(p, h)) is not None)
         for h in states]
    change = W[1]-W[3]-W[2]+W[0]
    rows = []
    defects = 0
    for port in portfolios:
        out = [conditional(port, h) for h in states]
        if out[0] is None:
            continue
        assert all(x is not None for x in out)
        f = [x[0] for x in out]
        s, rp, r, sp = [x[1] for x in out]
        d = tuple(x-y for x, y in zip(r, s))
        dp = tuple(x-y for x, y in zip(rp, sp))
        delta = tuple(max(0, y-x) for x, y in zip(d, dp))
        assert all(x in (0, 1) for x in delta)
        assert all(x-y >= z for x, y, z in zip(rp, r, delta))
        d0 = tuple(min(x, y) for x, y in zip(d, dp))
        rtilde = tuple(x-y for x, y in zip(rp, delta))
        assert all(x-y >= 0 for x, y in zip(rtilde, r))
        assert V(rtilde)-V(tuple(x-y for x, y in zip(rtilde, d0))) <= V(r)-V(tuple(x-y for x, y in zip(r, d0)))
        theta = V(rp)-V(rtilde)
        conditional_difference = f[1]+f[0]-f[2]-f[3]
        assert conditional_difference <= theta
        gaps = [w-x for w, x in zip(W, f)]
        bound = gaps[0]+gaps[1]+theta
        slack = gaps[2]+gaps[3]+theta-conditional_difference
        assert bound-change == slack >= 0
        maxima = [max([ZERO]+[e[t] for e in unit_edges if t in e]) for t in range(T)]
        assert 0 <= theta <= dot(delta, maxima)
        rows.append((bound, gaps[0]+gaps[1], theta, delta))
        defects += any(delta)
    bound = min(x[0] for x in rows)
    gamma = min(x[1] for x in rows)
    assert 0 <= max(ZERO, change) <= bound
    assert bound >= gamma
    integer_condition = all(x.denominator == 1 for x in a) or all((x-y).denominator == 1 for x, y in zip(Cp, C))
    if integer_condition:
        assert all(not any(x[3]) and x[2] == 0 for x in rows)
        assert bound == gamma
    admissions = 0
    for v, q in product([F(2), F(5), F(9)], [F(1, 2), F(1), F(2)]):
        p0, p1 = W[2]-W[0], W[1]-W[3]
        rho0, rho1 = (v-p0)/q, (v-p1)/q
        assert rho1 >= rho0-bound/q
        cutoff = lambda x: max(ZERO, min(F(4), x/2))
        tau0, tau1 = cutoff(rho0), cutoff(rho1)
        assert max(ZERO, tau0-tau1) <= bound/(2*q)
        assert max(ZERO, tau0/4-tau1/4) <= bound/(8*q)
        admissions += 1
    return {'conditional_portfolios': len(rows), 'defect_portfolios': defects,
            'integer_reductions': int(integer_condition), 'admission_checks': admissions,
            'positive_changes': int(change > 0), 'old_uncorrected_bound_failures': int(change > gamma),
            'bound': bound, 'gamma': gamma, 'change': change}


def run():
    rng = random.Random(202610241245)
    counts = Counter()
    for case in range(360):
        a0 = F(case % 5, 4)
        a = a0, 1-a0
        C = tuple(x+F(rng.randrange(7), 4) for x in a)
        Cp = tuple(x+F(rng.randrange(6), 4) for x in C)
        if case % 6 == 0:
            Cp = tuple(x+rng.randrange(3) for x in C)
        edges = []
        for i in range(3+case % 2):
            e = {t: F(rng.randrange(-6, 31), 3) for t in range(2) if rng.randrange(4)}
            edges.append(e)
        menus = []
        for j in range(case % 3):
            menus.append([((ZERO, ZERO), ZERO),
                          ((F(1, 2), F(1, 2)), F(rng.randrange(-3, 25), 2)),
                          ((F(1, 4), F(3, 4)), F(rng.randrange(-3, 25), 2))])
        out = audit_case(edges, menus, C, Cp, a)
        for key, val in out.items():
            if isinstance(val, int):
                counts[key] += val
        counts['constructed_capacity_cases'] += 1
    # The prior fractional counterexample becomes an equality in the corrected bound.
    threshold = audit_case([{0: F(7)}], [], (F(3, 4), F(1)),
                          (F(1), F(1)), (F(1, 4), F(3, 4)))
    assert threshold['gamma'] == 0 and threshold['change'] == threshold['bound'] == 7
    # Both the focal profile and expansion are fractional, but no rounded
    # reservation grows: the corrected certificate is zero without the old
    # sufficient integer condition.
    protected = audit_case([{0: F(7)}, {1: F(4)}], [], (F(1), F(1)),
                          (F(5, 4), F(5, 4)), (F(1, 2), F(1, 2)))
    assert protected['bound'] == 0 and protected['integer_reductions'] == 0
    result = {'status': 'passed', 'discrepancies': 0, 'arithmetic': 'independent exact rational enumeration',
              'counts': dict(counts), 'threshold_sharpness_case': threshold,
              'fractional_protection_case': protected,
              'scope': 'Finite checks supplement the analytic proof; no external peer review or priority certification.'}
    (P/'fractional_capacity_protection.json').write_text(json.dumps(result, default=str, indent=2)+'\n')
    print(json.dumps(result, default=str), flush=True)


if __name__ == '__main__':
    run()
