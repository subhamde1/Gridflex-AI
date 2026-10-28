"""Validated ex-ante menu decisions on the disclosed 27-scenario design.

No observations are imported. The full LP is a block-assignment value in
this design (proved in the article), not a general integer clearing oracle.
Quantile-cell Jensen and chord bounds are applied separately to each value;
the difference of values is never assumed convex. Directed dyadic rounding
avoids a common-denominator overflow, with exact rational expectations.
"""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import json
import random

import numpy as np
from scipy.optimize import linear_sum_assignment

from recovery_certificates import Catalog, mode, study_catalog, optimized_certificate
from prior_sensitivity import prior, outward
from reoptimized_certificates import full_relaxation

P = Path(__file__).resolve().parent
D = 2**24
ZERO = F(0)


def compressed_allocations(stress, k, full=True):
    """Theorem's exact block assignment, independent of portfolio enumeration."""
    options = [[0, 1, 2] + ([3] if full and i < k else []) for i in range(4)]
    return [z for z in product(*options)
            if z.count(1) <= stress and z.count(2) <= 1 and z.count(3) <= 2]


def rounded_scores(points, upward):
    result = []
    for p in points:
        x = p*D
        v = -((-x.numerator)//x.denominator) if upward else x.numerator//x.denominator
        result.append(v)
        assert (F(v, D) >= p) if upward else (F(v, D) <= p)
    return np.asarray(result, dtype=np.int64)


def value_tensor(stress, k, points, full=True, upward=False):
    pp = rounded_scores(points, upward)
    aa = compressed_allocations(stress, k, full)
    b = np.asarray([sum((4+i)*D for i, m in enumerate(z) if m) for z in aa], dtype=np.int64)
    q = np.asarray([[int(m in (2, 3)) for m in z] for z in aa], dtype=np.int64)
    assert int(b.max()) + 4*int(max(abs(int(p)) for p in pp)) < np.iinfo(np.int64).max
    shape = (len(points),)*4
    size = len(points)**4
    out = np.empty(size, dtype=np.int64)
    for start in range(0, size, 4096):
        ix = np.arange(start, min(size, start+4096), dtype=np.int64)
        coords = np.array(np.unravel_index(ix, shape)).T
        ss = pp[coords]
        out[start:start+len(ix)] = (b[None, :] - ss @ q.T).max(axis=1)
    assert out.min() >= 0
    return out.reshape(shape)


def expectation_interval(beta, stress, k, m, full=True):
    endpoints, means, weights = prior(beta, m)
    # Nonnegative exposure makes V nonincreasing in each penalty.
    low = value_tensor(stress, k, means, full, upward=True)
    lower = F(sum(map(int, low.flat)), D*m**4)
    high = value_tensor(stress, k, endpoints, full, upward=False)
    from math import lcm
    W = lcm(*(p.denominator for p in weights))
    wi = np.asarray([int(p*W) for p in weights], dtype=object)
    x = high.astype(object)
    for _ in range(4):
        x = np.tensordot(wi, x, axes=(0, 0))
    upper = F(int(x), D*W**4)
    assert lower <= upper
    return lower, upper


def matching_value(ins, penalty, block_profiles, eligibility, capacities):
    slots = [g for g, h in enumerate(capacities) for _ in range(h)]
    edges = {}
    for i, eligible in enumerate(eligibility):
        for g, mid in eligible.items():
            assert ins.menus[i][mid][0] == block_profiles[g]
            edges[i, g] = ins.score(i, mid, penalty)
    from math import lcm
    den = lcm(*(w.denominator for w in edges.values())) if edges else 1
    mat = np.zeros((ins.n, len(slots)+ins.n), dtype=np.int64)
    mat[:, :len(slots)] = -1-sum(abs(int(w*den)) for w in edges.values())
    for i in range(ins.n):
        for j, g in enumerate(slots):
            if (i, g) in edges:
                mat[i, j] = int(edges[i, g]*den)
    rr, cc = linear_sum_assignment(mat, maximize=True)
    return F(sum(int(mat[i, j]) for i, j in zip(rr, cc)), den)


def structural_checks():
    rng = random.Random(202610241013)
    count = 0
    for case in range(96):
        patterns = [(F(1), ZERO, ZERO, ZERO), (ZERO, F(1), ZERO, ZERO),
                    (ZERO, ZERO, F(1, 3), F(2, 3))]
        h = [case % 3, 1+(case//3) % 2, 1+(case//6) % 2]
        cap = [F(h[0]), F(h[1]), h[2]*F(1, 3), h[2]*F(2, 3)+F(1, 7)]
        menus, eligible = [], []
        for i in range(3):
            ms = [mode([0]*4)]
            eg = {}
            for g, a in enumerate(patterns):
                if g == 0 or rng.randrange(3):
                    eg[g] = len(ms)
                    ms.append(mode(a, F(rng.randrange(5), 3), F(rng.randrange(4), 5)))
            if len(eg) >= 2:
                g, j = rng.sample(list(eg), 2)
                p = F(rng.randrange(1, 4), 4)
                mg, mj = ms[eg[g]], ms[eg[j]]
                aa = [p*x+(1-p)*y for x, y in zip(mg[0], mj[0])]
                ms.append(mode(aa, p*mg[1]+(1-p)*mj[1]+F(case%2, 7),
                               p*mg[2]+(1-p)*mj[2]+F(case%3, 7)))
            menus.append(ms); eligible.append(eg)
        ins = Catalog(menus, [1+case%5, 4, 6], cap)
        penalty = [F(rng.randrange(15), 2) for _ in menus]
        lp = full_relaxation(ins, penalty)
        assert lp == ins.optimum(penalty) == matching_value(ins, penalty, patterns, eligible, h)
        count += 1
    # In the study, the half-firm mode can be split in a proof, and the
    # paired mode is a base block with two available slots.
    pointwise = []
    for stress, k in product(range(3), [0, 2, 4]):
        ins, keep = study_catalog(stress, k)
        for trial in range(13):
            penalties = ([F(4)]*4 if trial == 0 else
                         [F(rng.randrange(25), 2) for _ in range(4)])
            lp = full_relaxation(ins, penalties)
            compressed = max(sum(ins.values[i]*(m != 0)-penalties[i]*(m in (2, 3))
                                 for i, m in enumerate(z))
                             for z in compressed_allocations(stress, k))
            assert lp == ins.optimum(penalties) == compressed
            count += 1
            if trial == 0:
                old = optimized_certificate(ins, penalties, keep)
                pointwise.append({'stress_capacity': stress, 'bundled_eligible': k,
                    'integer_full': lp, 'integer_unit': old['unit'],
                    'actual_loss': old['gap'], 'reoptimized_certificate': lp-old['unit'],
                    'restricted_price_certificate': old['bound']})
    # Overlapping supports: pairwise bundles form a fractional triangle.
    tri = Catalog([[mode([0, 0, 0]), mode(a)] for a in
                   ([1, 1, 0], [1, 0, 1], [0, 1, 1])], [1, 1, 1], [1, 1, 1])
    assert tri.optimum([ZERO]*3) == 1
    assert full_relaxation(tri, [ZERO]*3) == F(3, 2)
    # Fractional effective block capacity: identical profiles need not suffice.
    frac = Catalog([[mode([0, 0]), mode([F(1, 2), F(1, 2)])]
                    for _ in range(2)], [1, 1], [F(3, 4)]*2)
    assert frac.optimum([ZERO]*2) == 1
    assert full_relaxation(frac, [ZERO]*2) == F(3, 2)
    return {'rational_lp_matching_enumeration_equalities': count,
            'boundary_examples': 2, 'pointwise_comparisons': pointwise}


def decision(lo, hi, target=F(19, 20)):
    return 'certified' if lo >= target else 'below target' if hi < target else 'unresolved'


def run():
    checks = structural_checks()
    original = json.loads((P/'prior_sensitivity.json').read_text())
    rows = []
    # Previously validated rational intervals are reused at m=12. The block
    # theorem identifies these same values with the full LP, profile by profile.
    for old in original['rows']:
        beta = F(old['beta']); stress = old['stress_capacity']; k = old['bundled_eligible']
        fl, fu = map(F, old['full_receipts']); ul, uu = map(F, old['unit_receipts'])
        steps = [12]
        if k == 0:
            blo = bhi = ZERO; rlo = rhi = F(1)
        else:
            blo, bhi = max(ZERO, fl-uu), fu-ul
            rlo, rhi = ul/fu, min(F(1), uu/fl)
            if decision(rlo, rhi) == 'unresolved':
                for m in (24, 48):
                    nfl, nfu = expectation_interval(beta, stress, k, m, True)
                    nul, nuu = expectation_interval(beta, stress, k, m, False)
                    fl, fu = max(fl, nfl), min(fu, nfu)
                    ul, uu = max(ul, nul), min(uu, nuu)
                    assert fl <= fu and ul <= uu
                    steps.append(m)
                    blo, bhi = max(ZERO, fl-uu), fu-ul
                    rlo, rhi = ul/fu, min(F(1), uu/fl)
                    if decision(rlo, rhi) != 'unresolved':
                        break
        row = {'beta': beta, 'stress_capacity': stress, 'bundled_eligible': k,
               'partitions_used': steps, 'full_lp_expectation': [fl, fu],
               'unit_expectation': [ul, uu], 'benchmark_loss': [blo, bhi],
               'refined_certificate_expectation': [blo, bhi],
               'certificate_excess_over_actual_loss': [ZERO, ZERO],
               'retention_interval': [rlo, rhi], 'target': F(19, 20),
               'decision': decision(rlo, rhi),
               'display': {'loss': [outward(blo, 4), outward(bhi, 4, True)],
                   'retention_percent': [outward(100*rlo, 2), outward(100*rhi, 2, True)]}}
        rows.append(row)
        print(beta, stress, k, steps, row['display'], row['decision'], flush=True)
    report = {'status': 'passed', 'discrepancies': 0, 'scenarios': len(rows),
              'all_inputs_constructed': True, 'retention_target': '19/20',
              'refinement_rule': 'm=12 then 24 then 48 only if 95% unresolved',
              'dyadic_denominator_for_refinements': D,
              'rounding_value_error_bound': F(4, D),
              'structural_checks': checks,
              'decision_counts': {key: sum(r['decision'] == key for r in rows)
                                  for key in ('certified', 'below target', 'unresolved')},
              'rows': rows}
    (P/'menu_standardization.json').write_text(json.dumps(report, default=str, indent=2)+'\n')
    print('SUMMARY', report['decision_counts'], flush=True)


if __name__ == '__main__':
    run()
