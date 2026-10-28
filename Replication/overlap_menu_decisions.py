"""Validated continuous-prior menu decisions with nonproportional overlap.

The design is frozen in overlap_design.json. All inputs are mathematical
constructions. Four-variable packing vertices are enumerated completely by
exact active-set arithmetic. HiGHS supplies independent rational primal/dual
witnesses at separate report profiles; solver tolerances do not certify an
expectation. Jensen and coordinatewise chord bounds are applied separately
to the integer and relaxed maxima of affine scores.
"""
from fractions import Fraction as F
from itertools import combinations, product
from math import lcm
from pathlib import Path
from collections import Counter
import hashlib
import json
import random

import numpy as np

from prior_sensitivity import prior, outward
from recovery_certificates import Catalog, mode
from reoptimized_certificates import full_relaxation

P = Path(__file__).resolve().parent
ZERO = F(0)
D = 2**24


def dot(a, b):
    return sum((x*y for x, y in zip(a, b)), ZERO)


def solve_square(A, b):
    """Exact Gauss-Jordan elimination; singular active sets are discarded."""
    n = len(b)
    M = [list(map(F, a))+[F(y)] for a, y in zip(A, b)]
    for j in range(n):
        pivot = next((i for i in range(j, n) if M[i][j]), None)
        if pivot is None:
            return None
        M[j], M[pivot] = M[pivot], M[j]
        p = M[j][j]
        M[j] = [x/p for x in M[j]]
        for i in range(n):
            if i != j and M[i][j]:
                q = M[i][j]
                M[i] = [x-q*y for x, y in zip(M[i], M[j])]
    return tuple(M[i][-1] for i in range(n))


def packing_vertices(profiles):
    n = len(profiles)
    A = [tuple(F(i == j) for i in range(n)) for j in range(n)]
    b = [F(1)]*n
    A += [tuple(-F(i == j) for i in range(n)) for j in range(n)]
    b += [ZERO]*n
    A += [tuple(a[t] for a in profiles) for t in range(3)]
    b += [F(1)]*3
    vertices = set()
    nonsingular = 0
    for ix in combinations(range(len(A)), n):
        x = solve_square([A[i] for i in ix], [b[i] for i in ix])
        if x is None:
            continue
        nonsingular += 1
        if all(dot(a, x) <= y for a, y in zip(A, b)):
            vertices.add(x)
    assert (ZERO,)*n in vertices
    # Strictly positive small vectors make this polytope full dimensional.
    # Every vertex therefore has four independent active inequalities.
    assert all(dot(a, [F(1, 20)]*n) < y for a, y in zip(A[:n]+A[2*n:], b[:n]+b[2*n:]))
    return sorted(vertices), {'active_sets': 330, 'nonsingular_active_sets': nonsingular,
                              'feasible_vertices': len(vertices)}


def integer_allocations(profiles):
    return [tuple(map(F, z)) for z in product((0, 1), repeat=4)
            if all(sum(profiles[i][t]*z[i] for i in range(4)) <= 1
                   for t in range(3))]


def scaled_options(options, values):
    K = lcm(*(x.denominator for z in options for x in z))
    q = np.array([[int(K*x) for x in z] for z in options], dtype=np.int64)
    intercept = np.array([int(K*dot(z, values)) for z in options], dtype=np.int64)
    return K, q, intercept


def rounded_penalties(points, upward):
    ans = []
    for x in points:
        y = x*D
        k = -((-y.numerator)//y.denominator) if upward else y.numerator//y.denominator
        assert F(k, D) >= x if upward else F(k, D) <= x
        ans.append(k)
    return np.array(ans, dtype=np.int64)


def grid_values(scaled, points, upward):
    K, q, intercept = scaled
    pp = rounded_penalties(points, upward)
    assert int(max(abs(int(a)) for a in intercept))*D + 4*int(q.max())*int(pp.max()) < np.iinfo(np.int64).max
    shape = (len(points),)*4
    size = len(points)**4
    out = np.empty(size, dtype=np.int64)
    for start in range(0, size, 8192):
        ix = np.arange(start, min(start+8192, size), dtype=np.int64)
        coords = np.array(np.unravel_index(ix, shape)).T
        out[start:start+len(ix)] = (intercept[None, :]*D-pp[coords]@q.T).max(axis=1)
    assert out.min() >= 0
    return out.reshape(shape), K*D


def expectation_bounds(scaled, beta, m):
    endpoints, means, weights = prior(beta, m)
    low, den = grid_values(scaled, means, True)
    lo = F(sum(map(int, low.flat)), den*m**4)
    high, den = grid_values(scaled, endpoints, False)
    W = lcm(*(x.denominator for x in weights))
    wi = np.array([int(x*W) for x in weights], dtype=object)
    x = high.astype(object)
    for _ in range(4):
        x = np.tensordot(wi, x, axes=(0, 0))
    hi = F(int(x), den*W**4)
    assert lo <= hi
    return lo, hi


def globally_affine(options, values, retained, beta):
    candidate = tuple(F(i in retained) for i in range(4))
    maxpenalty = F(4)*(1+beta)
    for z in options:
        minimum = dot(candidate, values)-dot(z, values)-maxpenalty*sum(max(ZERO, a-b) for a, b in zip(candidate, z))
        if minimum < 0:
            return None
    # E phi(theta)=4 for all three priors, by the scalar envelope identity.
    return dot(candidate, [v-4 for v in values])


def classification(full, relaxed, target=F(19, 20)):
    actual = F(16)/full[1], min(F(1), F(16)/full[0])
    cert = F(16)/relaxed[1], min(F(1), F(16)/relaxed[0])
    if cert[0] >= target:
        result = 'certified'
    elif actual[1] < target:
        result = 'below target'
    elif actual[0] >= target and cert[1] < target:
        result = 'retained but uncertified'
    else:
        result = 'numerically unresolved'
    return result, actual, cert


def independent_checks(family, vertices, allocations, values, H):
    profiles = [tuple(map(F, a)) for a in family['recovery_profiles']]
    ins = Catalog([[mode([0]*4), mode([0]+list(a), 1)] for a in profiles], values, [0, 1, 1, 1])
    rng = random.Random(20261024+ord(family['id'])+H)
    tests = [[ZERO]*4, [F(12)]*4, [F(4)]*4]
    tests += [[F(rng.randrange(49), 4) for _ in range(4)] for _ in range(9)]
    positive_gaps = 0
    for penalty in tests:
        weights = [v-s for v, s in zip(values, penalty)]
        vf = max(dot(z, weights) for z in allocations)
        lp = max(dot(z, weights) for z in vertices)
        assert vf == ins.optimum(penalty)
        assert lp == full_relaxation(ins, penalty)
        assert vf <= lp
        positive_gaps += lp > vf
    return {'rational_lp_vertex_equalities': len(tests), 'positive_gap_profiles': positive_gaps}


def run():
    raw = (P/'overlap_design.json').read_bytes()
    spec = json.loads(raw)
    assert spec['frozen_before_numerical_results']
    assert spec['retention_target'] == '19/20'
    rows, structural = [], {}
    for family in spec['families']:
        profiles = [tuple(map(F, a)) for a in family['recovery_profiles']]
        assert all(sum(a) == 1 for a in profiles)
        assert all(any(a[t] and b[t] for t in range(3)) for a, b in combinations(profiles[1:], 2)) if family['id'] != 'C' else True
        vertices, structure = packing_vertices(profiles)
        allocations = integer_allocations(profiles)
        assert all(z in vertices for z in allocations)
        structure['integer_allocations'] = len(allocations)
        structure['independent_checks'] = []
        for H in spec['public_bundle_values']:
            values = [F(H) if v == 'H' else F(v) for v in family['values']]
            check = independent_checks(family, vertices, allocations, values, H)
            structure['independent_checks'].append({'H': H, **check})
            full_options = scaled_options(allocations, values)
            lp_options = scaled_options(vertices, values)
            for beta in map(F, spec['cost_priors_beta']):
                restricted = sum(values[i]-4 for i in family['retained_applicants'])
                assert restricted == 16
                full_exact = globally_affine(allocations, values, family['retained_applicants'], beta)
                lp_exact = globally_affine(vertices, values, family['retained_applicants'], beta)
                full = relaxed = None
                steps = []
                for m in spec['integration_cells']:
                    f = (full_exact, full_exact) if full_exact is not None else expectation_bounds(full_options, beta, m)
                    l = (lp_exact, lp_exact) if lp_exact is not None else expectation_bounds(lp_options, beta, m)
                    # The restricted mechanism is feasible in both optimizations.
                    f = max(F(16), f[0]), f[1]
                    l = max(F(16), l[0], f[0]), l[1]
                    full = f if full is None else (max(full[0], f[0]), min(full[1], f[1]))
                    relaxed = l if relaxed is None else (max(relaxed[0], l[0]), min(relaxed[1], l[1]))
                    assert full[0] <= full[1] and relaxed[0] <= relaxed[1]
                    steps.append(m)
                    decision, actual_ratio, certificate_ratio = classification(full, relaxed)
                    if decision != 'numerically unresolved':
                        break
                loss = max(ZERO, full[0]-16), full[1]-16
                certificate = max(ZERO, relaxed[0]-16), relaxed[1]-16
                gap = max(ZERO, relaxed[0]-full[1]), relaxed[1]-full[0]
                assert loss[0] <= loss[1] and certificate[0] <= certificate[1] and gap[0] <= gap[1]
                dec = lambda x, p=4: [outward(x[0], p), outward(x[1], p, True)]
                row = {'family': family['id'], 'H': H, 'beta': beta, 'partitions_used': steps,
                       'restricted_expected_receipts': F(16), 'full_expected_receipts': full,
                       'full_lp_expectation': relaxed, 'actual_expected_loss': loss,
                       'refined_certificate_expectation': certificate, 'expected_integrality_gap': gap,
                       'actual_retention_interval': actual_ratio, 'certificate_retention_interval': certificate_ratio,
                       'full_exact_by_affine_dominance': full_exact is not None,
                       'lp_exact_by_affine_dominance': lp_exact is not None, 'decision': decision,
                       'display': {'loss': dec(loss), 'certificate': dec(certificate), 'gap': dec(gap),
                           'actual_retention_percent': dec([100*x for x in actual_ratio], 2),
                           'certificate_retention_percent': dec([100*x for x in certificate_ratio], 2)}}
                rows.append(row)
                print(family['id'], H, beta, steps, decision, row['display'], flush=True)
        structural[family['id']] = {**structure, 'vertices': vertices, 'integer_allocations_vectors': allocations}
    counts = dict(Counter(row['decision'] for row in rows))
    report = {'status': 'passed', 'discrepancies': 0, 'scenarios': len(rows),
              'design_sha256': hashlib.sha256(raw).hexdigest(), 'all_inputs_constructed': True,
              'target': '19/20', 'arithmetic': 'exact rational vertices and deterministic expectation enclosures',
              'structural_checks': structural, 'decision_counts': counts, 'rows': rows,
              'interpretation': 'Uncertified good menus are distinct from proven target failure; no sampling uncertainty is used.'}
    assert len(rows) == 27
    (P/'overlap_menu_decisions.json').write_text(json.dumps(report, default=str, indent=2)+'\n')
    print('SUMMARY', counts, flush=True)


if __name__ == '__main__':
    run()
