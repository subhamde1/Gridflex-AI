"""Exact-rational witnesses for the reoptimized certificate and separation.

HiGHS proposes LP solutions. Every accepted solution is reconstructed as
rational numbers, and primal feasibility, dual feasibility and equal
objectives are checked without a floating-point tolerance. These finite
checks supplement the manuscript's proofs; they do not establish priority.
"""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import json
import random

import numpy as np
from scipy.optimize import linprog

from recovery_certificates import (Catalog, mode, optimized_certificate,
                                   unit_matching_value, welfare)

P = Path(__file__).resolve().parent
Z = F(0)
COUNTS = {}


def count(key):
    COUNTS[key] = COUNTS.get(key, 0) + 1


def dot(x, y):
    return sum((a*b for a, b in zip(x, y)), Z)


def exact_min_lp(c, A, b, E=(), d=()):
    """Nonnegative-variable LP with <= inequalities and optional equalities."""
    kwargs = dict(A_ub=np.array(A, float), b_ub=np.array(b, float),
                  bounds=(0, None), method='highs')
    if E:
        kwargs.update(A_eq=np.array(E, float), b_eq=np.array(d, float))
    sol = linprog(np.array(c, float), **kwargs)
    assert sol.success, sol.message
    rat = lambda x: F(float(x)).limit_denominator(10**7)
    x = list(map(rat, sol.x))
    u = list(map(rat, sol.ineqlin.marginals))
    v = list(map(rat, sol.eqlin.marginals)) if E else []
    assert all(t >= 0 for t in x)
    assert all(dot(row, x) <= rhs for row, rhs in zip(A, b))
    assert all(dot(row, x) == rhs for row, rhs in zip(E, d))
    assert all(t <= 0 for t in u)
    assert all(c[j] - sum(row[j]*u[k] for k, row in enumerate(A))
               - sum(row[j]*v[k] for k, row in enumerate(E)) >= 0
               for j in range(len(c)))
    value = dot(c, x)
    assert value == dot(b, u) + dot(d, v)
    count('rational_primal_dual_equalities')
    return value, x, u, v


def full_relaxation(ins, penalties):
    columns = [(i, m) for i, ms in enumerate(ins.menus)
               for m in range(1, len(ms))]
    weights = [ins.score(i, m, penalties) for i, m in columns]
    A = [[F(i == k) for i, m in columns] for k in range(ins.n)]
    A += [[ins.menus[i][m][0][t] for i, m in columns]
          for t in range(ins.T)]
    b = [F(1)]*ins.n + list(ins.capacity)
    objective, y, multipliers, _ = exact_min_lp([-w for w in weights], A, b)
    beta, lam = [-u for u in multipliers[:ins.n]], [-u for u in multipliers[ins.n:]]
    assert all(beta[i] + dot(lam, ins.menus[i][m][0]) >= w
               for (i, m), w in zip(columns, weights))
    D = dot(lam, ins.capacity) + sum(max([Z] + [ins.score(i, m, penalties)
             - dot(lam, ins.menus[i][m][0]) for m in range(1, len(ms))])
             for i, ms in enumerate(ins.menus))
    assert -objective == dot(weights, y) == sum(beta) + dot(lam, ins.capacity) == D
    return -objective


def restricted_price_certificate(ins, penalties, retained):
    """Optimal-face LP; no full integer range is enumerated by this routine."""
    n, T = ins.n, ins.T
    unit = unit_matching_value(ins, penalties, retained)
    r = [F(c.numerator // c.denominator) for c in ins.capacity]
    c = [Z]*n + [ins.capacity[t]-r[t] for t in range(T)] + [F(1)]*n
    A, b = [], []
    for i, ms in enumerate(ins.menus):
        for m in range(1, len(ms)):
            row = [Z]*(2*n+T)
            row[i] = -1
            for t in range(T):
                row[n+t] = -ms[m][0][t]
            if m not in retained[i]:
                row[n+T+i] = -1
            A.append(row)
            b.append(-ins.score(i, m, penalties))
    E = [[F(1)]*n + r + [Z]*n]
    value, x, _, _ = exact_min_lp(c, A, b, E, [unit])
    count('restricted_price_lp_checks')
    return value


def general_catalog_checks():
    rng = random.Random(2026102407)
    for case in range(240):
        T, n = 2 + case % 2, 3 + (case % 4 == 0)
        menus, retained = [], []
        for i in range(n):
            ms = [mode([0]*T)]
            for t in range(T):
                if rng.randrange(4):
                    a = [Z]*T; a[t] = F(1)
                    ms.append(mode(a, F(rng.randrange(1, 5), 2), F(rng.randrange(3), 3)))
            keep = set(range(len(ms)))
            for j in range(1 + (i+case) % 2):
                a = [Z]*T
                t, u = rng.sample(range(T), 2)
                a[t] = F(rng.randrange(1, 4), 4); a[u] = 1-a[t]
                ms.append(mode(a, F(rng.randrange(1, 6), 2), F(rng.randrange(4), 3)))
            assert all(sum(m[0]) == 1 and m[1] > 0 for m in ms[1:])
            menus.append(ms); retained.append(keep)
        capacity = [F(rng.randrange(1, 7), 2) for t in range(T)]
        ins = Catalog(menus, [rng.randrange(1, 10) for i in range(n)], capacity)
        penalties = [F(rng.randrange(10), 3) for i in range(n)]
        old = optimized_certificate(ins, penalties, retained)
        L = full_relaxation(ins, penalties)
        new = L - old['unit']
        assert 0 <= old['gap'] <= new <= old['bound']
        assert new-old['gap'] == L-old['full']
        old2 = restricted_price_certificate(ins, penalties, retained)
        assert old2 == old['bound']
        count('general_catalog_inequality_chains')
        if L == old['full']:
            assert new == old['gap']
            count('integral_relaxation_exactness_checks')
        if old['gap'] == 0 and new > 0:
            count('zero_actual_loss_positive_integrality_gap')
        if new < old['bound']:
            count('strict_certificate_improvements')


def family_checks():
    rng = random.Random(2026102408)
    parameters = [(F(3, 4), F(1), F(20), F(35, 2)),
                  (F(2, 3), F(1, 2), F(12), F(19, 2)),
                  (F(4, 5), F(1, 3), F(20), F(18))]
    for a, h, V, H in parameters:
        assert a*V+2*h < H < V-2*h
        for n in [2, 3, 4, 8, 16, 32, 64]:
            for trial in range(5):
                theta = ([Z]*(n+1) if trial == 0 else [h]*(n+1)
                         if trial == 1 else [h*F(rng.randrange(13), 12) for j in range(n+1)])
                menus = [[mode([0, 0]), mode([1, 0], 1)]]
                menus += [[mode([0, 0]), mode([a, 1-a], 1)] for j in range(n)]
                ins = Catalog(menus, [V]+[H]*n, [1, 1])
                retained = [{0, 1}] + [{0} for j in range(n)]
                penalties = [2*t for t in theta]
                w0, wj = V-2*theta[0], [H-2*t for t in theta[1:]]
                assert all(w0 > w > a*w0 > 0 for w in wj)
                assert 2*a > 1 and a+1 > 1
                integer_full = max([Z, w0]+wj)
                assert integer_full == unit_matching_value(ins, penalties, retained) == w0
                if n <= 8:
                    assert ins.optimum(penalties) == w0
                    count('enumerated_separation_integer_optima')
                expected_old = n*(H-a*V)+2*a*n*theta[0]-2*sum(theta[1:])
                assert restricted_price_certificate(ins, penalties, retained) == expected_old
                t1, t2 = sorted(theta[1:])[:2]
                predicted_L = H/a-2*(t1+(1/a-1)*t2)
                L = full_relaxation(ins, penalties)
                assert L == predicted_L
                assert 0 < L-w0 <= expected_old
                count('separation_report_profiles')
            # Constant allocation: exact envelope, including both endpoints.
            for t, report in product([Z, h/3, h], repeat=2):
                charge = V-h
                utility = V-t-charge
                assert utility == h-t >= 0
                report_charge = V-report-(h-report)  # direct envelope evaluation
                assert utility >= V-t-report_charge
                count('separation_envelope_deviations')
    rows = []
    for n in range(2, 65):
        old = F(9*n, 4)
        refined = F(13, 3)-F(10, 3*(n+1))
        assert 0 < refined < old
        rows.append({'n': n, 'restricted_expected_bound': str(old),
                     'reoptimized_expected_bound': str(refined), 'actual_loss': '0',
                     'expected_receipts': '19',
                     'restricted_retention_bound': str(19/(19+old)),
                     'reoptimized_retention_bound': str(19/(19+refined)),
                     'actual_retention': '1'})
    assert rows[14]['reoptimized_expected_bound'] == '211/51'
    return rows


def slack_identity_checks():
    rng = random.Random(2026102409)
    for case in range(120):
        T = 2
        C = tuple(F(rng.randrange(2, 7), 2) for t in range(T))
        Cp = tuple(C[t]+rng.randrange(3) for t in range(T))  # integer expansion
        ordinary = [(tuple(t for t in range(T) if rng.randrange(3)), F(rng.randrange(-2, 10)))
                    for j in range(3)]
        bundles = [((F(1, 2), F(1, 2)), F(rng.randrange(-2, 12)))
                   for j in range(case % 3)]
        a = (F(1), Z) if case % 2 else (F(1, 2), F(1, 2))
        low = tuple(C[t]-a[t] for t in range(T))
        lowp = tuple(Cp[t]-a[t] for t in range(T))
        states = [low, Cp, C, lowp]
        W = [welfare(ordinary, bundles, h) for h in states]
        choices = []
        for z in product((0, 1), repeat=len(bundles)):
            if all(sum(bundles[j][0][t]*z[j] for j in range(len(bundles))) <= low[t]
                   for t in range(T)):
                Fz = [welfare(ordinary, bundles, h, z) for h in states]
                choices.append((Fz[0]+Fz[1], z, Fz))
        common, _, Fz = max(choices)
        gamma = W[0]+W[1]-common
        change = (W[1]-W[3])-(W[2]-W[0])
        sigma = Fz[2]+Fz[3]-Fz[0]-Fz[1]
        middle = (W[2]-Fz[2])+(W[3]-Fz[3])
        assert sigma >= 0 and middle >= 0 and gamma >= 0
        assert gamma-change == middle+sigma
        if gamma > 0:
            assert (max(Z, change) == gamma) == (middle == 0 and sigma == 0)
        count('portfolio_slack_identities')


def area_formula_checks():
    rng = random.Random(2026102410)
    for case in range(100):
        J = rng.randrange(2, 7)
        breaks = [F(j, 2) for j in range(J+1)]
        q = sorted([F(rng.randrange(9), 3) for j in range(J)], reverse=True)
        top = breaks[-1]
        tests = sorted(set(breaks+[F(rng.randrange(2*J+1), 4) for k in range(8)]))
        for t in tests:
            exact = sum(q[j]*max(Z, breaks[j+1]-max(t, breaks[j])) for j in range(J))
            areas = q[-1]*(top-t)+sum((q[j-1]-q[j])*max(Z, breaks[j]-t)
                                               for j in range(1, J))
            assert areas == exact
            count('critical_exposure_area_evaluations')


def run():
    general_catalog_checks()
    rows = family_checks()
    slack_identity_checks()
    area_formula_checks()
    result = {'status': 'passed', 'discrepancies': 0, 'counts': COUNTS,
              'separation_figure_data': rows,
              'scope': 'Constructed catalogs; exact rational LP witnesses and finite checks supplement proofs.'}
    (P/'reoptimized_certificates.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'status': 'passed', 'discrepancies': 0, 'counts': COUNTS}))


if __name__ == '__main__':
    run()
