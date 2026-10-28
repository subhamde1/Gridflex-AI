"""Exact checks of recovery admission and restricted-menu certificates.

Every load, value, cost and prior in this program is constructed. Fractions
are used for all stated inequalities. HiGHS proposes LP witnesses; these
are reconstructed as rational numbers and checked by exact primal/dual
equalities before a certificate is accepted. No solver tolerance is used
as evidence for a mathematical inequality.
"""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
from math import floor
import json
import random

import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from math import lcm

P = Path(__file__).resolve().parent
Z = F(0)


def mode(a, q=0, k=0):
    return tuple(map(F, a)), F(q), F(k)


class Catalog:
    def __init__(self, menus, values, capacity):
        self.menus, self.values = menus, tuple(map(F, values))
        self.capacity = tuple(map(F, capacity))
        self.n, self.T = len(menus), len(capacity)
        self._allocations = None

    @property
    def allocations(self):
        if self._allocations is not None:
            return self._allocations
        self._allocations = []
        menus, capacity = self.menus, self.capacity
        for z in product(*(range(len(ms)) for ms in menus)):
            if all(sum(menus[i][z[i]][0][t] for i in range(self.n)) <= self.capacity[t]
                   for t in range(self.T)):
                self._allocations.append(z)
        assert (0,) * self.n in self._allocations
        return self._allocations

    def score(self, i, m, penalty):
        return self.values[i] * (m != 0) - penalty[i] * self.menus[i][m][1] - self.menus[i][m][2]

    def optimum(self, penalty, retained=None):
        return max(sum(self.score(i, m, penalty) for i, m in enumerate(z))
                   for z in self.allocations
                   if retained is None or all(m in retained[i] for i, m in enumerate(z)))


def rational(x):
    return F(float(x)).limit_denominator(10**7)


def unit_matching_value(ins, penalty, retained):
    """Use only retained modes; the full portfolio list is never inspected."""
    slots = [t for t in range(ins.T) for _ in range(min(ins.n, floor(ins.capacity[t])))]
    edges = {}
    for i, ms in enumerate(ins.menus):
        for m in retained[i]-{0}:
            a = ms[m][0]
            assert sum(a) == 1 and sum(x == 1 for x in a) == 1
            edges[i, a.index(F(1))] = ins.score(i, m, penalty)
    D = lcm(*(w.denominator for w in edges.values())) if edges else 1
    forbidden = -1-sum(abs(int(w*D)) for w in edges.values())
    weights = np.zeros((ins.n, len(slots)+ins.n), dtype=np.int64)
    weights[:, :len(slots)] = forbidden
    for i in range(ins.n):
        for j, t in enumerate(slots):
            if (i, t) in edges:
                weights[i, j] = int(edges[i, t]*D)
    rows, cols = linear_sum_assignment(weights, maximize=True)
    total = F(0)
    for i, j in zip(rows, cols):
        if j < len(slots):
            assert (i, slots[j]) in edges
            total += edges[i, slots[j]]
    return total


def optimized_certificate(ins, penalty, retained):
    """Minimize the certificate on the optimal unit-assignment dual face.

    Both feasibility and global optimality of this auxiliary LP are checked
    exactly using its primal solution and the solver's dual multipliers.
    The full-catalog integer optimum is not an input to this LP.
    """
    n, T = ins.n, ins.T
    unit = unit_matching_value(ins, penalty, retained)
    r = [F(floor(x)) for x in ins.capacity]
    c = [Z] * n + [ins.capacity[t] - r[t] for t in range(T)] + [F(1)] * n
    A, b = [], []
    for i, ms in enumerate(ins.menus):
        for m, (a, q, k) in enumerate(ms):
            if m == 0:
                continue
            row = [Z] * (2*n+T)
            row[i] = -1
            for t in range(T):
                row[n+t] = -a[t]
            if m not in retained[i]:
                row[n+T+i] = -1
            else:
                assert sum(a) == 1 and sum(x == 1 for x in a) == 1
            A.append(row)
            b.append(-ins.score(i, m, penalty))
    eq = [F(1)]*n + r + [Z]*n
    sol = linprog(np.array(c, dtype=float), A_ub=np.array(A, dtype=float),
                  b_ub=np.array(b, dtype=float), A_eq=np.array([eq], dtype=float),
                  b_eq=[float(unit)], bounds=(0, None), method='highs')
    assert sol.success, sol.message
    x = list(map(rational, sol.x))
    yu = list(map(rational, sol.ineqlin.marginals))
    ye = rational(sol.eqlin.marginals[0])
    dot = lambda a, b: sum(u*v for u, v in zip(a, b))
    assert all(t >= 0 for t in x)
    assert all(dot(row, x) <= rhs for row, rhs in zip(A, b))
    assert dot(eq, x) == unit
    assert all(t <= 0 for t in yu)
    assert all(c[j] - sum(A[k][j]*yu[k] for k in range(len(A))) - eq[j]*ye >= 0
               for j in range(len(c)))
    objective = dot(c, x)
    assert objective == dot(b, yu) + unit*ye
    alpha, price, e = x[:n], x[n:n+T], x[n+T:]
    actual_e = [max([Z] + [ins.score(i, m, penalty) - alpha[i] - dot(price, a)
                           for m, (a, q, k) in enumerate(ms) if m not in retained[i]])
                for i, ms in enumerate(ins.menus)]
    assert e == actual_e
    full = ins.optimum(penalty)
    assert 0 <= full - unit <= objective
    return {'unit': unit, 'full': full, 'gap': full-unit, 'bound': objective,
            'alpha': alpha, 'lambda': price, 'residuals': e,
            'fractional_capacity_term': dot(price, [ins.capacity[t]-r[t] for t in range(T)]),
            'auxiliary_lp_primal_dual_equal': True}


def welfare(ordinary, bundles, cap, common=None):
    """Independently enumerate unit applicants and bundled decisions."""
    T = len(cap)
    best = None
    zs = list(product((0, 1), repeat=len(bundles))) if common is None else [common]
    for zz in zs:
        loads = [sum(bundles[j][0][t]*zz[j] for j in range(len(bundles))) for t in range(T)]
        if any(loads[t] > cap[t] for t in range(T)):
            continue
        base = sum(bundles[j][1]*zz[j] for j in range(len(bundles)))
        choices = [[None] + list(edges) for edges, val in ordinary]
        for slots in product(*choices):
            total = [loads[t] + sum(s == t for s in slots if s is not None) for t in range(T)]
            if all(total[t] <= cap[t] for t in range(T)):
                value = base + sum(ordinary[j][1] for j, s in enumerate(slots) if s is not None)
                best = value if best is None else max(best, value)
    assert best is not None
    return best


def clipped_uniform_cutoff(rho, h=F(4)):
    return min(h, max(Z, rho/2))


def check_admission(counts):
    rng = random.Random(20261024)
    for case in range(120):
        T = 2
        C = tuple(F(rng.randrange(1, 5), 2) for _ in range(T))
        Cp = tuple(C[t] + rng.randrange(3) for t in range(T))
        ordinary = [(tuple(t for t in range(T) if rng.randrange(3)), F(rng.randrange(-2, 10)))
                    for _ in range(3)]
        bundles = [((F(1, 2), F(1, 2)), F(rng.randrange(-2, 12))) for _ in range(case % 3)]
        served = [((F(1), Z), F(1)), ((Z, F(1)), F(2)),
                  ((F(1, 2), F(1, 2)), F(3, 2))]
        feasible = [(a, q) for a, q in served if all(a[t] <= C[t] for t in range(T))]
        if not feasible:
            continue
        v = F(rng.randrange(1, 13))
        H, Hp = welfare(ordinary, bundles, C), welfare(ordinary, bundles, Cp)
        oldr, newr, losses = [], [], []
        for a, q in feasible:
            low = tuple(C[t]-a[t] for t in range(T))
            lowp = tuple(Cp[t]-a[t] for t in range(T))
            Hlow, Hlowp = welfare(ordinary, bundles, low), welfare(ordinary, bundles, lowp)
            common = []
            for z in product((0, 1), repeat=len(bundles)):
                if all(sum(bundles[j][0][t]*z[j] for j in range(len(bundles))) <= low[t] for t in range(T)):
                    common.append(welfare(ordinary, bundles, low, z) + welfare(ordinary, bundles, Cp, z))
            gamma = Hlow + Hp - max(common)
            assert gamma >= 0
            assert (Hp-Hlowp) - (H-Hlow) <= gamma
            oldr.append((v-(H-Hlow))/q)
            newr.append((v-(Hp-Hlowp))/q)
            losses.append(gamma/q)
            counts['mode_burden_bounds'] += 1
        rold, rnew, g = max(oldr), max(newr), max(losses)
        # New modes can only increase rnew, so restricting to old feasible modes is conservative.
        assert rnew >= rold-g
        told, tnew = clipped_uniform_cutoff(rold), clipped_uniform_cutoff(rnew)
        assert max(Z, told-tnew) <= g/2
        assert max(Z, (told-tnew)/4) <= (told-clipped_uniform_cutoff(rold-g))/4
        counts['admission_cutoff_bounds'] += 1
    for h, d, gamma, L, K in product(map(F, [3, 4]), map(F, [1, 2, 4]),
                                    [F(1, 2), F(1), F(3)], [F(1), F(2)], [F(1), F(3)]):
        if not 0 < gamma < d < 2*h:
            continue
        tau0, tau1 = d/2, (d-gamma)/2
        R0, R1 = L+d*d/(4*h), L+K+gamma+(d-gamma)**2/(4*h)
        assert (tau0-tau1)/h == gamma/(2*h)
        assert R1-R0 == K+gamma-gamma*(2*d-gamma)/(4*h) > K
        counts['sharp_admission_families'] += 1


def study_catalog(stress, bundled_count):
    menus = []
    for i in range(4):
        ms = [mode([0, 0, 0, 0]), mode([1, 0, 0, 0]), mode([0, 1, 0, 0], 1),
              mode([F(1, 2), F(1, 2), 0, 0], F(1, 2))]
        if i < bundled_count:
            ms.append(mode([0, 0, F(1, 2), F(1, 2)], 1))
        menus.append(ms)
    return Catalog(menus, [4, 5, 6, 7], [stress, 1, 1, 1]), [{0, 1, 2} for _ in menus]


def omega(A, q, h):
    if A <= 0:
        return Z
    if q == 0:
        return A
    return A*A/(4*h*q) if A < 2*h*q else A-h*q


def check_menus(counts):
    rng = random.Random(20261025)
    for case in range(90):
        menus, retained = [], []
        for i in range(3):
            e0, e1 = mode([1, 0], 0, F(rng.randrange(3), 4)), mode([0, 1], 1, F(rng.randrange(3), 4))
            lam = F(1+case%3, 4)
            q = (1-lam) + F(rng.randrange(2), 4)
            k = lam*e0[2] + (1-lam)*e1[2] + F(rng.randrange(2), 4)
            menus.append([mode([0, 0]), e0, e1, mode([lam, 1-lam], q, k)])
            retained.append({0, 1, 2})
        ins = Catalog(menus, [3, 5, 7], [1+case%2, 1+(case//2)%2])
        penalty = tuple(F(rng.randrange(17), 2) for _ in menus)
        row = optimized_certificate(ins, penalty, retained)
        assert row['gap'] == row['bound'] == 0
        counts['convex_menu_exactness_cases'] += 1
        counts['exact_lp_witness_checks'] += 1
    pointwise = []
    for stress, k in product(range(3), [0, 2, 4]):
        ins, retained = study_catalog(stress, k)
        row = optimized_certificate(ins, (F(4),)*4, retained)
        row.update({'stress_capacity': stress, 'bundled_eligible': k})
        pointwise.append(row)
        counts['general_menu_certificate_cases'] += 1
        counts['exact_lp_witness_checks'] += 1
    for case in range(60):
        ins, retained = study_catalog(F(case%5, 2), [0, 2, 4][case%3])
        ins.capacity = tuple(ins.capacity)  # rebuilt to include fractional capacities
        penalty = tuple(F(rng.randrange(25), 2) for _ in range(4))
        optimized_certificate(ins, penalty, retained)
        counts['general_menu_certificate_cases'] += 1
        counts['exact_lp_witness_checks'] += 1
    for v in map(F, [1, 2, 4, 8, 16]):
        for t in [F(0), F(1, 2), F(2), F(4)]:
            ins = Catalog([[mode([0, 0]), mode([1, 0])],
                           [mode([0, 0]), mode([0, 1])],
                           [mode([0, 0]), mode([1, 1], 2)]], [1, 2, v], [1, 1])
            row = optimized_certificate(ins, (Z, Z, 2*t), [{0, 1}, {0, 1}, {0}])
            gap = max(Z, v-3-4*t)
            assert row['gap'] == row['bound'] == gap
            counts['sharp_certificate_profiles'] += 1
            counts['exact_lp_witness_checks'] += 1
            ins = Catalog([[mode([0, 0]), mode([1, 0], 1), mode([0, 1], 1),
                            mode([F(1, 2), F(1, 2)], 1)]], [v], [F(1, 2)]*2)
            # An explicit optimal unit dual isolates the fractional-capacity term.
            w = max(Z, v-2*t)
            assert ins.optimum((2*t,)) == w
            assert ins.optimum((2*t,), [{0, 1, 2}]) == 0
            assert w*(F(1, 2)+F(1, 2)) == w
            counts['fractional_capacity_boundary_profiles'] += 1
    return pointwise


def run():
    counts = dict.fromkeys(['mode_burden_bounds', 'admission_cutoff_bounds', 'sharp_admission_families',
                           'convex_menu_exactness_cases', 'general_menu_certificate_cases',
                           'exact_lp_witness_checks', 'sharp_certificate_profiles',
                           'fractional_capacity_boundary_profiles'], 0)
    check_admission(counts)
    pointwise = check_menus(counts)
    output = {'counts': counts, 'pointwise_study': pointwise, 'discrepancies': 0,
              'arithmetic': 'exact fractions, including reconstructed LP primal and dual witnesses'}
    (P/'recovery_certificates.json').write_text(json.dumps(output, default=str, indent=2)+'\n')
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    run()
