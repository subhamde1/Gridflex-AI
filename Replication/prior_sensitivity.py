"""Deterministic rational bounds on continuous-prior expected receipts.

Conditional Jensen bounds and coordinatewise convex chords bracket the
expectation. This is not Monte Carlo or a point estimate from a type grid.
Every arithmetic comparison and every reported bound uses integers or
fractions. No empirical data are imported.
"""
from fractions import Fraction as F
from itertools import product
from math import lcm
from pathlib import Path
import json

import numpy as np

from recovery_certificates import study_catalog, omega

P = Path(__file__).resolve().parent


def prior(beta, m, h=F(4)):
    beta = F(beta)
    u = [F(j, m) for j in range(m+1)]
    t = [h*x/(beta+(1-beta)*x) for x in u]
    s = [2*x+(beta-1)*x*x/h for x in t]
    means = [m*(u[j+1]*t[j+1]-u[j]*t[j]) for j in range(m)]
    weights = [F(0)]*(m+1)
    for j in range(m):
        assert s[j] <= means[j] <= s[j+1]
        p = (means[j]-s[j])/(s[j+1]-s[j])
        weights[j] += (1-p)/m
        weights[j+1] += p/m
    assert sum(weights) == 1
    return s, means, weights


def values_on_grid(ins, points, retained=None):
    """Return an exact integer tensor and its common denominator.

    An explicit worst-case bound prevents int64 overflow before matrix
    multiplication. If a finer grid breaches it the program stops rather
    than silently losing exactness.
    """
    D = lcm(*(x.denominator for x in points))
    pp = np.asarray([int(x*D) for x in points], dtype=np.int64)
    aa = [z for z in ins.allocations if retained is None or
          all(z[i] in retained[i] for i in range(ins.n))]
    intercept = [sum(ins.values[i]*(m != 0)-ins.menus[i][m][2] for i, m in enumerate(z)) for z in aa]
    q = [[ins.menus[i][m][1] for i, m in enumerate(z)] for z in aa]
    assert all((x*4).denominator == 1 for row in q for x in row)
    assert all((x*4).denominator == 1 for x in intercept)
    qi = np.asarray([[int(x*4) for x in row] for row in q], dtype=np.int64)
    a0 = np.asarray([int(x*4)*D for x in intercept], dtype=np.int64)
    worst = int(max(abs(x) for x in a0)) + ins.n*int(qi.max())*max(abs(int(x)) for x in pp)
    assert worst < np.iinfo(np.int64).max, ('Choose a coarser grid', D, worst)
    allpoints = np.asarray(list(product(pp, repeat=ins.n)), dtype=np.int64)
    out = np.empty(len(allpoints), dtype=np.int64)
    for start in range(0, len(allpoints), 512):
        x = allpoints[start:start+512]
        out[start:start+len(x)] = (a0[None, :] - x @ qi.T).max(axis=1)
    assert out.min() >= 0
    return out.reshape((len(points),)*ins.n), 4*D


def expectation_interval(ins, beta, m=12, retained=None):
    endpoints, means, weights = prior(beta, m)
    low, denominator = values_on_grid(ins, means, retained)
    lower = F(sum(map(int, low.flat)), denominator*m**ins.n)
    high, denominator = values_on_grid(ins, endpoints, retained)
    W = lcm(*(x.denominator for x in weights))
    wi = np.asarray([int(x*W) for x in weights], dtype=object)
    contracted = high.astype(object)
    for _ in range(ins.n):
        contracted = np.tensordot(wi, contracted, axes=(0, 0))
    upper = F(int(contracted), denominator*W**ins.n)
    assert lower <= upper
    return lower, upper


def checks():
    from recovery_certificates import Catalog, mode
    cases = 0
    for v, q, h in product(map(F, [1, 4, 8, 12]), map(F, [1, 2]), [F(4)]):
        ins = Catalog([[mode([0]), mode([1], q)]], [v], [1])
        exact = omega(v, q, h)
        previous = None
        for m in (2, 4, 8, 16):
            lo, hi = expectation_interval(ins, F(1), m)
            assert lo <= exact <= hi
            if previous is not None:
                assert lo >= previous[0] and hi <= previous[1]
            previous = lo, hi
            cases += 1
    # Nonuniform conditional moments, evaluated independently by endpoint theta*F(theta).
    for beta in [F(1, 2), F(1), F(2)]:
        endpoints, means, weights = prior(beta, 12)
        for j in range(12):
            a, b = F(j, 12), F(j+1, 12)
            ta = 4*a/(beta+(1-beta)*a)
            tb = 4*b/(beta+(1-beta)*b)
            assert means[j] == (tb*b-ta*a)/(b-a)
            cases += 1
    return cases


def outward(x, places, upward=False):
    d = 10**places
    y = x*d
    k = -((-y.numerator)//y.denominator) if upward else y.numerator//y.denominator
    sign = '-' if k < 0 else ''
    k = abs(k)
    return f'{sign}{k//d}.{k%d:0{places}d}'


def run():
    tests = checks()
    rows = []
    for beta, stress, k in product([F(1, 2), F(1), F(2)], range(3), [0, 2, 4]):
        ins, retained = study_catalog(stress, k)
        full = expectation_interval(ins, beta)
        unit = expectation_interval(ins, beta, retained=retained)
        if k == 0:
            assert full == unit
            gap = F(0), F(0)
        else:
            gap = max(F(0), full[0]-unit[1]), full[1]-unit[0]
        row = {'beta': beta, 'stress_capacity': stress, 'bundled_eligible': k,
               'full_receipts': full, 'unit_receipts': unit, 'receipt_loss': gap,
               'guaranteed_retention': F(1) if k == 0 else unit[0]/full[1],
               'display': {'full': [outward(full[0], 4), outward(full[1], 4, True)],
                           'unit': [outward(unit[0], 4), outward(unit[1], 4, True)],
                           'loss': [outward(gap[0], 4), outward(gap[1], 4, True)],
                           'retention': outward(F(1) if k == 0 else unit[0]/full[1], 4)}}
        rows.append(row)
        print(beta, stress, k, row['display'], flush=True)
    output = {'partitions_per_dimension': 12, 'center_points_per_menu': 12**4,
              'endpoint_points_per_menu': 13**4, 'scenarios': len(rows),
              'arithmetic': 'rational conditional means and exact integer optimization; outward decimal rounding',
              'one_dimensional_checks': tests, 'rows': rows, 'discrepancies': 0}
    (P/'prior_sensitivity.json').write_text(json.dumps(output, default=str, indent=2)+'\n')


if __name__ == '__main__':
    run()
