"""Independent verification of allocation-preserving sequential charges.

The master manuscript includes both zero-charge and envelope variants.
All calculations are exact Fractions. Early-access values are public and
private scalar costs have the paper's support [0, 4]. The allocation order
and all physical awards are identical to its zero-charge sequential rules.
"""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import json

P = Path(__file__).resolve().parent
D = [1, 2, 1, 2]
V = list(map(F, [4, 7, 5, 8]))
REC = [(1, 0), (0, 1), (F(1, 2), F(1, 2)), (F(1, 2), F(1, 2))]
BAR = F(4)
MENUS = []
for d, (r1, r2) in zip(D, REC):
    modes = [((F(0),) * 3, F(0))]
    for f in [F(0), F(1, 2), F(1)]:
        modes.append(((d * (1 - f), d * f * r1, d * f * r2), d * f))
    MENUS.append(modes)


def allocate(theta, capacity, order):
    residual = list(map(F, capacity))
    z = [0] * 4
    payment = [F(0)] * 4
    for i in order:
        feasible = [m for m, (a, q) in enumerate(MENUS[i])
                    if all(x <= c for x, c in zip(a, residual))]
        chosen = max(feasible, key=lambda m: V[i] * (m != 0) - theta[i] * MENUS[i][m][1])
        z[i] = chosen
        if chosen:
            # At this turn, residual capacity is independent of i's report.
            # Every served mode has the same public gross value. For t > 0,
            # zero-price choice minimizes exposure q among feasible modes.
            # It is served up to min(BAR, V[i]/q), or always when q == 0.
            # The envelope with u_i(BAR)=0 charges max(0,V[i]-BAR*q).
            qmin = min(MENUS[i][m][1] for m in feasible if m != 0)
            payment[i] = max(F(0), V[i] - BAR * qmin)
        residual = [c - a for c, a in zip(residual, MENUS[i][chosen][0])]
    assert all(c >= 0 for c in residual)
    return tuple(z), payment


def main():
    original = json.loads((P / 'policy_grid.json').read_text())
    expected = {(tuple(r['capacity']), tuple(map(F, r['theta'])), r['policy']): tuple(r['modes'])
                for r in original['rows'] if r['policy'] in ['Arrival order', 'Public flexibility order']}
    grid = list(map(F, ['1/2', '3/2', '5/2', '7/2']))
    reports = [F(0), *grid, BAR]
    counts = {'allocation_matches': 0, 'individual_rationality_checks': 0, 'finite_deviation_checks': 0}
    summary = {}
    for name, order in [('Arrival order', [0, 1, 2, 3]), ('Public flexibility order', [1, 3, 0, 2])]:
        total_revenue = F(0)
        total_surplus = F(0)
        nprofiles = 0
        for cap in [(2, 2, 2), (3, 1, 2), (1, 3, 2)]:
            for theta in product(grid, repeat=4):
                z, p = allocate(theta, cap, order)
                assert z == expected[(cap, theta, name)]
                counts['allocation_matches'] += 1
                w = sum(V[i] * (z[i] != 0) - theta[i] * MENUS[i][z[i]][1] for i in range(4))
                for i in range(4):
                    u = V[i] * (z[i] != 0) - theta[i] * MENUS[i][z[i]][1] - p[i]
                    assert u >= 0
                    counts['individual_rationality_checks'] += 1
                    for report in reports:
                        changed = list(theta)
                        changed[i] = report
                        z2, p2 = allocate(changed, cap, order)
                        deviation = V[i] * (z2[i] != 0) - theta[i] * MENUS[i][z2[i]][1] - p2[i]
                        assert deviation <= u
                        counts['finite_deviation_checks'] += 1
                total_revenue += sum(p)
                total_surplus += w
                nprofiles += 1
        summary[name] = {'profiles': nprofiles, 'original_revenue': '0',
                         'repriced_revenue': str(total_revenue / nprofiles),
                         'repriced_revenue_decimal': float(total_revenue / nprofiles),
                         'unchanged_surplus': str(total_surplus / nprofiles),
                         'repriced_rent': str((total_surplus - total_revenue) / nprofiles)}
    result = {'status': 'passed', 'interpretation': 'Exact means on the existing constructed grid; not continuous-prior expected revenues.',
              'scope': 'Same allocations and order; scalar envelope payments with zero highest-type utility.',
              'counts': counts, 'policies': summary}
    (P / 'sequential_pricing_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
