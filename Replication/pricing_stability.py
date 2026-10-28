"""Constructed checks and benchmarks for capacity-price stability.

python pricing_stability.py verify
python pricing_stability.py benchmark

No company data are read. Loads and exposures use integer quarter units;
reported welfare uses integer quarter-money units. The mixed optimizer uses
optional assignment. Independent mode enumeration checks the small cases,
and a separate binary MILP supplies bounds for the larger comparisons.
The benchmark times welfare/exclusion kernels, not lexicographic allocation
reconstruction, certification, or operating latency.
"""
import argparse
import hashlib
import itertools
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp

P = Path(__file__).resolve().parent
S = 4


class Generator:
    def __init__(self, seed):
        self.z = int(seed)

    def draw(self, n):
        self.z = (1664525 * self.z + 1013904223) % 2**32
        return self.z * n // 2**32


def agent(unit, b, c, modes):
    return {"unit": bool(unit), "b": int(b), "c": int(c),
            "modes": [(np.asarray(a, dtype=np.int64), int(q)) for a, q in modes]}


def value(person, m):
    return 0 if m == 0 else S * person["b"] - person["c"] * person["modes"][m][1]


def make_instance(seed, u, k, periods, catalog, small=False):
    g = Generator(seed)
    aa = []
    for i in range(u + k):
        b, c = (2 + g.draw(25), g.draw(7)) if small else (20 + g.draw(81), g.draw(4))
        modes = [(np.zeros(periods, dtype=int), 0)]
        if i < u:
            if small:
                eligible = [t for t in range(periods) if g.draw(4) != 0]
            else:
                start, width = g.draw(periods), 1 + g.draw(min(4, periods))
                eligible = sorted({(start + j) % periods for j in range(width)})
            for t in eligible:
                a = np.zeros(periods, dtype=int)
                a[t] = S
                modes.append((a, S * (t + 1)))
        else:
            for _ in range(catalog - 1):
                a = np.zeros(periods, dtype=int)
                t = g.draw(periods)
                other = (t + 1 + g.draw(periods - 1)) % periods
                a[t] += 1
                a[other] += 1
                for _ in range(S - 2):
                    a[g.draw(periods)] += 1
                modes.append((a, int(np.arange(1, periods + 1) @ a)))
        aa.append(agent(i < u, b, c, modes))
    if small:
        cap = np.array([1 + g.draw(12) for _ in range(periods)], dtype=int)
    else:
        base = max(1, (u + k) // (2 * periods))
        cap = S * np.array([base + g.draw(2) for _ in range(periods)], dtype=int)
    return aa, cap


class Mixed:
    def __init__(self, aa, periods):
        self.aa, self.T = aa, periods
        self.units = [i for i, a in enumerate(aa) if a["unit"]]
        self.bundles = [i for i, a in enumerate(aa) if not a["unit"]]
        self.weight = np.full((len(aa), periods), -10**8, dtype=np.int64)
        for i in self.units:
            for m, (a, _) in enumerate(aa[i]["modes"][1:], 1):
                assert a.sum() == S and np.count_nonzero(a) == 1
                self.weight[i, int(a.argmax())] = value(aa[i], m)
        self.cache = {}
        self.portfolio_cache = {}
        self.assignment_calls = 0
        self.portfolio_visits = 0

    def matching(self, ids, cap):
        cap = np.asarray(cap, dtype=int)
        assert np.all(cap >= 0)
        if not ids:
            return 0
        key = (tuple(ids), tuple(int(x) for x in cap))
        if key in self.cache:
            return self.cache[key]
        slots = np.repeat(np.arange(self.T), np.minimum(cap, len(ids)))
        score = np.zeros((len(ids), len(slots) + len(ids)), dtype=np.int64)
        if len(slots):
            score[:, :len(slots)] = self.weight[np.ix_(ids, slots)]
        rows, cols = linear_sum_assignment(score, maximize=True)
        ans = int(score[rows, cols].sum())
        assert np.all(score[rows, cols] > -10**8)
        self.assignment_calls += 1
        self.cache[key] = ans
        return ans

    def portfolios(self, exclude):
        key = exclude if exclude in self.bundles else None
        if key in self.portfolio_cache:
            return self.portfolio_cache[key]
        bundles = [i for i in self.bundles if i != exclude]
        out = []
        for modes in itertools.product(*(range(len(self.aa[i]["modes"])) for i in bundles)):
            load = np.zeros(self.T, dtype=np.int64)
            val = 0
            for i, m in zip(bundles, modes):
                load += self.aa[i]["modes"][m][0]
                val += value(self.aa[i], m)
            out.append((tuple(modes), load, int(val)))
        self.portfolio_cache[key] = out
        return out

    def evaluate(self, cap, exclude=None, records=False):
        cap = np.asarray(cap, dtype=np.int64)
        assert np.all(cap >= 0)
        ids = [i for i in self.units if i != exclude]
        result = {}
        best = 0
        for modes, load, fixed in self.portfolios(exclude):
            self.portfolio_visits += 1
            if np.any(load > cap):
                continue
            ordinary = self.matching(ids, (cap - load) // S)
            total = fixed + ordinary
            best = max(best, total)
            if records:
                result[modes] = {"value": total, "load": load,
                                 "fixed": fixed, "ordinary": ordinary}
        return (int(best), result) if records else int(best)


class Direct:
    """Independent enumeration: no matching or portfolio decomposition."""
    def __init__(self, aa):
        self.aa = aa
        self.cache = {}

    def evaluate(self, cap, exclude=None):
        if exclude not in self.cache:
            ids = [i for i in range(len(self.aa)) if i != exclude]
            modes = list(itertools.product(*(range(len(self.aa[i]["modes"])) for i in ids)))
            loads = np.zeros((len(modes), len(cap)), dtype=np.int64)
            vals = np.zeros(len(modes), dtype=np.int64)
            for row, mm in enumerate(modes):
                for i, m in zip(ids, mm):
                    loads[row] += self.aa[i]["modes"][m][0]
                    vals[row] += value(self.aa[i], m)
            self.cache[exclude] = (loads, vals)
        loads, vals = self.cache[exclude]
        feasible = np.all(loads <= cap, axis=1)
        return int(vals[feasible].max())


def stability(engine, cap, expanded, a, exclude=None):
    w00, f00 = engine.evaluate(cap - a, exclude, records=True)
    w10, f10 = engine.evaluate(cap, exclude, records=True)
    w01, f01 = engine.evaluate(expanded - a, exclude, records=True)
    w11, f11 = engine.evaluate(expanded, exclude, records=True)
    shared = max(f["value"] + f11[z]["value"] for z, f in f00.items())
    gamma = w00 + w11 - shared
    change = (w11 - w01) - (w10 - w00)
    branch_checks = 0
    for z, f in f00.items():
        assert f11[z]["value"] + f["value"] <= f10[z]["value"] + f01[z]["value"]
        # Any feasible same-portfolio pair gives an upper certificate.
        cert = w00 + w11 - f["value"] - f11[z]["value"]
        assert max(0, change) <= gamma <= cert
        branch_checks += 1
    common = any(f["value"] == w00 and f11[z]["value"] == w11 for z, f in f00.items())
    assert (gamma == 0) == common
    return {"welfare4": [w00, w10, w01, w11], "old_price4": w10 - w00,
            "new_price4": w11 - w01, "change4": change, "gamma4": gamma,
            "common_optimum": common, "branch_checks": branch_checks}


def write_json(name, obj):
    path = P / name
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def environment():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "platform": platform.platform(),
            "processor": platform.processor(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def verify():
    counts = {"instances": 0, "four_state_comparisons": 0,
              "independent_welfare_comparisons": 0, "conditional_inequalities": 0,
              "zero_loss_common_optimum_checks": 0, "integer_profile_comparisons": 0,
              "integer_expansion_comparisons": 0, "positive_price_changes": 0,
              "pure_matching_comparisons": 0, "sharp_family_checks": 0,
              "unchanged_award_family_checks": 0, "fractional_boundary_checks": 0}
    rows = []
    for j in range(360):
        periods, u, k = 2 + j % 2, (j // 2) % 5, (j // 10) % 3
        aa, cap = make_instance(20261020 + 7919 * j, u, k, periods, 3, small=True)
        engine, direct = Mixed(aa, periods), Direct(aa)
        counts["instances"] += 1
        for i, person in enumerate(aa):
            for m, (a, _) in enumerate(person["modes"][1:], 1):
                if np.any(a > cap):
                    continue
                # Whole-quantum expansion permits arbitrary focal profiles.
                delta = np.zeros(periods, dtype=int)
                delta[(i + m + j) % periods] = S
                if j % 3 == 0:
                    delta[(i + m + j + 1) % periods] += S
                comparisons = [(delta, "integer_expansion_comparisons")]
                if np.all(a % S == 0):
                    # A fractional expansion is valid for an integer focal mode.
                    comparisons.append((delta // S, "integer_profile_comparisons"))
                for addition, key in comparisons:
                    ans = stability(engine, cap, cap + addition, a, exclude=i)
                    hs = [cap - a, cap, cap + addition - a, cap + addition]
                    brute = [direct.evaluate(h, exclude=i) for h in hs]
                    assert ans["welfare4"] == brute
                    assert ans["gamma4"] >= max(0, ans["change4"])
                    counts[key] += 1
                    counts["four_state_comparisons"] += 1
                    counts["independent_welfare_comparisons"] += 4
                    counts["conditional_inequalities"] += ans["branch_checks"]
                    counts["zero_loss_common_optimum_checks"] += 1
                    counts["positive_price_changes"] += int(ans["change4"] > 0)
                    if not [b for b in engine.bundles if b != i]:
                        assert ans["gamma4"] == 0 and ans["change4"] <= 0
                        counts["pure_matching_comparisons"] += 1
                    rows.append({"case": j, "focal": i, "mode": m,
                                 "capacity4": cap.tolist(), "delta4": addition.tolist(),
                                 "a4": a.tolist(), **ans})

    examples = []
    for l, kval, gamma in itertools.product([1, 2, 5], [1, 3, 7], [1, 2, 8, 32, 128]):
        h = l + kval + gamma
        aa = [agent(True, l, 0, [(np.zeros(2, dtype=int), 0), ([4, 0], 0)]),
              agent(True, kval, 0, [(np.zeros(2, dtype=int), 0), ([0, 4], 0)]),
              agent(False, h, 0, [(np.zeros(2, dtype=int), 0), ([2, 2], 0)])]
        eng, d = Mixed(aa, 2), Direct(aa)
        ans = stability(eng, np.array([4, 0]), np.array([4, 4]), np.array([4, 0]))
        assert ans["old_price4"] == 4*l and ans["change4"] == ans["gamma4"] == 4*gamma
        assert ans["welfare4"] == [d.evaluate(np.array(c)) for c in [[0,0],[4,0],[0,4],[4,4]]]
        counts["sharp_family_checks"] += 1
        examples.append({"kind": "sharp", "L": l, "K": kval, "gamma": gamma, **ans})

    for l, pval, qval, eta in itertools.product([1, 3], [4, 16, 64], [1, 2], [1, 2]):
        h = pval + l + eta
        aa = [agent(True, l, 0, [(np.zeros(2, dtype=int), 0), ([4, 0], 0)]),
              agent(True, pval, 0, [(np.zeros(2, dtype=int), 0), ([0, 4], 0)]),
              agent(True, qval, 0, [(np.zeros(2, dtype=int), 0), ([0, 4], 0)]),
              agent(False, h, 0, [(np.zeros(2, dtype=int), 0), ([2, 2], 0)])]
        eng, d = Mixed(aa, 2), Direct(aa)
        ans = stability(eng, np.array([4,4]), np.array([4,8]), np.array([4,0]))
        assert ans["old_price4"] == 4*(h-pval) and ans["new_price4"] == 4*(h-qval)
        assert ans["change4"] == 4*(pval-qval)
        assert ans["gamma4"] - ans["change4"] == 4*eta
        assert ans["welfare4"] == [d.evaluate(np.array(c)) for c in [[0,4],[4,4],[0,8],[4,8]]]
        # Add a focal applicant whose net value ensures the same award twice.
        focal = agent(True, h-qval+1, 0, [(np.zeros(2,dtype=int),0),([4,0],0)])
        for cap, expected in [(np.array([4,4]),h-pval),(np.array([4,8]),h-qval)]:
            all_direct = Direct(aa+[focal])
            full = all_direct.evaluate(cap)
            wo = all_direct.evaluate(cap, exclude=4)
            payment = wo-full+4*focal["b"]
            assert payment == 4*expected and full>wo
        counts["unchanged_award_family_checks"] += 1
        examples.append({"kind":"unchanged_award","L":l,"P":pval,"Q":qval,"eta":eta,**ans})

    for l in [1, 2, 5, 20, 100]:
        aa = [agent(True,l,0,[(np.zeros(2,dtype=int),0),([4,0],0)])]
        eng = Mixed(aa,2)
        a, c, cp = np.array([1,3]), np.array([3,4]), np.array([4,4])
        old = eng.evaluate(c)-eng.evaluate(c-a)
        new = eng.evaluate(cp)-eng.evaluate(cp-a)
        assert old == 0 and new == 4*l
        counts["fractional_boundary_checks"] += 1
    out={"status":"passed", "errors":0, "counts":counts,
         "environment":environment(), "rows":rows, "examples":examples}
    write_json("pricing_stability_verification.json",out)
    print(json.dumps({"status":"passed","errors":0,"counts":counts}),flush=True)


def binary_milp(aa, cap, exclude=None, limit=10.0):
    ids=[i for i in range(len(aa)) if i!=exclude]
    variables=[(i,m) for i in ids for m in range(1,len(aa[i]["modes"]))]
    if not variables:
        return {"lower4":0,"upper4":0,"closed":True,"status":0,"time":0.0}
    index={i:j for j,i in enumerate(ids)}
    A=np.zeros((len(ids)+len(cap),len(variables)))
    obj=np.zeros(len(variables),dtype=int)
    for col,(i,m) in enumerate(variables):
        A[index[i],col]=1
        A[len(ids):,col]=aa[i]["modes"][m][0]
        obj[col]=value(aa[i],m)
    upper=np.r_[np.ones(len(ids)),cap]
    start=time.perf_counter()
    result=milp(c=-obj,integrality=np.ones(len(variables)),bounds=Bounds(0,1),
                constraints=LinearConstraint(A,-np.inf,upper),
                options={"time_limit":limit,"mip_rel_gap":0.0,"presolve":True,
                         "threads":1,"random_seed":0})
    elapsed=time.perf_counter()-start
    lower=0
    if result.x is not None:
        x=np.rint(result.x).astype(int)
        assert np.max(np.abs(x-result.x))<1e-5
        assert np.all(A@x<=upper+1e-6)
        lower=int(obj@x)
    dual=getattr(result,"mip_dual_bound",None)
    ub=int(np.ceil(-dual-1e-7)) if dual is not None and np.isfinite(dual) else sum(max(0,max(value(aa[i],m) for m in range(len(aa[i]["modes"])))) for i in ids)
    return {"lower4":lower,"upper4":max(lower,ub),"closed":lower==ub,
            "status":int(result.status),"time":elapsed}


def pruning(engine, cap, exclude, a):
    wo, records=engine.evaluate(cap,exclude,records=True)
    feasible=[(wo-r["value"],z,r) for z,r in records.items() if np.all(r["load"]<=cap-a)]
    feasible.sort(key=lambda x:(x[0],x[1]))
    ids=[i for i in engine.units if i!=exclude]
    best=None
    used=0
    for gap,z,r in feasible:
        if best is not None and gap>best:
            continue
        remaining=engine.matching(ids,(cap-a-r["load"])//S)
        term=gap+r["ordinary"]-remaining
        best=term if best is None else min(best,term)
        used+=1
    exact=wo-engine.evaluate(cap-a,exclude)
    assert best==exact
    return {"feasible_portfolios":len(feasible),"residual_evaluations":used,
            "omitted":len(feasible)-used,"exact_price4":exact}


def benchmark():
    cells=[(u,k,3) for u in [8,32,128] for k in [0,1,2,4]]
    cells.extend([(32,2,2),(32,2,4),(32,6,3)])
    rows=[]
    for cell,(u,k,j) in enumerate(cells):
        periods=4 if u==8 else (8 if u==32 else 16)
        for rep in range(2):
            seed=20261021+1009*cell+104729*rep
            aa,cap=make_instance(seed,u,k,periods,j)
            start=time.perf_counter()
            eng=Mixed(aa,periods)
            full=eng.evaluate(cap)
            excl=[eng.evaluate(cap,exclude=i) for i in range(len(aa))]
            exact_time=time.perf_counter()-start
            oracle_calls=eng.assignment_calls
            visits=eng.portfolio_visits
            start=time.perf_counter()
            mm=[binary_milp(aa,cap,exclude=exc) for exc in [None]+list(range(len(aa)))]
            mip_time=time.perf_counter()-start
            exacts=[full]+excl
            for val,bound in zip(exacts,mm):
                assert bound["lower4"]<=val<=bound["upper4"]
                if bound["closed"]:assert bound["lower4"]==val
            i=0
            a=aa[i]["modes"][1][0]
            prune=pruning(eng,cap,i,a)
            delta=np.zeros(periods,dtype=int)
            delta[(cell+rep)%periods]=S
            stab=stability(eng,cap,cap+delta,a,exclude=i)
            row={"cell":cell,"replicate":rep,"seed":seed,"unit_applicants":u,
                 "bundled_applicants":k,"n":u+k,"periods":periods,
                 "catalog_including_rejection":j,"Q":j**k,
                 "capacity4":cap.tolist(),"full_welfare4":full,
                 "exclusion_welfare4":excl,"mixed_seconds":exact_time,
                 "milp_batch_seconds":mip_time,"milp_solver_seconds":sum(b["time"] for b in mm),
                 "matching_calls":oracle_calls,"portfolio_visits":visits,
                 "milp_calls":len(mm),"milp_closed":sum(b["closed"] for b in mm),
                 "milp_bounds":mm,"pruning":prune,"stability":stab}
            rows.append(row)
            write_json("mixed_scaling_benchmark.json",{"status":"running","environment":environment(),"rows":rows})
            print(json.dumps({"case":len(rows),"u":u,"k":k,"J":j,
                  "mixed_seconds":round(exact_time,4),"milp_seconds":round(mip_time,4),
                  "closed":row["milp_closed"],"calls":len(mm)}),flush=True)
    summary={"instances":len(rows),"welfare_endpoints":sum(r["milp_calls"] for r in rows),
             "closed_milp_endpoints":sum(r["milp_closed"] for r in rows),
             "interval_violations":0,"closed_endpoint_discrepancies":0,
             "pruning_queries":len(rows),"stability_queries":len(rows),
             "max_n":max(r["n"] for r in rows),"max_Q":max(r["Q"] for r in rows)}
    out={"status":"passed","errors":0,"environment":environment(),"summary":summary,"rows":rows}
    write_json("mixed_scaling_benchmark.json",out)
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["verify","benchmark"])
    args=parser.parse_args()
    verify() if args.action=="verify" else benchmark()
