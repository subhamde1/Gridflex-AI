"""Fresh enumeration audit, independent of the manuscript's matching routines.

All inputs are constructed. Exact Fraction arithmetic is used throughout.
This checks finite instances, not all possible inputs or literature novelty.
"""
from fractions import Fraction as F
from itertools import product, combinations
from pathlib import Path
import random, json

P=Path(__file__).resolve().parent
Z=F(0)

def add(a,b):return tuple(x+y for x,y in zip(a,b))
def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def fits(a,c):return all(x<=y for x,y in zip(a,c))
def zero(T):return (Z,)*T

def value(menus,c,forced=None):
    """Enumerate complete mode vectors, retaining lexicographic ties."""
    best=Z;winner=(0,)*len(menus)
    for zz in product(*(range(len(ms)) for ms in menus)):
        if forced and any(zz[i]!=m for i,m in forced.items()):continue
        load=tuple(sum(menus[i][m][0][t] for i,m in enumerate(zz)) for t in range(len(c)))
        if fits(load,c):
            w=sum(menus[i][m][1] for i,m in enumerate(zz))
            if w>best:best,winner=w,zz
    return best,winner

def conditional(unit,bundles,c,z):
    load=tuple(sum(bundles[i][m][0][t] for i,m in enumerate(z)) for t in range(len(c)))
    if not fits(load,c):return None
    return sum(bundles[i][m][1] for i,m in enumerate(z))+value(unit,sub(c,load))[0]

def capacity_audit():
    rng=random.Random(20261024)
    counts={'multi_mode_capacity_instances':0,'conditional_four_state_inequalities':0,
            'zero_loss_equivalences':0,'margin_comparisons':0,'positive_exposure_cutoff_bounds':0,
            'zero_exposure_margin_comparisons':0,'negative_or_zero_weight_modes':0}
    for case in range(180):
        T=2+case%2;c=tuple(F(rng.randrange(1,9),4) for _ in range(T))
        if case%2==0:
            a=tuple(F(rng.randrange(int(x)+1)) for x in c)
            delta=tuple(F(rng.randrange(5),4) for _ in c)
        else:
            a=tuple(F(rng.randrange(int(4*x)+1),4) for x in c)
            delta=tuple(F(rng.randrange(3)) for _ in c)
        cp=add(c,delta);states=[c,sub(c,a),cp,sub(cp,a)]
        unit=[]
        for i in range(case%4):
            ms=[(zero(T),Z)]
            for t in range(T):
                if rng.randrange(3):
                    w=F(rng.randrange(-4,17),4)
                    counts['negative_or_zero_weight_modes']+=w<=0
                    ms.append((tuple(F(j==t) for j in range(T)),w))
            unit.append(ms)
        bundles=[]
        for i in range((case//4)%3):
            ms=[(zero(T),Z)]
            for m in range(2):
                cuts=sorted([0,4]+[rng.randrange(5) for _ in range(T-1)])
                load=tuple(F(cuts[t+1]-cuts[t],4) for t in range(T))
                assert sum(load)==1
                w=F(rng.randrange(-4,25),4)
                counts['negative_or_zero_weight_modes']+=w<=0
                ms.append((load,w))
            bundles.append(ms)
        H=[value(unit+bundles,h)[0] for h in states]
        common=[];optimal_both=False
        for zz in product(*(range(len(ms)) for ms in bundles)):
            ff=[conditional(unit,bundles,h,zz) for h in states]
            if ff[1] is None:continue
            assert all(f is not None for f in ff)
            assert ff[2]+ff[1]<=ff[0]+ff[3]
            common.append(ff[1]+ff[2])
            optimal_both |= ff[1]==H[1] and ff[2]==H[2]
            counts['conditional_four_state_inequalities']+=1
        gamma=H[1]+H[2]-max(common)
        rise=(H[2]-H[3])-(H[0]-H[1])
        assert max(Z,rise)<=gamma
        assert (gamma==0)==optimal_both
        counts['zero_loss_equivalences']+=1
        # A one-mode focal applicant, including a zero-exposure mode.
        v=F(7);k=F(case%3,4)
        burden0=k+H[0]-H[1];burden1=k+H[2]-H[3]
        for q in (Z,F(1,2),F(1),F(2)):
            for theta in (Z,F(1,2),F(2),F(4)):
                m0=v-burden0-2*theta*q;m1=v-burden1-2*theta*q
                assert m1>=m0-gamma
                if m0>gamma:assert m1>0
                counts['margin_comparisons']+=1
                counts['zero_exposure_margin_comparisons']+=q==0
            if q:
                tau=lambda b:min(F(4),max(Z,(v-b)/(2*q)))
                assert max(Z,tau(burden0)-tau(burden1))<=gamma/(2*q)
                counts['positive_exposure_cutoff_bounds']+=1
        counts['multi_mode_capacity_instances']+=1
    return counts

def envelope_audit():
    rng=random.Random(442026)
    counts={'tie_and_zero_exposure_envelopes':0,'exact_deviation_comparisons':0,'flat_penalty_monotonicity_comparisons':0}
    for case in range(100):
        # Lines are conditional virtual scores; duplicated modes stress ties.
        lines=[(Z,Z,0)]+[(F(rng.randrange(-4,21),4),F(rng.randrange(9),4),1) for _ in range(4)]
        lines.append(lines[1]);v=F(10)
        points={Z,F(4)}
        for (a,q,x),(b,r,y) in combinations(lines,2):
            if q!=r:
                t=(a-b)/(2*(q-r))
                if 0<t<4:points.add(t)
        points=sorted(points)
        choose=lambda t:max(range(len(lines)),key=lambda m:lines[m][0]-2*t*lines[m][1])
        intervals=[(a,b,lines[choose((a+b)/2)][1]) for a,b in zip(points[:-1],points[1:])]
        def pay(t):
            _,q,x=lines[choose(t)]
            rent=sum(qj*max(Z,b-max(a,t)) for a,b,qj in intervals)
            return v*x-t*q-rent
        tested=sorted(set(points+[(a+b)/2 for a,b in zip(points[:-1],points[1:])]))
        for t in tested:
            _,q,x=lines[choose(t)];u=v*x-t*q-pay(t)
            assert u>=0
            for report in tested:
                _,qr,xr=lines[choose(report)]
                assert v*xr-t*qr-pay(report)<=u
                counts['exact_deviation_comparisons']+=1
        # Fixed tie order when a nondecreasing penalty has flat intervals.
        penalties=[Z,Z,F(1),F(1),F(2),F(4),F(4)]
        qq=[lines[max(range(len(lines)),key=lambda m:lines[m][0]-s*lines[m][1])][1] for s in penalties]
        assert all(a>=b for a,b in zip(qq,qq[1:]))
        counts['flat_penalty_monotonicity_comparisons']+=len(qq)-1
        counts['tie_and_zero_exposure_envelopes']+=1
    return counts

if __name__=='__main__':
    output={'status':'passed','arithmetic':'exact rational enumeration','counts':capacity_audit()|envelope_audit(),'discrepancies':0,
            'independence':'No imports from existing allocation, matching, payment, or certificate implementations.',
            'scope':'Finite computational audit; no external human review or formal theorem-prover certification.'}
    (P/'final_adversarial_audit.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))
