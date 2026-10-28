"""Exact verification of endogenous recovery-permission design.

All inputs are constructed. Integer assignment objectives are checked against
independent physical mode enumeration on separate small cases. Fractions are
used for the analytic prior integrals, sharp families and deletion budgets.
"""
from pathlib import Path
from fractions import Fraction as F
from itertools import product,combinations
from math import comb,lcm
import random,json
import numpy as np
from scipy.optimize import linear_sum_assignment

P=Path(__file__).resolve().parent
Z=F(0)

def ceiling(x):return -((-x.numerator)//x.denominator)

def assignment(base,extra,capacity,mask):
    n=len(base);den=lcm(*(x.denominator for row in base for x in row),*(x.denominator for x in extra))
    T=len(base[0]);W=np.zeros((n,T+capacity+n),dtype=np.int64)
    for i in range(n):
        W[i,:T]=[int(x*den) for x in base[i]]
        W[i,T:T+capacity]=int(extra[i]*den) if mask&(1<<i) else -10**9
    a,b=linear_sum_assignment(W,maximize=True)
    return F(sum(int(W[i,j]) for i,j in zip(a,b)),den)

def multiply(a,b):
    c=[Z]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(b):c[i+j]+=x*y
    return c

def integral(poly,a,b):
    return sum((v*(b**(j+1)-a**(j+1))/F(j+1) for j,v in enumerate(poly)),Z)

def uniform_recovery_gain(b,hs):
    """Integral of 1-product(1-G_i) over [0,b], Z_i uniform [0,2h_i]."""
    if not hs:return Z
    cut=sorted(set([Z,b]+[2*h for h in hs if 0<2*h<b]))
    out=Z
    for a,c in zip(cut,cut[1:]):
        poly=[F(1)]
        for h in hs:
            if a>=2*h:poly=[Z];break
            poly=multiply(poly,[F(1),-F(1)/(2*h)])
        out+=(c-a)-integral(poly,a,c)
    return out

def equal_uniform_gain(b,h,k):
    if not k:return Z
    if b<=2*h:return b-2*h/F(k+1)*(1-(1-b/(2*h))**(k+1))
    return b-2*h/F(k+1)

def check_pool_submodularity(counts):
    rng=random.Random(2026102701)
    for case in range(500):
        n=3+case%4;T=1+case%4;b=1+case%3
        base=[[F(rng.randrange(-3,14)) if rng.randrange(4) else F(-1000) for _ in range(T)] for _ in range(n)]
        extra=[F(rng.randrange(-3,14)) for _ in range(n)]
        val={S:assignment(base,extra,b,S) for S in range(1<<n)}
        for S in range(1<<n):
            absent=[j for j in range(n) if not S&(1<<j)]
            for i in absent:
                assert val[S]<=val[S|(1<<i)];counts['pool_monotonicity']+=1
            for i,j in combinations(absent,2):
                assert val[S|(1<<i)]+val[S|(1<<j)]>=val[S]+val[S|(1<<i)|(1<<j)]
                counts['pool_diminishing_returns']+=1
        for k in range(1,n+1):
            S=0
            for _ in range(k):
                j=max((i for i in range(n) if not S&(1<<i)),key=lambda i:(val[S|(1<<i)]-val[S],-i))
                S|=1<<j
            opt=max(v for m,v in val.items() if m.bit_count()<=k)
            assert val[S]-val[0]>=(1-(1-F(1,k))**k)*(opt-val[0]);counts['greedy_bounds']+=1
    # Independent mode enumeration, with actual nonnegative loads in two
    # base periods and one equal-proportion disjoint recovery block.
    for case in range(36):
        n=3;cap=(case%3,1,1,1);block_capacity=2
        vals=[F(3+i) for i in range(n)];pen=[F(rng.randrange(25),4) for _ in vals]
        qbase=[[F(rng.randrange(5),4) for _ in range(2)] for i in vals]
        qextra=[F(rng.randrange(5),4) for i in vals]
        profiles=[(0,0,0,0),(1,0,0,0),(0,1,0,0),(0,0,F(1,2),F(1,2))]
        base=[[vals[i]-pen[i]*qbase[i][t] for t in range(2)] for i in range(n)]
        slots=[t for t,c in enumerate(cap[:2]) for _ in range(c)]
        baseexpanded=[[row[t] for t in slots] for row in base]
        for S in range(1<<n):
            best=Z
            for modes in product(range(4),repeat=n):
                if any(m==3 and not S&(1<<i) for i,m in enumerate(modes)):continue
                if any(sum(profiles[m][t] for m in modes)>cap[t] for t in range(4)):continue
                v=sum((vals[i]-pen[i]*(qextra[i] if m==3 else qbase[i][m-1]) for i,m in enumerate(modes) if m),Z)
                best=max(best,v)
            assert best==assignment(baseexpanded,[vals[i]-pen[i]*qextra[i] for i in range(n)],block_capacity,S)
            counts['pool_physical_enumeration']+=1

def check_exact_design(counts):
    rng=random.Random(2026102702)
    frontier=[]
    for n in range(2,8):
        values=[F(3+i) for i in range(n)];B=sum(values)-min(values);b=min(values)
        for trial in range(12):
            penalty=[F(rng.randrange(33),4) for _ in values]
            base=[[v]*(n-1) for v in values];extra=[v-p for v,p in zip(values,penalty)]
            for S in range(1<<n):
                analytic=B+max([Z]+[b-penalty[i] for i in range(n) if S&(1<<i)])
                assert analytic==assignment(base,extra,1,S);counts['near_full_formula']+=1
    for b,h in product(map(F,[1,2,3,4,8]),[F(1,2),F(1),F(2)]):
        for k in range(9):
            assert uniform_recovery_gain(b,[h]*k)==equal_uniform_gain(b,h,k);counts['uniform_integral_identities']+=1
    for n in range(2,15):
        hs=[F(j+1,3) for j in range(n)];b=F(2);baseline=F(3*n)
        for k in range(n+1):
            selected=tuple(hs[:k]);value=baseline+uniform_recovery_gain(b,list(selected))
            for alternative in combinations(hs,k):
                assert uniform_recovery_gain(b,list(alternative))<=value-baseline;counts['prior_order_designs']+=1
        # Above-support closed form for minimum permission count.
        h=F(1);b=F(2);A=n*b;RF=A-2*h/F(n+1);baseline=A-b
        for target in [F(1,2),F(9,10),F(19,20),F(99,100)]:
            direct=min(k for k in range(n+1) if baseline+equal_uniform_gain(b,h,k)>=target*RF)
            D=(1-target)*A+2*target*h/F(n+1)
            formula=0 if baseline>=target*RF else max(1,ceiling(2*h/D)-1)
            assert direct==formula;counts['closed_form_minimum_counts']+=1
    n=4;h=F(1);b=F(2);A=F(8);RF=A-2*h/5
    for k in range(5):
        v=A-b+equal_uniform_gain(b,h,k);frontier.append({'permissions':k,'receipts':str(v),'retention':str(v/RF)})
    return frontier

def check_joint_value(counts):
    rng=random.Random(2026102703);rows=[]
    for r in range(2,11):
        h=F(1);V=F(8*r);B=(F(r)-F(1,2))*V
        assert (r-1)*V+2*h<B<r*V-2*r*h
        base=B-h;full=r*(V-h)
        rows.append({'r':r,'h':str(h),'V':str(V),'B':str(B),'proper_menu_receipts':str(base),'full_menu_receipts':str(full),'joint_gain':str(full-base)})
        if r>6:continue
        profile=[(F(1),)+tuple(Z for _ in range(r))]
        for i in range(r):profile.append((F(1,r),)+tuple(F(r-1,r) if j==i else Z for j in range(r)))
        cap=(F(1),)+tuple(F(r-1,r) for _ in range(r))
        feasible=[z for z in product([0,1],repeat=r+1) if all(sum(profile[i][t]*z[i] for i in range(r+1))<=cap[t] for t in range(r+1))]
        for trial in range(8):
            theta=[F(rng.randrange(9),8) for _ in range(r+1)];w=[B-2*theta[0]]+[V-2*t for t in theta[1:]]
            for S in range(1<<r):
                best=max(sum(w[i]*z[i] for i in range(r+1)) for z in feasible if all(not z[i+1] or S&(1<<i) for i in range(r)))
                expected=w[0] if S!=(1<<r)-1 else sum(w[1:])
                assert best==expected;counts['hidden_joint_value_profiles']+=1
    return rows

def check_necessity(counts):
    rng=random.Random(2026102704)
    for case in range(180):
        n=2+case%3;T=2+case%2;C=[F(rng.randrange(1,4)) for _ in range(T)]
        menus=[];weights=[]
        for i in range(n):
            mm=[tuple(Z for _ in range(T))];ww=[Z]
            for j in range(1+case%2):
                mm.append(tuple(F(rng.randrange(3),2) for _ in range(T)));ww.append(F(rng.randrange(-4,25),4))
            menus.append(mm);weights.append(ww)
        alloc=[z for z in product(*[range(len(mm)) for mm in menus]) if all(sum(menus[i][m][t] for i,m in enumerate(z))<=C[t] for t in range(T))]
        scores={z:sum(weights[i][m] for i,m in enumerate(z)) for z in alloc};V=max(scores.values())
        total=Z
        for i,mm in enumerate(menus):
            for m in range(1,len(mm)):
                deleted=max(w for z,w in scores.items() if z[i]!=m);total+=V-deleted;counts['permission_deletion_losses']+=1
        assert total<=V;counts['deletion_budget_catalogs']+=1
    sharp=[]
    for zeta in [F(1,2),F(3,4),F(9,10),F(19,20),F(99,100)]:
        e=1-zeta;d=ceiling(1/e)-1;b=(1/e-d)/2;h=min(b/4,F(1,8));RF=b+d
        assert d+b<1/e and RF-1<zeta*RF
        assert b+h>=2*h and 1+h>=2*h
        sharp.append({'target':str(zeta),'necessary_permissions':d,'positive_unit_baseline':str(b),'h':str(h),'full_receipts':str(RF),'one_deleted_receipts':str(RF-1)})
        counts['sharp_necessity_families']+=1
    large=[]
    for n in [19,20,40,100,200]:
        target=F(19,20);RF=F(n+1);k=max(0,ceiling(target*RF-1))
        assert 1+k>=target*RF and k<target*RF and RF-1>=target*RF
        large.append({'candidate_permissions':n,'minimum_required':k,'global_indispensable_permissions':0,'full_receipts':str(RF),'selected_receipts':str(k+1)})
        counts['large_minimum_menu_families']+=1
    return sharp,large

def check_reversal(counts):
    cases=[]
    for H,h,b in product([F(2),F(3),F(8)],[F(1,2),F(1)],[F(1,2),F(1)]):
        if not H>h or b>2*h:continue
        a=b+2*H+F(1,10)
        smallA=a-H;smallB=uniform_recovery_gain(b,[h])
        expandedA=a+uniform_recovery_gain(b,[H]);expandedB=a+uniform_recovery_gain(b,[h]);expandedFull=a+uniform_recovery_gain(b,[H,h])
        assert smallA>smallB and expandedB>expandedA
        assert (expandedB-a)/(expandedA-a)==H/h
        assert expandedFull==a+b*b/(4*H)+b*b/(4*h)-b**3/(12*H*h)
        counts['capacity_recipient_reversals']+=1
        if (H,h,b)==(F(2),F(1),F(1)):
            cases=[{'firm_capacity':0,'baseline':'0','permission_A':str(smallA),'permission_B':str(smallB),'A_increment':str(smallA),'B_increment':str(smallB)}, {'firm_capacity':1,'baseline':str(a),'permission_A':str(expandedA),'permission_B':str(expandedB),'A_increment':str(expandedA-a),'B_increment':str(expandedB-a),'full_receipts':str(expandedFull)}]
    return cases

def run():
    counts=dict.fromkeys(['pool_monotonicity','pool_diminishing_returns','greedy_bounds','pool_physical_enumeration','near_full_formula','uniform_integral_identities','prior_order_designs','closed_form_minimum_counts','hidden_joint_value_profiles','permission_deletion_losses','deletion_budget_catalogs','sharp_necessity_families','large_minimum_menu_families','capacity_recipient_reversals'],0)
    check_pool_submodularity(counts);frontier=check_exact_design(counts);joint=check_joint_value(counts);sharp,large=check_necessity(counts);reversal=check_reversal(counts)
    result={'status':'passed','discrepancies':0,'all_inputs_constructed':True,'arithmetic':'integer assignment and exact fractions; finite checks supplement proofs','counts':counts,'uniform_design_frontier':frontier,'hidden_joint_value_families':joint,'sharp_global_necessity_families':sharp,'unbounded_local_complexity_families':large,'recipient_reversal_example':reversal}
    (P/'permission_design.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':run()
