"""Construct public intermediate menus on the unchanged 27-setting block study.

No values, priors, physical capacities, or performance targets are changed.
Half-firm modes are removed by the manuscript's convexification result.
The search enumerates subsets of applicant-specific paired-recovery modes.
Integration uses the supplied exact quantile-cell method, with directed dyadic
rounding when refinement is required. No sampled type average certifies a row.
"""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
import sys,json,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from recovery_certificates import study_catalog
from prior_sensitivity import expectation_interval,prior,outward
from menu_standardization import rounded_scores,D

TARGET=F(19,20)
OLD=json.loads((ROOT/'menu_standardization.json').read_text())

def compressed_allocations(stress,mask):
    opts=[[0,1,2]+([3] if mask&(1<<i) else []) for i in range(4)]
    return [z for z in product(*opts) if z.count(1)<=stress and z.count(2)<=1 and z.count(3)<=2]

def rounded_values(stress,mask,points,upward):
    pp=rounded_scores(points,upward)
    aa=compressed_allocations(stress,mask)
    intercept=np.array([sum((4+i)*D for i,m in enumerate(z) if m) for z in aa],dtype=np.int64)
    exposure=np.array([[int(m in (2,3)) for m in z] for z in aa],dtype=np.int64)
    shape=(len(points),)*4;n=len(points)**4
    assert int(intercept.max())+4*int(np.max(np.abs(pp)))<np.iinfo(np.int64).max
    out=np.empty(n,dtype=np.int64)
    for start in range(0,n,4096):
        ix=np.arange(start,min(start+4096,n));coords=np.array(np.unravel_index(ix,shape)).T
        out[start:start+len(ix)]=(intercept[None,:]-pp[coords]@exposure.T).max(axis=1)
    return out.reshape(shape)

def refined_interval(beta,stress,mask,m):
    endpoints,means,weights=prior(beta,m)
    low=rounded_values(stress,mask,means,True)
    lo=F(sum(map(int,low.flat)),D*m**4)
    high=rounded_values(stress,mask,endpoints,False)
    from math import lcm
    W=lcm(*(p.denominator for p in weights));wi=np.array([int(p*W) for p in weights],dtype=object)
    x=high.astype(object)
    for _ in range(4):x=np.tensordot(wi,x,axes=(0,0))
    hi=F(int(x),D*W**4)
    assert lo<=hi
    return lo,hi

cache={}
def evaluate(beta,stress,k,mask,m=12):
    key=(str(beta),stress,mask,m)
    if key in cache:return cache[key]
    if m==12:
        ins,_=study_catalog(stress,4)
        retained=[{0,1,2}|({4} if mask&(1<<i) else set()) for i in range(4)]
        ans=expectation_interval(ins,beta,m,retained=retained)
    else:ans=refined_interval(beta,stress,mask,m)
    cache[key]=ans
    return ans

def ids(mask):return [i+1 for i in range(4) if mask&(1<<i)]

def run():
    start=time.monotonic();rows=[];evaluated=0
    for old in OLD['rows']:
        beta=F(old['beta']);stress=old['stress_capacity'];k=old['bundled_eligible']
        fullmask=(1<<k)-1;fl,fu=map(F,old['full_lp_expectation'])
        bounds={mask:evaluate(beta,stress,k,mask) for mask in range(1<<k)}
        # Full objective equality follows analytically from convexification;
        # intersect independent enclosures for its equivalent base-only menu.
        fl=max(fl,bounds[fullmask][0]);fu=min(fu,bounds[fullmask][1]);assert fl<=fu
        partitions=[12]
        while True:
            passed=[mask for mask,(lo,hi) in bounds.items() if lo>=TARGET*fu]
            failed=[mask for mask,(lo,hi) in bounds.items() if hi<TARGET*fl]
            known=passed+failed
            smallest=min((m.bit_count() for m in passed),default=99)
            uncertain=[m for m in bounds if m not in known and m.bit_count()<=smallest]
            if not uncertain:break
            nxt=24 if partitions[-1]==12 else 48
            if nxt>48:break
            partitions.append(nxt)
            for mask in set(uncertain+[fullmask]):
                lo,hi=refined_interval(beta,stress,mask,nxt)
                bounds[mask]=(max(bounds[mask][0],lo),min(bounds[mask][1],hi));assert bounds[mask][0]<=bounds[mask][1]
            fl=max(fl,bounds[fullmask][0]);fu=min(fu,bounds[fullmask][1]);assert fl<=fu
        passed=[m for m,(lo,hi) in bounds.items() if lo>=TARGET*fu]
        minsize=min(m.bit_count() for m in passed)
        best=[m for m in passed if m.bit_count()==minsize]
        smaller=[m for m in bounds if m.bit_count()<minsize]
        minimal_proved=all(bounds[m][1]<TARGET*fl for m in smaller)
        selected=min(best,key=lambda m:(-bounds[m][0],ids(m)))
        deletions=[selected&~(1<<i) for i in range(4) if selected&(1<<i)]
        necessities=all(bounds[m][1]<TARGET*fl for m in deletions)
        lo,hi=bounds[selected]
        retention=(F(1),F(1)) if selected==fullmask else (lo/fu,min(F(1),hi/fl))
        global_core=[i+1 for i in range(k) if bounds[fullmask&~(1<<i)][1]<TARGET*fl]
        row={'beta':str(beta),'stress_capacity':stress,'original_bundled_eligible':k,'original_unit_decision':old['decision'],'target':str(TARGET),'full_receipts':[str(fl),str(fu)],'minimum_paired_option_count':minsize,'minimum_cardinality_proved':minimal_proved,'minimum_cardinality_sets':[ids(m) for m in best],'selected_applicants':ids(selected),'selected_receipts':[str(lo),str(hi)],'selected_retention_interval':list(map(str,retention)),'globally_indispensable_in_candidate_class':global_core,'all_subsets_resolved':all(a>=TARGET*fu or b<TARGET*fl for a,b in bounds.values()),'each_selected_option_necessary':necessities,'largest_single_deletion_retention_upper':str(max((bounds[m][1]/fl for m in deletions),default=F(0))),'partitions_used':partitions,'all_subsets':[{'applicants':ids(mask),'receipts':[str(a),str(b)],'retention_lower':str(a/fu),'retention_upper':str(min(F(1),b/fl)),'decision':'pass' if a>=TARGET*fu else 'fail' if b<TARGET*fl else 'unresolved'} for mask,(a,b) in bounds.items()]}
        row['display']={'retention_percent':[outward(100*retention[0],2),outward(100*retention[1],2,True)],'worst_deletion_upper_percent':outward(100*max((bounds[m][1]/fl for m in deletions),default=F(0)),2,True)}
        rows.append(row);print(beta,stress,k,'options',minsize,ids(selected),'retention',row['display'],'minimum proved',minimal_proved,'necessary',necessities,flush=True)
        out={'status':'complete' if len(rows)==27 else 'running','all_original_inputs_preserved':True,'target':str(TARGET),'search_space':'subsets of existing applicant-specific paired-recovery options; all unit modes retained; half-firm modes removed by convexification','rows':rows,'seconds':round(time.monotonic()-start,2)}
        (ROOT/'richer_menu_results.json').write_text(json.dumps(out,indent=2))
    assert all(r['minimum_cardinality_proved'] and r['each_selected_option_necessary'] for r in rows)
    print('COMPLETE',len(rows),'seconds',round(time.monotonic()-start,2),flush=True)

if __name__=='__main__':run()
