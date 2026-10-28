"""Independent exact-rational checks of scalar screening and efficient pricing.

All inputs are mathematical constructions. No external data are imported.
Run this file from any directory. Results and complete finite policy rows are
written beside this script; all transfers use exact Fraction arithmetic.
"""
from fractions import Fraction as F
from itertools import product, combinations
from pathlib import Path
from functools import lru_cache
import json, hashlib, random, platform

P=Path(__file__).resolve().parent
ZERO=F(0)
BAR=F(4)
COUNTS={k:0 for k in ['screening_cases','scalar_deviation_comparisons','conditional_revenue_identities','distortion_bounds','vcg_deviation_comparisons','vcg_menu_checks','energy_checks','policy_profile_comparisons','two_applicant_checks','ranking_constructions','scenario_checks','allocation_reversal_constructions','displayed_example_checks']}

def encode(x):
 if isinstance(x,F): return str(x)
 raise TypeError(type(x).__name__)

class Finite:
 def __init__(self,menus,values,cap,costs=None):
  self.menus=menus;self.values=tuple(map(F,values));self.cap=tuple(map(F,cap));self.n=len(menus)
  self.costs=costs or [tuple(ZERO for _ in ms) for ms in menus]
  self.alloc=[]
  for z in product(*(range(len(ms)) for ms in menus)):
   if all(sum(menus[i][z[i]][0][t] for i in range(self.n))<=self.cap[t] for t in range(len(cap))):
    self.alloc.append(tuple(z))
  assert self.alloc and self.alloc[0]==(0,)*self.n
  self.q={z:tuple(self.menus[i][z[i]][1] for i in range(self.n)) for z in self.alloc}
  self.intercept={z:sum(self.values[i]*(z[i]!=0)-self.costs[i][z[i]] for i in range(self.n)) for z in self.alloc}

 def solve(self,theta,factor=F(1),values=None,exclude=None):
  best=None;bestval=None
  for z in self.alloc:
   if exclude is not None and z[exclude]!=0: continue
   if values is None: a=self.intercept[z]
   else: a=sum(values[i]*(z[i]!=0)-self.costs[i][z[i]] for i in range(self.n))
   val=a-sum(factor*theta[i]*self.q[z][i] for i in range(self.n))
   if bestval is None or val>bestval: best,bestval=z,val
  return best,bestval

 @lru_cache(maxsize=None)
 def envelope(self,i,others,factor=F(2),firm_only=False):
  alpha={}
  for z in self.alloc:
   if firm_only and any(self.q[z]):continue
   v=self.intercept[z]-sum(factor*others[j]*self.q[z][j] for j in range(self.n) if j!=i)
   if z[i] not in alpha or v>alpha[z[i]]:alpha[z[i]]=v
  pts={ZERO,BAR}
  for m,l in combinations(alpha,2):
   qm=self.menus[i][m][1];ql=self.menus[i][l][1]
   if qm!=ql:
    t=(alpha[m]-alpha[l])/(factor*(qm-ql))
    if 0<t<BAR:pts.add(t)
  pts=sorted(pts);intervals=[]
  for a,b in zip(pts[:-1],pts[1:]):
   mid=(a+b)/2
   # Tie choice of the focal mode does not affect utility; exact allocation
   # below reconstructs the full lexicographic optimum.
   m=max(sorted(alpha),key=lambda m:alpha[m]-factor*mid*self.menus[i][m][1])
   intervals.append((a,b,m,self.menus[i][m][1]))
  return intervals

 def choose(self,theta,factor=F(2),firm_only=False):
  if not firm_only:return self.solve(theta,factor)[0]
  aa=[z for z in self.alloc if not any(self.q[z])]
  return max(aa,key=lambda z:self.intercept[z])

 def payment(self,i,theta,z=None,factor=F(2),firm_only=False):
  z=z or self.choose(theta,factor,firm_only)
  others=tuple(ZERO if j==i else theta[j] for j in range(self.n))
  intervals=self.envelope(i,others,factor,firm_only)
  rent=sum(q*max(ZERO,b-max(theta[i],a)) for a,b,m,q in intervals)
  return self.values[i]*(z[i]!=0)-theta[i]*self.q[z][i]-rent

 def vcg(self,theta,values=None):
  values=tuple(values or self.values)
  z,w=self.solve(theta,values=values)
  pp=[]
  for i in range(self.n):
   _,we=self.solve(theta,values=values,exclude=i)
   own=values[i]*(z[i]!=0)-theta[i]*self.q[z][i]
   pp.append(we-(w-own))
  return z,pp,w

def make_menus(d,rec):
 d=F(d);r=tuple(map(F,rec));z=(ZERO,)*3
 return [(z,ZERO),((d,ZERO,ZERO),ZERO),((d/2,d*r[0]/2,d*r[1]/2),d/2),((ZERO,d*r[0],d*r[1]),d)]

def check_screening():
 rng=random.Random(20261023)
 for case in range(60):
  n=2+case%3
  menus=[make_menus(1+rng.randrange(2),rng.choice([(1,0),(0,1),(F(1,2),F(1,2))])) for _ in range(n)]
  vals=[F(2+rng.randrange(8)) for _ in range(n)]
  cap=[F(1+rng.randrange(6),2) for _ in range(3)]
  costs=[[ZERO]+[F(rng.randrange(3),4) for _ in range(3)] for _ in range(n)]
  ins=Finite(menus,vals,cap,costs)
  for i,ms in enumerate(menus):
   for a,q in ms[1:]:assert sum(a)==sum(ms[1][0]);COUNTS['energy_checks']+=1
  theta=tuple(F(rng.randrange(9),2) for _ in range(n))
  for factor in (F(1),F(2)):
   for i in range(n):
    others=tuple(ZERO if j==i else theta[j] for j in range(n))
    intervals=ins.envelope(i,others,factor)
    points={ZERO,BAR,F(1,2),F(3,2),F(5,2),F(7,2)}
    for a,b,m,q in intervals:points.update((a,b,(a+b)/2))
    pts=sorted(points)
    prev=None
    for t in pts:
     tt=list(theta);tt[i]=t;tt=tuple(tt)
     z=ins.choose(tt,factor);p=ins.payment(i,tt,z,factor)
     q=ins.q[z][i]
     if prev is not None:assert prev>=q
     prev=q
     u=vals[i]*(z[i]!=0)-t*q-p
     assert u>=0
     for report in pts:
      rr=list(tt);rr[i]=report;rr=tuple(rr)
      zz=ins.choose(rr,factor);pay=ins.payment(i,rr,zz,factor)
      dev=vals[i]*(zz[i]!=0)-t*ins.q[zz][i]-pay
      assert dev<=u,(case,i,t,report,u,dev)
      COUNTS['scalar_deviation_comparisons']+=1
    actual=virtual=ZERO
    for a,b,m,q in intervals:
     mid=(a+b)/2;tt=list(theta);tt[i]=mid;tt=tuple(tt)
     zz=ins.choose(tt,factor);pay=ins.payment(i,tt,zz,factor)
     actual+=pay*(b-a)/BAR
     # Expected receipts use phi=2t, including for the efficient allocation.
     virtual+=(vals[i]*(zz[i]!=0)*(b-a)-q*(b*b-a*a))/BAR
    assert actual==virtual,(actual,virtual)
    COUNTS['conditional_revenue_identities']+=1
  ze,se=ins.solve(theta,F(1));zr,vr=ins.solve(theta,F(2))
  sr=ins.intercept[zr]-sum(theta[i]*ins.q[zr][i] for i in range(n))
  ie=sum(theta[i]*ins.q[ze][i] for i in range(n));ir=sum(theta[i]*ins.q[zr][i] for i in range(n))
  assert ZERO<=se-sr<=ie-ir
  COUNTS['distortion_bounds']+=1;COUNTS['screening_cases']+=1

def check_vcg():
 # Both private dimensions are checked using an independent enumeration
 # objective, without the ratepayer virtual-cost transformation.
 for case in range(12):
  menus=[make_menus(1,rec) for rec in [(1,0),(0,1),(F(1,2),F(1,2))]]
  ins=Finite(menus,[4,5,6],[F(1+case%3),F(1+(case//3)%2),F(1)])
  theta=tuple(F(x) for x in (1,2,3));values=ins.values
  z,pp,w=ins.vcg(theta)
  for i in range(3):
   own=values[i]*(z[i]!=0)-theta[i]*ins.q[z][i]
   assert 0<=pp[i]<=own
   trueu=own-pp[i]
   for b,c in product(map(F,[0,1,3,7]),map(F,[0,1,2,4])):
    vv=list(values);vv[i]=b;tt=list(theta);tt[i]=c
    zz,p2,_=ins.vcg(tuple(tt),tuple(vv))
    du=values[i]*(zz[i]!=0)-theta[i]*ins.q[zz][i]-p2[i]
    assert du<=trueu;COUNTS['vcg_deviation_comparisons']+=1
   _,we=ins.solve(theta,exclude=i)
   for m in range(len(menus[i])):
    zs=[zz for zz in ins.alloc if zz[i]==m]
    if not zs:continue
    other=max(sum(values[j]*(zz[j]!=0)-theta[j]*ins.q[zz][j] for j in range(3) if j!=i) for zz in zs)
    price=we-other
    mu=values[i]*(m!=0)-theta[i]*menus[i][m][1]-price
    assert mu<=trueu
    if m==z[i]:assert price==pp[i]
    COUNTS['vcg_menu_checks']+=1

def sequential(ins,theta,order,priced=False):
 remain=list(ins.cap);z=[0]*ins.n;payments=[ZERO]*ins.n
 for i in order:
  options=[m for m,(a,q) in enumerate(ins.menus[i]) if all(a[t]<=remain[t] for t in range(3))]
  m=max(options,key=lambda m:ins.values[i]*(m!=0)-theta[i]*ins.menus[i][m][1])
  z[i]=m
  if priced and m:
   qmin=min(ins.menus[i][j][1] for j in options if j)
   payments[i]=max(ZERO,ins.values[i]-BAR*qmin)
  for t in range(3):remain[t]-=ins.menus[i][m][0][t]
 assert tuple(z) in ins.alloc
 return tuple(z),payments

def policy_grid():
 menus=[make_menus(d,r) for d,r in zip([1,2,1,2],[(1,0),(0,1),(F(1,2),F(1,2)),(F(1,2),F(1,2))])]
 capacities=[(2,2,2),(3,1,2),(1,3,2)]
 policies=['Full information','Ratepayer optimal','Efficient envelope','Efficient VCG','Firm only','Arrival order','Arrival envelope','Public flexibility order','Flexibility envelope']
 sums={p:[ZERO]*7 for p in policies};rows=[]
 for cap in capacities:
  ins=Finite(menus,[4,7,5,8],cap)
  for theta in product([F(1,2),F(3,2),F(5,2),F(7,2)],repeat=4):
   er,se=ins.solve(theta)
   solutions={'Full information':(er,[ins.values[i]*(er[i]!=0)-theta[i]*ins.q[er][i] for i in range(4)])}
   for name,factor,firm in [('Ratepayer optimal',F(2),False),('Efficient envelope',F(1),False),('Firm only',F(2),True)]:
    z=ins.choose(theta,factor,firm)
    solutions[name]=(z,[ins.payment(i,theta,z,factor,firm) for i in range(4)])
   z,p,_=ins.vcg(theta);solutions['Efficient VCG']=(z,p)
   solutions['Arrival order']=sequential(ins,theta,range(4))
   solutions['Public flexibility order']=sequential(ins,theta,[1,3,0,2])
   solutions['Arrival envelope']=sequential(ins,theta,range(4),True)
   solutions['Flexibility envelope']=sequential(ins,theta,[1,3,0,2],True)
   for name,(z,p) in solutions.items():
    w=ins.intercept[z]-sum(theta[i]*ins.q[z][i] for i in range(4))
    revenue=sum(p);rent=w-revenue
    adm=sum(m!=0 for m in z);kw=sum([1,2,1,2][i] for i in range(4) if z[i])
    expo=sum(ins.q[z]);delay=F(4-adm,4)
    assert rent>=0 and z in ins.alloc
    # Complete certified profiles finish all admitted work in the horizon.
    vals=[F(kw),F(adm),revenue,w,rent,expo,delay]
    sums[name]=[a+b for a,b in zip(sums[name],vals)]
    rows.append({'capacity':cap,'theta':theta,'policy':name,'modes':z,'payments':p,'metrics':vals})
   COUNTS['policy_profile_comparisons']+=1
 count=COUNTS['policy_profile_comparisons']
 means={p:[x/count for x in row] for p,row in sums.items()}
 (P/'policy_grid.json').write_text(json.dumps({'metric_order':['early_capacity','admissions','ratepayer_receipts','total_surplus','rents','exposure','mean_normalized_delay'],'profiles':count,'means':means,'rows':rows},default=encode,indent=2))
 return means

def check_analytical():
 for b,d,h in product([F(1),F(2)],[F(1,2),F(1),F(2)],[F(3),F(4),F(5)]):
  if d>=h:continue
  tau=d/2
  rev=b+tau*(d-tau)/h;sur=b+(d*tau-tau*tau/2)/h;rent=tau*tau/(2*h)
  assert sur==rev+rent
  assert (b+d*d/(2*h))-sur==d*d/(8*h)
  assert (b+d*d/(2*h))-rev==d*d/(4*h)
  COUNTS['two_applicant_checks']+=1
 for r in range(2,13):
  broad=[F(1,r)]*r+[ZERO]
  narrow=[[F(1,r) if j==i else ZERO for j in range(r)]+[1-F(1,r)] for i in range(r)]
  cap=[F(1,r)]*r+[F(r-1)]
  assert sum(broad)==1 and all(sum(a)==1 for a in narrow)
  assert all(sum(a[t] for a in narrow)<=cap[t] for t in range(r+1))
  assert all(any(broad[t]+a[t]>cap[t] for t in range(r+1)) for a in narrow)
  COUNTS['ranking_constructions']+=1
 pair=(1,1)
 probs=[]
 for states in [[(2,2),(0,0)],[(2,0),(0,2)]]:
  violation=F(sum(any(pair[t]>c[t] for t in range(2)) for c in states),2)
  probs.append(violation)
 assert probs==[F(1,2),F(1)];COUNTS['scenario_checks']+=2
 # Explicit counterexamples for both directions of early capacity and an
 # individual exposure increase in Proposition propRentDistortion.
 loads=[(F(1,2),F(1,2)),(F(1),ZERO),(ZERO,F(1))]
 for values,qq,expected in [([4,F(5,4),F(5,4)],[1,0,0],((1,0,0),(0,1,1))),([F(5,2)]*3,[0,1,1],((0,1,1),(1,0,0)))]:
  menus=[[((ZERO,ZERO),ZERO),(a,F(q))] for a,q in zip(loads,qq)]
  ins=Finite(menus,values,[1,1]);tt=(F(1),)*3
  assert (ins.solve(tt)[0],ins.solve(tt,F(2))[0])==expected
  COUNTS['allocation_reversal_constructions']+=1
 ins=Finite([[((ZERO,),ZERO),((F(2),),F(2))],[((ZERO,),ZERO),((F(1),),F(1))]],[5,F(7,2)],[2])
 assert ins.solve((F(1),F(1)))[0]==(1,0)
 assert ins.solve((F(1),F(1)),F(2))[0]==(0,1)
 COUNTS['allocation_reversal_constructions']+=1
 # The displayed efficient welfare and payment vectors.
 menus=[[((ZERO,ZERO),ZERO),((F(1),ZERO),ZERO),((ZERO,F(1)),F(1))],
        [((ZERO,ZERO),ZERO),((F(1),ZERO),ZERO)],
        [((ZERO,ZERO),ZERO),((ZERO,F(1)),ZERO)]]
 ins=Finite(menus,[10,8,7],[1,1])
 for t,w,p in [(F(1,2),F(35,2),[F(7),F(15,2),ZERO]),(F(2),F(17),[F(8),ZERO,F(6)])]:
  z,pp,ww=ins.vcg((t,ZERO,ZERO));assert ww==w and pp==p
  COUNTS['displayed_example_checks']+=1
 # The exact cost report and repair counterexample uses the relaxed model's
 # VCG charges before the public repair, as stated in the manuscript.
 allmenus=[menus[0]]*3
 relax=Finite(allmenus,[10,10,10],[1,3])
 z,p,w=relax.vcg((F(3),F(2),F(1)));assert z==(1,2,2) and p==[F(2),ZERO,ZERO]
 z2,p2,w2=relax.vcg((F(3),F(2),F(4)));assert z2==(2,2,1) and p2[2]==3
 assert 10-p2[2]==7
 COUNTS['displayed_example_checks']+=1
 chain=[[[((ZERO,ZERO,ZERO),ZERO),((F(1),ZERO,ZERO),ZERO),((ZERO,F(1),ZERO),F(1))],
         [((ZERO,ZERO,ZERO),ZERO),((ZERO,F(1),ZERO),ZERO),((ZERO,ZERO,F(1)),F(1))],
         [((ZERO,ZERO,ZERO),ZERO),((ZERO,ZERO,F(1)),ZERO)]]]
 for cap,w in [((1,1,1),23),((0,1,1),16),((1,0,1),18),((1,1,0),19)]:
  assert Finite(chain[0],[10,9,4],cap).solve((F(2),F(1),ZERO))[1]==w
 COUNTS['displayed_example_checks']+=1

def main():
 check_screening();check_vcg();check_analytical();means=policy_grid()
 out={'status':'passed','arithmetic':'exact fractions','counts':COUNTS,'policy_means':means,'python':platform.python_version(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 (P/'screening_verification.json').write_text(json.dumps(out,default=encode,indent=2))
 print(json.dumps(out,default=encode))

if __name__=='__main__':main()
