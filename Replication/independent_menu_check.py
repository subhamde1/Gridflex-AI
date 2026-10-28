"""Check compressed-menu values against direct physical mode enumeration.
This script does not import the manuscript's clearing or matching routines.
"""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import random,json
ROOT=Path(__file__).resolve().parent
VALUES=[4,5,6,7]
LOADS=[(0,0,0,0),(2,0,0,0),(0,2,0,0),(1,1,0,0),(0,0,1,1)]
EXPOSURE=[F(0),F(0),F(1),F(1,2),F(1)]

def physical_allocations(stress,mask):
    opts=[[0,1,2,3]+([4] if mask&(1<<i) else []) for i in range(4)]
    return [z for z in product(*opts) if all(sum(LOADS[m][t] for m in z)<=2*c for t,c in enumerate((stress,1,1,1)))]

def compressed_allocations(stress,mask):
    opts=[[0,1,2]+([3] if mask&(1<<i) else []) for i in range(4)]
    return [z for z in product(*opts) if z.count(1)<=stress and z.count(2)<=1 and z.count(3)<=2]

def value(z,penalty,physical=True):
    return sum((F(VALUES[i])*(m!=0)-penalty[i]*(EXPOSURE[m] if physical else F(m in (2,3))) for i,m in enumerate(z)),F(0))

def run():
    rng=random.Random(2026102617);checks=0;subset_checks=0;three_checks=0
    for stress in range(3):
        full={mask:physical_allocations(stress,mask) for mask in range(16)}
        short={mask:compressed_allocations(stress,mask) for mask in range(16)}
        penalties=[[F(rng.randrange(49),4) for _ in range(4)] for _ in range(32)]
        penalties+=[[F(0)]*4,[F(4)]*4,[F(8)]*4,[F(12)]*4]
        for penalty in penalties:
            val={mask:max(value(z,penalty) for z in full[mask]) for mask in range(16)}
            for mask in range(16):
                assert val[mask]==max(value(z,penalty,False) for z in short[mask]);checks+=1
                for i in range(4):
                    if not mask&(1<<i):assert val[mask]<=val[mask|(1<<i)];subset_checks+=1
                if mask.bit_count()==3:assert val[mask]==val[15];three_checks+=1
    result={'status':'passed','direct_physical_vs_compressed_values':checks,'menu_inclusion_monotonicity_checks':subset_checks,'three_permission_exactness_checks':three_checks,'all_arithmetic':'exact fractions','input_values':VALUES,'physical_loads_in_half_units':LOADS,'source_menus_include_half_firm_options':True,'claims':'These are finite independent checks; the complete-recovery-pool reassignment argument establishes three-permission exactness for all profiles in this design.'}
    (ROOT/'independent_menu_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':run()
