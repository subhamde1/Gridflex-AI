"""Regenerate all seven theoretical figures from formulas and exact checks."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.colors import Normalize
from figure_export import export_figure

ROOT=Path(__file__).parent
P=ROOT;R=ROOT/'replication'
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.labelsize':10,
 'axes.titlesize':10,'legend.fontsize':9.5,'xtick.labelsize':9,'ytick.labelsize':9,
 'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
 'lines.linewidth':1.6,'savefig.facecolor':'white','axes.axisbelow':True})
blue='#16557a';orange='#be641c';green='#208568';purple='#875694'
CURRENT_NUMBERS={1:3,2:4,3:5,14:6,15:7}
def load(name):return json.loads((R/name).read_text())
def save(fig,n,values):
 export_figure(fig,P/f'Pic{CURRENT_NUMBERS[n]}',padding=.08)
 plt.close(fig)
def grid(ax):ax.grid(alpha=.18,linestyle=':')

# Figure 1: exact piecewise-linear formulas, including the utility switch.
x=np.linspace(0,4,401)
fig,axs=plt.subplots(1,2,figsize=(7.5,4),layout='constrained')
for ax,a,b,ylabel in [(axs[0],18-x,np.full_like(x,17),'Total welfare'),
                       (axs[1],3-x,np.full_like(x,2),'A utility')]:
 ax.plot(x,a,color=blue,label='A deferred and B' if ax==axs[0] else 'Deferred utility: 3 − cost')
 ax.plot(x,b,color=green,label='A firm and C' if ax==axs[0] else 'Firm utility: 2')
 ax.plot(x,np.maximum(a,b),'--',color='black',label='Efficient envelope' if ax==axs[0] else 'Truthful utility')
 ax.axvline(1,color='.45',ls=':',lw=1);ax.set(xlabel='A deferral cost',ylabel=ylabel)
 grid(ax);ax.legend(loc='upper right')
save(fig,1,{'welfare':'max(18-theta,17)','utility':'max(3-theta,2)','switch':1})

# Figure 2: exact r=4 energy-conserving construction; flat vector cells.
a=np.zeros((5,5));a[0,:4]=.25
for j in range(4):a[j+1,j]=.25;a[j+1,4]=.75
fig,ax=plt.subplots(figsize=(7.5,4.2),layout='constrained')
im=ax.pcolormesh(np.arange(6)-.5,np.arange(6)-.5,a,cmap='Blues',vmin=0,vmax=.75,edgecolors='white',lw=.6,rasterized=False)
ax.set(xticks=range(5),xticklabels=['Recovery 1','Recovery 2','Recovery 3','Recovery 4','Shared slack'],yticks=range(5),yticklabels=['Broad profile']+[f'Narrow {i}' for i in range(1,5)])
ax.invert_yaxis();ax.set_xlabel('Capacity: 0.25 in each recovery slot; 3.00 in shared slack')
for (i,j),v in np.ndenumerate(a):ax.text(j,i,f'{v:.2f}',ha='center',va='center',color='white' if v>.5 else 'black')
cb=fig.colorbar(im,ax=ax,label='Recovered energy',fraction=.04,pad=.025)
cb.solids.set_rasterized(False); cb.solids.set_edgecolor('face')
save(fig,2,a.tolist())

# Figure 3: displacement graph, no raster or approximated image tracing.
fig,ax=plt.subplots(figsize=(7.5,2.1));ax.set(xlim=(-.15,10.15),ylim=(-.25,2.0));ax.axis('off')
centers=[.8,3.55,6.3,9.05]
for c,label in zip(centers,['Period A','Period B','Period C','Rejection']):
 ax.add_patch(FancyBboxPatch((c-.69,.55),1.38,.51,boxstyle='round,pad=0.08',edgecolor=blue,facecolor='#eff6fa',lw=1.4))
 ax.text(c,.805,label,ha='center',va='center',color=blue,fontweight='bold')
for j,label in enumerate(['Move 1: loss 2','Move 2: loss 1','Drop 3: loss 4']):
 ax.add_patch(FancyArrowPatch((centers[j]+.81,.805),(centers[j+1]-.81,.805),arrowstyle='->',mutation_scale=16,color=blue,lw=1.6))
 ax.text((centers[j]+centers[j+1])/2,1.38,label,ha='center',fontsize=10)
ax.text(4.95,.0,'Price of reserving A = 2 + 1 + 4 = 7',ha='center',color=blue,fontweight='bold',fontsize=12)
save(fig,3,{'arcs':[['A','B',2],['B','C',1],['C','exit',4]],'price':7})


# Figure 14: closed-form mixed-portfolio counterexample, not observed data.
mixed=load('mixed_portfolio_results.json')
assert mixed['maximum_absolute_difference']==0
xx=np.linspace(0,10,501)
exact=np.minimum(xx,1)
fig,ax=plt.subplots(figsize=(7.2,3.8),layout='constrained')
ax.plot(xx,xx,color=orange,ls='--',label='Bundled award held fixed: L')
ax.plot(xx,exact,color=blue,lw=2.3,label='Exact opportunity price: min(L, 1)')
ax.axvline(1,color='.5',ls=':',lw=1)
ax.scatter([1],[1],color=blue,s=24,zorder=4)
ax.set(xlabel='Ordinary competitor net value L',ylabel='Opportunity price',
       xlim=(0,10),ylim=(0,10.4),xticks=[0,1,2,4,6,8,10])
ax.text(5.2,1.45,'For L > 1, release capacity by rejecting\nthe bundled competitor',
        fontsize=9,ha='center',va='bottom',color=blue)
grid(ax)
ax.legend(loc='upper left')
save(fig,14,{'ordinary_value':xx.tolist(),'bundled_value':1,
             'exact_price':exact.tolist(),'conditional_price':xx.tolist(),
             'capacity':[2,1],'focal_load':[1,0],
             'source':'Proposition propositionFrozen; mixed_portfolio_results.json'})

plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.labelsize':10,
 'axes.titlesize':10,'legend.fontsize':8.5,'xtick.labelsize':9,'ytick.labelsize':9,
 'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
 'lines.linewidth':1.8,'savefig.facecolor':'white','axes.axisbelow':True})
blue,orange,green='#16557a','#be641c','#208568'
v=load('pricing_stability_verification.json')


def save(fig,n,values):
    export_figure(fig,ROOT/f'Pic{CURRENT_NUMBERS[n]}',padding=.10)
    plt.close(fig)


fig,ax=plt.subplots(1,2,figsize=(8,3.8),layout='constrained')
pval=np.array([4,8,16,32,64],dtype=int)
ax[0].plot(pval,np.full(len(pval),2),color=blue,marker='o',label='Original charge: 2')
ax[0].plot(pval,pval+1,color=orange,marker='s',label='Expanded charge: P + 1')
ax[0].set(xscale='log',xlabel='Ordinary competitor value P',ylabel='Exact focal charge',title='(a) Same profile, higher charge')
ax[0].set_xticks(pval,labels=[str(p) for p in pval]);ax[0].legend(loc='upper left')
sharp=[z for z in v['examples'] if z['kind']=='sharp']
strict=[z for z in v['examples'] if z['kind']=='unchanged_award']
gx=[z['gamma4']/4 for z in sharp];gy=[z['change4']/4 for z in sharp]
sx=[z['gamma4']/4 for z in strict];sy=[z['change4']/4 for z in strict]
line=np.array([1,140])
ax[1].plot(line,line,color='.4',ls='--',label='Bound: increase = loss')
ax[1].scatter(gx,gy,s=32,color=blue,marker='o',label='Exact sharpness cases',zorder=3)
ax[1].scatter(sx,sy,s=29,color=green,marker='^',label='Strict same-award cases',zorder=3)
ax[1].set(xscale='log',yscale='log',xlabel='Common-portfolio welfare loss',ylabel='Increase in exact price',title='(b) A sharp additive bound')
ax[1].legend(loc='upper left',fontsize=8)
for a in ax:a.grid(alpha=.2,ls=':')
save(fig,15,{'P':pval.tolist(),'original_charge':[2]*len(pval),'expanded_charge':(pval+1).tolist(),
             'sharp_gamma':gx,'sharp_increase':gy,'strict_gamma':sx,'strict_increase':sy,
             'source':'pricing_stability_verification.json and Proposition capacity formulas'})



# New scalar screening figures use the same pre-increment typography as the
# retained generators. export_figure applies exactly +1 pt to every Text object.
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.labelsize':10,
 'axes.titlesize':10,'legend.fontsize':9.5,'xtick.labelsize':9,'ytick.labelsize':9,
 'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})

theta=np.linspace(0,4,401)
fig,ax=plt.subplots(1,2,figsize=(8,4),layout='constrained')
ax[0].plot(theta,np.maximum(2-theta,0),color=blue,label='Positive efficient advantage')
ax[0].plot(theta,np.maximum(2-2*theta,0),color=orange,label='Positive virtual advantage')
ax[0].axvline(1,color=orange,ls=':',lw=1.4)
ax[0].axvline(2,color=blue,ls=':',lw=1.4)
ax[0].set(xlabel='Private deferral cost',ylabel='Positive advantage over B',ylim=(-.04,2.25),title='(a) Early-admission thresholds')
ax[0].legend(loc='upper right',fontsize=8.5)
left=theta[theta<1];right=theta[theta>1]
ax[1].plot(left,np.full_like(left,2),color=orange,label='A charge when admitted')
ax[1].plot(left,1-left,color=blue,label='A truthful rent')
ax[1].plot(right,np.zeros_like(right),color=blue)
ax[1].scatter([1],[2],s=25,facecolors='white',edgecolors=orange,zorder=4)
ax[1].axvline(1,color='.5',ls=':',lw=1)
ax[1].set(xlabel='Private deferral cost',ylabel='Charge or applicant rent',ylim=(-.05,2.3),title='(b) Critical-cost implementation')
ax[1].legend(loc='upper right',fontsize=8.5)
for a in ax:a.grid(alpha=.2,ls=':')
export_figure(fig,ROOT/'Pic1',padding=.10)
plt.close(fig)

d=np.linspace(.05,3.95,391);b=1;h=4
fig,ax=plt.subplots(1,2,figsize=(8,4),layout='constrained')
ax[0].plot(d,b+d*d/(2*h),color=green,label='Full-information surplus')
ax[0].plot(d,b+3*d*d/(8*h),color=blue,label='Ratepayer-rule surplus')
ax[0].plot(d,b+d*d/(4*h),color=orange,label='Ratepayer receipts')
ax[0].set(xlabel='Connection-value difference d',ylabel='Expected normalized value',title='(a) Receipts and total surplus')
ax[0].legend(loc='upper left',fontsize=8.5)
ax[1].plot(d,d*d/(8*h),color=blue,label='Allocation surplus loss')
ax[1].plot(d,d*d/(8*h),'--',color=green,label='Ratepayer-rule applicant rent')
ax[1].plot(d,d*d/(4*h),color=orange,label='Full-information financial gap')
ax[1].set(xlabel='Connection-value difference d',ylabel='Expected loss or rent',title='(b) Exact information costs')
ax[1].legend(loc='upper left',fontsize=8.5)
for a in ax:a.grid(alpha=.2,ls=':')
export_figure(fig,ROOT/'Pic2',padding=.10)
plt.close(fig)
print('Generated seven mathematical figures, each as PNG only, with +1 pt text, at >=8000 x 3000 pixels and 1200 dpi.')
