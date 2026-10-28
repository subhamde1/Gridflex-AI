"""Generate Pic12 and Pic13 from validated intervals and exact fractions.

The exact source records, rather than rounded table endpoints, determine all
positions. Original figures Pic1--Pic11 are not touched by this program.
"""
from pathlib import Path
from fractions import Fraction as F
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,FixedFormatter
from figure_export import export_figure

P=Path(__file__).resolve().parent
R=P/'replication'
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,
 'axes.labelsize':10,'axes.titlesize':10,'legend.fontsize':9,
 'xtick.labelsize':9,'ytick.labelsize':9,'lines.linewidth':1.6,
 'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True,
 'savefig.facecolor':'white'})
blue,green,orange='#16557a','#208568','#b35a29'
richer=json.loads((R/'richer_menu_results.json').read_text())
original=json.loads((R/'menu_standardization.json').read_text())
old={(r['beta'],r['stress_capacity'],r['bundled_eligible']):r for r in original['rows']}
rows=[r for r in richer['rows'] if r['original_unit_decision']=='below target']
y=np.arange(len(rows))
fig,(ax,counts)=plt.subplots(1,2,figsize=(8.8,6.5),sharey=True,
    gridspec_kw={'width_ratios':[4.2,1.2]},layout='constrained')
for j,r in enumerate(rows):
    key=(r['beta'],r['stress_capacity'],r['original_bundled_eligible'])
    unit=list(map(F,old[key]['retention_interval']))
    selected=list(map(F,r['selected_retention_interval']))
    for pos,ab,color,marker,label in [(j+.12,unit,orange,'s','Unit base'),
                                     (j-.12,selected,green,'o','Minimum richer menu')]:
        lo,hi=[100*float(x) for x in ab];mid=(lo+hi)/2
        ax.errorbar([mid],[pos],xerr=[[mid-lo],[hi-mid]],fmt=marker,
            color=color,markerfacecolor='white',markersize=4.4,capsize=2.8,
            label=label if j==0 else None,zorder=3)
    counts.barh(j,r['minimum_paired_option_count'],height=.6,color=blue,alpha=.84)
    counts.text(r['minimum_paired_option_count']+.10,j,str(r['minimum_paired_option_count']),va='center',fontsize=9)
ax.axvline(95,color='#7a3430',ls='--',lw=1.25,label='95% target')
ax.set(yticks=y,yticklabels=[f"({r['beta']}, {r['stress_capacity']}, {r['original_bundled_eligible']})" for r in rows],
       xlabel='Expected financial retention (%)',ylabel=r'Original setting $(\beta,s,k)$',
       xlim=(50,102),xticks=[50,60,70,80,90,95,100],title='(a) Same full-menu benchmark')
ax.set_ylim(len(rows)-.5,-.7)
ax.grid(axis='x',alpha=.17)
ax.legend(loc='lower left',bbox_to_anchor=(0,1.025),ncol=3,frameon=False,
          handlelength=1.5,columnspacing=1.0)
counts.set(xlabel='Permissions',xticks=[0,1,2],xlim=(0,2.7),title='(b) Exact minimum')
counts.grid(axis='x',alpha=.13);counts.tick_params(axis='y',left=False)
export_figure(fig,P/'Pic12',padding=.10)
plt.close(fig)

theory=json.loads((R/'permission_design.json').read_text())
front=theory['uniform_design_frontier'];rev=theory['recipient_reversal_example']
fig,axes=plt.subplots(1,2,figsize=(8.8,3.8),layout='constrained',
                        gridspec_kw={'width_ratios':[1.12,1]})
k=[r['permissions'] for r in front]
ret=[100*float(F(r['retention'])) for r in front]
axes[0].plot(k,ret,color=blue,marker='o',markersize=5,markerfacecolor='white')
axes[0].axhline(95,color='#7a3430',ls='--',lw=1.2,label='95% target')
axes[0].scatter([2],[ret[2]],s=55,color=green,zorder=4,label='Minimum: 2 permissions')
axes[0].annotate('96.49%',(2,ret[2]),xytext=(2.1,91.1),fontsize=9,color=green,
                 arrowprops={'arrowstyle':'-','color':green,'lw':.9})
axes[0].set(xlabel='Retained recovery permissions',ylabel='Expected financial retention (%)',
            xlim=(-.2,4.2),ylim=(77,102),xticks=k,yticks=[80,85,90,95,100],
            title='(a) Exact near-full-firm frontier')
axes[0].grid(alpha=.17);axes[0].legend(loc='lower right',frameon=False)
xs=[r['firm_capacity'] for r in rev]
for key,name,color,marker in [('A_increment','Applicant A',blue,'o'),('B_increment','Applicant B',orange,'s')]:
    vals=[float(F(r[key])) for r in rev]
    axes[1].plot(xs,vals,color=color,marker=marker,markersize=5,
                 markerfacecolor='white',label=name)
    for x,val in zip(xs,vals):
        dx=-.025 if x==0 else .025
        axes[1].text(x+dx,val*(1.13 if name.endswith('A') else .82),f'{val:g}',
          ha='right' if x==0 else 'left',va='center',color=color,fontsize=9)
axes[1].set_yscale('log')
axes[1].set(xlabel='Firm capacity',ylabel='Gain over firm-only receipts (log scale)',
            xticks=[0,1],xlim=(-.23,1.23),ylim=(.085,4.8),
            title='(b) Capacity reverses recipient priority')
axes[1].yaxis.set_major_locator(FixedLocator([.1,.25,.5,1,2,4]))
axes[1].yaxis.set_major_formatter(FixedFormatter(['0.1','0.25','0.5','1','2','4']))
axes[1].yaxis.set_minor_locator(FixedLocator([]))
axes[1].grid(axis='y',alpha=.17)
axes[1].legend(loc='upper right',frameon=False)
export_figure(fig,P/'Pic13',padding=.10)
plt.close(fig)

print('Generated Pic12.png and Pic13.png from validated or exact rational inputs.')
