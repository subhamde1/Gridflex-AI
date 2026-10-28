"""Plot validated retention enclosures, not sampled observations."""
from pathlib import Path
from fractions import Fraction as F
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figure_export import export_figure

P = Path(__file__).resolve().parent
data = json.loads((P/'replication/menu_standardization.json').read_text())
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10,
    'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'axes.spines.top': False,
    'axes.spines.right': False, 'lines.linewidth': 1.5})
fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.65), sharey=True,
                         layout='constrained')
colors = {2:'#164a72', 4:'#237f72'}
for ax, beta, title in zip(axes, ['1/2', '1', '2'],
                            [r'$\beta=1/2$', r'$\beta=1$', r'$\beta=2$']):
    for k in [2, 4]:
        rows = [r for r in data['rows'] if r['beta']==beta and r['bundled_eligible']==k]
        xx = [r['stress_capacity']+(-0.06 if k==2 else 0.06) for r in rows]
        lo = [float(F(r['retention_interval'][0]))*100 for r in rows]
        hi = [float(F(r['retention_interval'][1]))*100 for r in rows]
        mid = [(l+h)/2 for l, h in zip(lo, hi)]
        ax.errorbar(xx, mid, yerr=[[v-l for v,l in zip(mid,lo)],
                                 [h-v for h,v in zip(hi,mid)]],
                    fmt='o' if k==2 else 's', markersize=4, capsize=4,
                    color=colors[k], markerfacecolor='white',
                    label=rf'$k={k}$')
    ax.axhline(95, color='#a9462e', linestyle='--', linewidth=1.3,
               label='95% target')
    ax.set(xlabel=r'Stressed-period capacity $s$', title=title,
           xlim=(-0.23, 2.23), ylim=(50, 102))
    ax.set_xticks([0, 1, 2]); ax.set_yticks([50, 60, 70, 80, 90, 100])
    ax.grid(axis='y', alpha=.18)
axes[0].set_ylabel('Expected financial retention (%)')
axes[0].legend(loc='center left', frameon=False)
export_figure(fig, P/'Pic11')
plt.close(fig)
print('Generated Pic11.png from validated intervals.')
