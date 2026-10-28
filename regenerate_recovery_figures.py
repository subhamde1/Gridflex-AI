"""Generate PNG figures from the proved admission and menu-loss formulas."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from figure_export import export_figure

P = Path(__file__).resolve().parent
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10,
                     'axes.labelsize': 10, 'axes.titlesize': 10,
                     'xtick.labelsize': 9, 'ytick.labelsize': 9,
                     'legend.fontsize': 8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'lines.linewidth': 1.7})
blue, green, orange = '#164a72', '#237f72', '#bf682b'

gamma = np.linspace(0, 6, 121)
oldp = np.full_like(gamma, .75)
newp = (6-gamma)/8
oldr = np.full_like(gamma, 13/4)
newr = 2+gamma+(6-gamma)**2/16
fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.8), layout='constrained')
ax[0].plot(gamma, oldp, '--', color=blue, label='Original capacity')
ax[0].plot(gamma, newp, color=green, label='Expanded capacity')
ax[0].set(xlabel=r'Bundled-portfolio advantage $\gamma$', ylabel='Early-admission probability',
          title='(a) Focal applicant', xlim=(0, 6), ylim=(0, .85))
ax[0].set_yticks([0, .25, .5, .75]); ax[0].legend(loc='lower left', frameon=False)
ax[1].plot(gamma, oldr, '--', color=blue, label='Original capacity')
ax[1].plot(gamma, newr, color=green, label='Expanded capacity')
ax[1].set(xlabel=r'Bundled-portfolio advantage $\gamma$', ylabel='Expected ratepayer net benefit',
          title='(b) Entire cohort', xlim=(0, 6), ylim=(2.8, 8.4))
ax[1].set_yticks([3, 4, 5, 6, 7, 8]); ax[1].legend(loc='upper left', frameon=False)
for a in ax: a.grid(axis='y', alpha=.18)
export_figure(fig, P/'Pic8'); plt.close(fig)

def omega(A, q, h):
    A = np.asarray(A)
    return np.where(A <= 0, 0, np.where(A < 2*h*q, A*A/(4*h*q), A-h*q))

A = np.linspace(-1, 20, 211)
v = np.linspace(0, 12, 121)
left, right = omega(A, 2, 4), omega(v, 1, 4)
fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.8), layout='constrained')
ax[0].plot(A, left, color=blue, label='Expected receipt loss')
ax[0].plot(A[::20], left[::20], 'o', ms=3.5, mfc='white', mec=orange,
           label='Expected omitted-mode deficit')
ax[0].set(xlabel=r'Net public bundle value $A=v-L-K$', ylabel='Expected value',
          title='(a) Integer capacity', xlim=(-1, 20), ylim=(-.4, 12.8))
ax[0].set_xticks([0, 4, 8, 12, 16, 20]); ax[0].legend(loc='upper left', frameon=False)
ax[1].plot(v, right, color=green, label='Expected receipt loss')
ax[1].plot(v[::15], right[::15], 's', ms=3.5, mfc='white', mec=orange,
           label='Expected fractional-capacity term')
ax[1].set(xlabel=r'Public value $v$', ylabel='Expected value',
          title='(b) Half-unit capacities', xlim=(0, 12), ylim=(-.3, 8.5))
ax[1].set_xticks([0, 2, 4, 6, 8, 10, 12]); ax[1].legend(loc='upper left', frameon=False)
for a in ax: a.grid(axis='y', alpha=.18)
export_figure(fig, P/'Pic9'); plt.close(fig)

print('Generated Pic8.png and Pic9.png')
