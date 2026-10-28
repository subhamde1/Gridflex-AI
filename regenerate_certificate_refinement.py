"""Plot only the proved formulas in the certificate separation proposition."""
from pathlib import Path
from fractions import Fraction as F
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figure_export import export_figure

P = Path(__file__).resolve().parent
record = json.loads((P/'replication/reoptimized_certificates.json').read_text())
rows = record['separation_figure_data']
n = [r['n'] for r in rows]
old = [float(F(r['restricted_expected_bound'])) for r in rows]
new = [float(F(r['reoptimized_expected_bound'])) for r in rows]
oldr = [float(F(r['restricted_retention_bound'])) for r in rows]
newr = [float(F(r['reoptimized_retention_bound'])) for r in rows]
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10,
                     'axes.labelsize': 10, 'axes.titlesize': 10,
                     'xtick.labelsize': 9, 'ytick.labelsize': 9,
                     'legend.fontsize': 8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'lines.linewidth': 1.7})
blue, green, orange = '#164a72', '#237f72', '#bf682b'
fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.8), layout='constrained')
ax[0].plot(n, old, '--', color=blue, label='Restricted-price certificate')
ax[0].plot(n, new, color=green, label='Reoptimized certificate')
ax[0].plot(n, [0]*len(n), ':', color=orange, label='Actual loss')
ax[0].set(xlabel=r'Number of bundled applicants $n$', ylabel='Expected bound or loss',
          title='(a) Same optimal integer menu', xlim=(2, 64), ylim=(-4, 150))
ax[0].set_xticks([2, 16, 32, 48, 64]); ax[0].legend(loc='upper left', frameon=False)
ax[1].plot(n, oldr, '--', color=blue, label='Restricted-price guarantee')
ax[1].plot(n, newr, color=green, label='Reoptimized guarantee')
ax[1].plot(n, [1]*len(n), ':', color=orange, label='Actual retention')
ax[1].set(xlabel=r'Number of bundled applicants $n$', ylabel='Financial retention ratio',
          title='(b) Receipts fixed at 19', xlim=(2, 64), ylim=(0, 1.06))
ax[1].set_xticks([2, 16, 32, 48, 64]); ax[1].set_yticks([0, .25, .5, .75, 1])
ax[1].legend(loc='center right', frameon=False)
for a in ax:
    a.grid(axis='y', alpha=.18)
export_figure(fig, P/'Pic10')
plt.close(fig)
print('Generated Pic10.png from exact formulas.')
