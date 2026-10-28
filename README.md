# Who Gets Connected First? Replication code

Code for **Who Gets Connected First? Incentive-Compatible Capacity Allocation for Flexible AI Data Centers**, by **Subham De**, University of Texas at Dallas.

This folder supplies the calculation and figure-generation code for the manuscript, including its recovery-permission design analysis. Researchers generate the numerical result records and figures locally. No generated images, result archives, image-DPI reports, or editorial verification files are distributed here.

All numerical inputs are mathematical constructions. No company observations, proprietary datasets, paid solver, or account credentials are needed. The manuscript contains the theoretical statements and proofs; the finite computational checks supplement those proofs.

## License: All rights reserved

**Copyright (c) 2026 Subham De. All rights reserved.** See [LICENSE](LICENSE) for the complete notice.

No open-source or other public reuse license is granted for the original code and other original copyrighted materials in this repository. Except as permitted by applicable law or GitHub's applicable Terms of Service, prior written permission is required to use, execute, reproduce, modify, redistribute, or create derivative works from these materials, including for research, replication, teaching, or commercial purposes. Citing the manuscript does not grant permission.

Public access remains subject to GitHub's platform permissions, including viewing and forking public repositories within GitHub. Third-party dependencies retain their own license terms. Permission requests should be sent to **Subham De: subham.de@utdallas.edu**.

The technical instructions below describe the reproduction workflow for users who have the required permission or another applicable lawful basis. They do not grant additional rights to the code.

## Installation

The code was tested with CPython 3.12.14 on Linux x86-64 and the package versions in `requirements.txt`. Use Python 3.12 and open a terminal in this folder.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Or activate it in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Other operating systems have not been tested for this release. No LaTeX installation is needed to run the calculations or generate the PNGs.

## Reproduce the calculations

```bash
python run_replication.py
```

This runs all fourteen programs in dependency order. It writes numerical JSON records into `replication/` and execution logs into `outputs/calculation_logs/`. The runner prints the outcome of each program and stops on an error. Existing result records are overwritten when a calculation is rerun.

Keep `replication/overlap_design.json`: it is the one supplied scenario-input file, read by the overlapping-menu program. All other constructed inputs are embedded in the programs. Later calculations read some earlier generated records, so use the runner for a full reproduction.

## Generate the figures

After the calculations finish, run:

```bash
python run_figures.py
```

This produces `Pic1.png` through `Pic13.png` in the repository root, in manuscript order. It writes execution logs into `outputs/figure_logs/`. The figure scripts create no separate image-DPI or figure-verification reports.

The shared `figure_export.py` helper specifies **1200 dpi** and a minimum size of **8000 x 3000 pixels**, retaining the pixel minimum in the supplied LaTeX package's export helper. The existing one-point increase from the plotting scripts' base font sizes is retained. PNG density metadata may show a small rounding difference because the format stores density in integer pixels per metre.

Generation uses the mathematical formulas and calculation outputs directly. Figures are not reconstructed from screenshots or existing image files. Exporting these large PNGs requires more memory than running the numerical calculations. The scripts use a noninteractive plotting backend and do not require a display.

The generated filenames match the manuscript's LaTeX image references. Copy the desired PNGs beside the manuscript source when rebuilding the article. This folder does not rebuild or modify the manuscript PDF.

`.gitignore` excludes generated PNGs, result records, logs, and Python caches so they are not added to the repository through normal Git commands.

## Calculation programs and outputs

All programs below are in `replication/`. Their JSON outputs are created when the code runs; they are not supplied in this folder.

| Program | Research content | Generated record |
| --- | --- | --- |
| `verify_theory.py` | Scalar screening, payment identities, efficient pricing, constructed examples, and finite policy comparisons. | `screening_verification.json`, `policy_grid.json` |
| `mixed_portfolio_verification.py` | Mixed unit/bundled portfolios, rematching, and counterfactual opportunity prices. | `mixed_portfolio_results.json` |
| `pricing_stability.py verify` | Capacity-price stability and common-portfolio comparisons. | `pricing_stability_verification.json` |
| `recovery_certificates.py` | Admission protection, convex recovery alternatives, and pointwise menu-loss certificates. | `recovery_certificates.json` |
| `prior_sensitivity.py` | Continuous-prior financial outcomes for the 27 block settings. | `prior_sensitivity.json` |
| `sequential_pricing_verification.py` | Envelope charges preserving the sequential allocations. | `sequential_pricing_verification.json` |
| `final_adversarial_audit.py` | Additional independent enumeration of constructed cases. | `final_adversarial_audit.json` |
| `reoptimized_certificates.py` | Reoptimized certificates and the certificate-separation construction. | `reoptimized_certificates.json` |
| `menu_standardization.py` | Validated retention and unit-menu decisions in the 27 block settings. | `menu_standardization.json` |
| `fractional_capacity_protection.py` | Fractional-profile capacity and admission comparisons. | `fractional_capacity_protection.json` |
| `overlap_menu_decisions.py` | Overlapping-menu financial outcomes and certification decisions; reads `overlap_design.json`. | `overlap_menu_decisions.json` |
| `permission_design.py` | Permission selection, exact frontiers, necessity constructions, and recipient reversal. | `permission_design.json` |
| `check_richer_menus.py` | Subset enumeration and minimum richer menus in the unchanged 27-setting design. | `richer_menu_results.json` |
| `independent_menu_check.py` | Independent physical enumeration of compressed-menu values. | `independent_menu_checks.json` |

The optional `benchmark` action in `pricing_stability.py` is a separate timing experiment. It is not run by `run_replication.py` and is not needed for the manuscript's tables and figures.

## Locating the table values

This mapping follows the numbering in the supplied manuscript.

| Table | Manuscript label | Calculation source |
| --- | --- | --- |
| 1: Displacement example | `table3` | The analytical assignment instance and constructed-example checks in `verify_theory.py`. |
| 2: Two-applicant accounts | `tabTwoApplicant` | The analytical formulas and two-applicant checks in `verify_theory.py`, with illustrated parameters `a=3`, `b=1`, `h=4`. |
| 3: Policy financial outcomes | `tabPolicyFinancial` | `policy_grid.json`: `means`, using the financial entries in `metric_order`. |
| 4: Policy physical outcomes | `tabPolicyPhysical` | `policy_grid.json`: `means`, using the physical entries in `metric_order`. |
| 5: Unit-menu decisions | `tabMenuStandardization` | `menu_standardization.json`: `rows`, `display`, and `decision`. |
| 6: Overlapping menus | `tabOverlapMenus` | `overlap_menu_decisions.json`: `rows`. |
| 7: Minimum richer menus | `tabRicherMenus` | `richer_menu_results.json`: rows whose `original_unit_decision` is `below target`. |
| 8: Continuous-prior outcomes | `tabPriorSensitivity` | `prior_sensitivity.json`: all 27 `rows`. |
| 9: Pointwise certificates | `tabPointwiseCertificates` | `recovery_certificates.json`: `pointwise_study`. |
| 10: Finite verification coverage | `tabVerification` | Counts in the screening, mixed-portfolio, pricing-stability, recovery-certificate, prior-sensitivity, and sequential-pricing records. |

## Figure scripts

`run_figures.py` runs the following scripts in order. They can also be run individually once their calculation inputs exist.

| Generated figures | Script | Content |
| --- | --- | --- |
| 1-7 | `regenerate_figures.py` | Screening thresholds, financial formulas, welfare/utility envelopes, recovery profiles, displacement paths, and capacity-price comparisons. |
| 8-9 | `regenerate_recovery_figures.py` | Admission contraction and sharp menu-loss formulas. |
| 10 | `regenerate_certificate_refinement.py` | Restricted-price and reoptimized certificate comparisons. |
| 11 | `regenerate_menu_standardization.py` | Validated unit-menu retention intervals. |
| 12-13 | `regenerate_permission_figures.py` | Minimum richer menus, exact permission frontiers, and recipient reversal. |

Keep the plotting scripts, `figure_export.py`, and the `replication/` directory in their supplied relative locations.

## Interpreting the calculations

Exact fractions, integer arithmetic, and rationally checked optimization witnesses are used where stated in the programs. Continuous-prior intervals are deterministic expectation enclosures, not Monte Carlo confidence intervals. A failed financial certificate is distinguished from a menu that actually fails its financial target. Verification counts describe finite constructed checks, not firm observations or formal proof certification.

If you have permission or another applicable lawful basis to conduct a new experiment, work on a separate copy or branch and document changes to the constructed primitives. Cite the manuscript title and author above when using this research; citation does not replace any required permission. No repository address or publication DOI has been assigned by this prepared folder.
