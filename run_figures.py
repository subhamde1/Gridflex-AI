"""Generate all thirteen manuscript PNGs after running the calculations."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PROGRAMS = [
    "regenerate_figures.py",
    "regenerate_recovery_figures.py",
    "regenerate_certificate_refinement.py",
    "regenerate_menu_standardization.py",
    "regenerate_permission_figures.py",
]
REQUIRED_RESULTS = [
    "mixed_portfolio_results.json", "pricing_stability_verification.json",
    "reoptimized_certificates.json", "menu_standardization.json",
    "richer_menu_results.json", "permission_design.json",
]


def main():
    if any(not (ROOT / "replication" / name).is_file() for name in REQUIRED_RESULTS):
        print("Run python run_replication.py before generating the figures.", file=sys.stderr)
        return 1
    logs = ROOT / "outputs/figure_logs"
    logs.mkdir(parents=True, exist_ok=True)
    for program in PROGRAMS:
        log = logs / (program + ".log")
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                [sys.executable, str(ROOT / program)], cwd=ROOT,
                stdout=stream, stderr=subprocess.STDOUT,
            )
        print(program, "passed" if result.returncode == 0 else "FAILED", flush=True)
        if result.returncode:
            print("See", log.relative_to(ROOT), flush=True)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
