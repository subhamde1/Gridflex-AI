"""Run the manuscript's fourteen calculation programs in dependency order."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PROGRAMS = [
    ["verify_theory.py"],
    ["mixed_portfolio_verification.py"],
    ["pricing_stability.py", "verify"],
    ["recovery_certificates.py"],
    ["prior_sensitivity.py"],
    ["sequential_pricing_verification.py"],
    ["final_adversarial_audit.py"],
    ["reoptimized_certificates.py"],
    ["menu_standardization.py"],
    ["fractional_capacity_protection.py"],
    ["overlap_menu_decisions.py"],
    ["permission_design.py"],
    ["check_richer_menus.py"],
    ["independent_menu_check.py"],
]


def main():
    logs = ROOT / "outputs/calculation_logs"
    logs.mkdir(parents=True, exist_ok=True)
    for program, *args in PROGRAMS:
        log = logs / (program + ".log")
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                [sys.executable, str(ROOT / "replication" / program), *args],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            )
        print(program, "passed" if result.returncode == 0 else "FAILED", flush=True)
        if result.returncode:
            print("See", log.relative_to(ROOT), flush=True)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
