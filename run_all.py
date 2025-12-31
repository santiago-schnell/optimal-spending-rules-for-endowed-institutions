#!/usr/bin/env python3
"""
Master Execution Script: Optimal Spending Rules for Endowed Institutions

This script runs all analysis scripts in sequence and generates all tables and figures
for the paper. Use this to reproduce all results from scratch.

USAGE:
    python run_all.py              # Full analysis (recommended, ~15-20 minutes)
    python run_all.py --fast       # Quick verification (~2-3 minutes)
    python run_all.py --figures    # Generate figures only

OUTPUT:
    - All figures saved to the script directory as PDF files
    - Console output shows all table values

AUTHORS: Evgeny Havkin and Santiago Schnell
DATE: December 2025
"""

import subprocess
import sys
import time
import os
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# List of scripts to run in order
SCRIPTS = [
    ("calibration_analysis_fixed.py", "Baseline calibration and Tables 1-2"),
    ("robustness_analysis.py", "Multi-specification robustness, Table 3"),
    ("crra_analysis.py", "CRRA robustness analysis and Table 4"),
    ("welfare_loss_analysis.py", "Welfare loss calculations and Table 5"),
    ("welfare_loss_sensitivity_v2.py", "Welfare loss sensitivity, Table 6"),
    ("generate_welfare_loss_figure.py", "Generate welfare loss sensitivity figure"),
    ("endowment_tax_analysis.py", "OBBBA tax analysis and Tables 7-8"),
    ("ce_verification.py", "Certainty equivalence verification"),
]

# Fast mode flag
FAST_MODE = '--fast' in sys.argv
FIGURES_ONLY = '--figures' in sys.argv

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_header(text):
    """Print a formatted header."""
    width = 70
    print("\n" + "=" * width)
    print(f" {text}")
    print("=" * width)

def run_script(script_name, description):
    """Run a single Python script and capture output."""
    print(f"\n>>> Running: {script_name}")
    print(f"    ({description})")
    
    start_time = time.time()
    
    # Build full path to script
    script_path = os.path.join(SCRIPT_DIR, script_name)
    
    # Check if script exists
    if not os.path.exists(script_path):
        print(f"    ✗ Script not found: {script_path}")
        return False, "File not found", 0
    
    # Build command
    cmd = [sys.executable, script_path]
    if FAST_MODE and script_name == 'ce_verification.py':
        cmd.append('--fast')
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per script
            cwd=SCRIPT_DIR  # Run from the script directory
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"    ✓ Completed in {elapsed:.1f}s")
            return True, result.stdout, elapsed
        else:
            print(f"    ✗ Failed (exit code {result.returncode})")
            # Show both stdout and stderr for debugging
            if result.stdout and result.stdout.strip():
                stdout_tail = result.stdout[-500:] if len(result.stdout) > 500 else result.stdout
                print(f"    STDOUT (last 500 chars):\n{stdout_tail}")
            if result.stderr and result.stderr.strip():
                stderr_tail = result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
                print(f"    STDERR (last 500 chars):\n{stderr_tail}")
            if not result.stdout and not result.stderr:
                print("    (No output captured - script may have crashed immediately)")
            return False, result.stderr or result.stdout or "Unknown error", elapsed
            
    except subprocess.TimeoutExpired:
        print(f"    ✗ Timed out after 600s")
        return False, "Timeout", 600
    except FileNotFoundError:
        print(f"    ✗ Script not found: {script_name}")
        return False, "File not found", 0

def check_outputs():
    """Check that expected output files were created."""
    expected_figures = [
        "figure_gamma_vs_T.pdf",
        "figure_total_vs_gamma.pdf",
        "figure_crra_sensitivity.pdf",
        "figure_welfare_loss.pdf",
        "figure_welfare_loss_sensitivity.pdf",
        "figure_endowment_tax.pdf",
        "figure_endowment_tax_detail.pdf",
        "figure_ce_verification.pdf",
        "figure_inflow_sensitivity.pdf",
        "figure_policy_comparison.pdf",
    ]
    
    print(f"\n>>> Checking output files in: {SCRIPT_DIR}")
    missing = []
    for fig in expected_figures:
        fig_path = os.path.join(SCRIPT_DIR, fig)
        if os.path.exists(fig_path):
            print(f"    ✓ {fig}")
        else:
            print(f"    ✗ {fig} (missing)")
            missing.append(fig)
    
    return missing

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run all analyses."""
    
    print_header("MASTER EXECUTION SCRIPT")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Script directory: {SCRIPT_DIR}")
    print(f"Mode: {'FAST (reduced accuracy)' if FAST_MODE else 'FULL (publication quality)'}")
    
    # Track results
    results = []
    total_time = 0
    
    # Run each script
    for script_name, description in SCRIPTS:
        if FIGURES_ONLY and 'verification' in script_name:
            print(f"\n>>> Skipping: {script_name} (figures-only mode)")
            continue
            
        success, output, elapsed = run_script(script_name, description)
        results.append((script_name, success, elapsed))
        total_time += elapsed
    
    # Summary
    print_header("EXECUTION SUMMARY")
    
    n_success = sum(1 for _, success, _ in results if success)
    n_total = len(results)
    
    print(f"\nScripts completed: {n_success}/{n_total}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    for script_name, success, elapsed in results:
        status = "✓" if success else "✗"
        print(f"  {status} {script_name}: {elapsed:.1f}s")
    
    # Check outputs
    missing = check_outputs()
    
    if missing:
        print(f"\n⚠ Warning: {len(missing)} expected figure(s) not found")
    else:
        print("\n✓ All expected figures generated successfully")
    
    # Final status
    print_header("DONE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if n_success == n_total and not missing:
        print("\n✓ All analyses completed successfully!")
        print("  Results are ready for inclusion in the paper.")
        return 0
    else:
        print("\n⚠ Some analyses failed or outputs are missing.")
        print("  Please check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
