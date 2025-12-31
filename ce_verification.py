#!/usr/bin/env python3
"""
Monte Carlo Verification of Certainty-Equivalence Result

This script:
1. Verifies E[W(gamma)] = W^det(gamma; alpha_bar) under baseline assumptions
2. Demonstrates CE breakdown when assumptions are violated
3. Provides numerical examples with realistic volatility

UPDATED: Figure styling for Journal of Public Economic Theory submission
- Removed in-figure titles (captions in LaTeX)
- Removed grid lines
- Serif fonts, frameless legends

USAGE:
    python ce_verification.py           # Full mode (~5-10 minutes)
    python ce_verification.py --fast    # Fast mode (~30-60 seconds)
"""

import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import pandas as pd
import time
import sys

# Apply journal styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'axes.linewidth': 0.8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'legend.frameon': False,
    'lines.linewidth': 1.5,
    'axes.grid': False,
    'mathtext.fontset': 'cm',
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

FIG_TWO_PANEL = (10, 4)

def save_figure(filename):
    """Save figure in PDF and PNG formats."""
    plt.savefig(f'{filename}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {filename}.pdf")

# =============================================================================
# CONFIGURATION
# =============================================================================

FAST_MODE = False

if '--fast' in sys.argv:
    FAST_MODE = True
    print("[INFO] Fast mode enabled via command line flag")

if FAST_MODE:
    N_SIMS_DEFAULT = 1000
    N_SIMS_REDUCED = 500
    N_GAMMA_POINTS = 12
    print(f"[INFO] FAST_MODE enabled: using {N_SIMS_DEFAULT} simulations (reduced accuracy)")
else:
    N_SIMS_DEFAULT = 10000
    N_SIMS_REDUCED = 5000
    N_GAMMA_POINTS = 23

np.random.seed(42)

# =============================================================================
# DETERMINISTIC BASELINE
# =============================================================================

def H1(x, T):
    """Geometric sum H1(x) = (1 - x^T) / (1 - x)."""
    if abs(1.0 - x) < 1e-12:
        return float(T)
    return (1.0 - x**T) / (1.0 - x)

def total_grants_deterministic(gamma, alpha, S, M0, T):
    """Deterministic total grants under constant payout."""
    if gamma <= 0 or gamma >= 1:
        return -np.inf
    
    A = alpha * (1 - gamma)
    B = S * (1 - gamma)
    
    if abs(1.0 - A) < 1e-10:
        gamma_c = 1.0 - 1.0 / alpha
        return ((alpha - 1.0) * (T * M0 + (S / alpha) * T * (T - 1) / 2.0)
                + gamma_c * S * T)
    
    H1A = H1(A, T)
    denom = (1.0 - A)
    term2 = 0.0 if abs(denom) < 1e-14 else (B / denom) * (T - H1A)
    return gamma * alpha * (M0 * H1A + term2) + gamma * S * T

def find_optimal_gamma_det(alpha, S, M0, T):
    """Find optimal constant payout rate (deterministic)."""
    res = minimize_scalar(
        lambda g: -total_grants_deterministic(g, alpha, S, M0, T),
        bounds=(1e-6, 1.0 - 1e-6),
        method='bounded'
    )
    return res.x

# =============================================================================
# STOCHASTIC SIMULATION
# =============================================================================

def simulate_stochastic_welfare(gamma, mean_alpha, sigma, S, M0, T, n_sims=10000,
                                 serial_corr=0.0, state_dependent=False, 
                                 concave_impact=False, crra_gamma=1.0):
    """Simulate welfare under stochastic returns."""
    mu = np.log(mean_alpha) - 0.5 * sigma**2
    
    all_W = np.zeros(n_sims)
    
    for sim in range(n_sims):
        M = M0
        W = 0.0
        prev_shock = 0.0
        
        for t in range(T):
            if serial_corr > 0:
                innovation = np.random.normal(0, sigma * np.sqrt(1 - serial_corr**2))
                shock = serial_corr * prev_shock + innovation
                prev_shock = shock
                r_t = mu + shock
            else:
                r_t = np.random.normal(mu, sigma)
            
            alpha_t = np.exp(r_t)
            
            if state_dependent:
                alpha_t = alpha_t * np.exp(-0.01 * max(0, M - 1))
            
            R_t = alpha_t * M + S
            C_t = gamma * R_t
            
            if concave_impact:
                if abs(crra_gamma - 1.0) < 1e-10:
                    impact = np.log(max(C_t, 1e-10))
                else:
                    impact = (max(C_t, 1e-10)**(1 - crra_gamma)) / (1 - crra_gamma)
            else:
                impact = C_t
            
            W += impact
            M = (1 - gamma) * R_t
        
        all_W[sim] = W
    
    return np.mean(all_W), np.std(all_W), all_W

# =============================================================================
# VERIFICATION TESTS
# =============================================================================

def test_certainty_equivalence():
    """Test 1: Verify E[W(gamma)] = W^det(gamma; alpha_bar) under baseline assumptions."""
    print("\n" + "="*70)
    print("TEST 1: CERTAINTY-EQUIVALENCE VERIFICATION")
    print("="*70)
    
    mean_alpha = 1.05
    sigma = 0.15
    S = 0.0
    M0 = 1.0
    T = 50
    n_sims = N_SIMS_DEFAULT
    
    gamma_grid = np.linspace(0.01, 0.15, N_GAMMA_POINTS)
    
    results = []
    for gamma in gamma_grid:
        W_det = total_grants_deterministic(gamma, mean_alpha, S, M0, T)
        W_stoch_mean, W_stoch_std, _ = simulate_stochastic_welfare(
            gamma, mean_alpha, sigma, S, M0, T, n_sims
        )
        se = W_stoch_std / np.sqrt(n_sims)
        rel_diff = 100 * (W_stoch_mean - W_det) / W_det
        
        results.append({
            'gamma': gamma * 100,
            'W_det': W_det,
            'W_stoch': W_stoch_mean,
            'W_stoch_se': se,
            'rel_diff_pct': rel_diff
        })
    
    df = pd.DataFrame(results)
    print(f"\nParameters: mean_alpha = {mean_alpha}, sigma = {sigma}, T = {T}, S = {S}")
    print(f"Number of simulations: {n_sims}")
    print("\n" + df.to_string(index=False))
    
    return df

def test_optimal_gamma_equivalence():
    """Test 2: Verify optimal gamma is same under deterministic and stochastic."""
    print("\n" + "="*70)
    print("TEST 2: OPTIMAL GAMMA EQUIVALENCE")
    print("="*70)
    
    mean_alpha = 1.05
    sigma = 0.15
    S = 0.0
    M0 = 1.0
    T = 50
    n_sims = N_SIMS_REDUCED
    
    gamma_det = find_optimal_gamma_det(mean_alpha, S, M0, T)
    
    gamma_grid = np.linspace(0.01, 0.10, 20)
    welfare = []
    for gamma in gamma_grid:
        W_mean, _, _ = simulate_stochastic_welfare(gamma, mean_alpha, sigma, S, M0, T, n_sims)
        welfare.append(W_mean)
    
    gamma_stoch = gamma_grid[np.argmax(welfare)]
    
    print(f"\nDeterministic optimal gamma: {gamma_det*100:.2f}%")
    print(f"Stochastic optimal gamma:    {gamma_stoch*100:.2f}%")
    print(f"Difference:                  {abs(gamma_det - gamma_stoch)*100:.3f} pp")
    
    return gamma_det, gamma_stoch

def test_ce_breakdown_serial_correlation():
    """Test 3: Show CE breaks with serial correlation."""
    print("\n" + "="*70)
    print("TEST 3: CE BREAKDOWN WITH SERIAL CORRELATION")
    print("="*70)
    
    mean_alpha = 1.05
    sigma = 0.15
    S = 0.0
    M0 = 1.0
    T = 50
    gamma = 0.035
    n_sims = N_SIMS_REDUCED
    
    W_det = total_grants_deterministic(gamma, mean_alpha, S, M0, T)
    
    rhos = [0.0, 0.2, 0.4, 0.6, 0.8]
    
    print(f"\nDeterministic welfare: W = {W_det:.4f}")
    print(f"\n{'Correlation':<15} {'E[W]':<12} {'Rel Diff':<12}")
    print("-" * 40)
    
    results = []
    for rho in rhos:
        W_mean, _, _ = simulate_stochastic_welfare(
            gamma, mean_alpha, sigma, S, M0, T, n_sims, serial_corr=rho
        )
        rel_diff = 100 * (W_mean - W_det) / W_det
        print(f"ρ = {rho:<10.1f} {W_mean:<12.4f} {rel_diff:<12.2f}%")
        results.append({'rho': rho, 'E_W': W_mean, 'rel_diff': rel_diff})
    
    return pd.DataFrame(results)

def test_ce_breakdown_concave_impact():
    """Test 4: Show CE breaks with concave impact."""
    print("\n" + "="*70)
    print("TEST 4: CE BREAKDOWN WITH CONCAVE IMPACT")
    print("="*70)
    
    mean_alpha = 1.05
    sigma = 0.15
    S = 0.0
    M0 = 1.0
    T = 50
    n_sims = N_SIMS_REDUCED
    
    gamma_grid = np.linspace(0.02, 0.10, 15)
    crra_values = [0, 0.5, 1.0, 2.0]
    
    print("\nOptimal gamma under different risk aversion levels:")
    print(f"{'CRRA coeff':<12} {'Optimal gamma':<15} {'Interpretation':<30}")
    print("-" * 60)
    
    for crra in crra_values:
        if crra == 0:
            gamma_opt = find_optimal_gamma_det(mean_alpha, S, M0, T)
            interp = "Risk neutral (linear impact)"
        else:
            welfare = []
            for gamma in gamma_grid:
                W_mean, _, _ = simulate_stochastic_welfare(
                    gamma, mean_alpha, sigma, S, M0, T, n_sims,
                    concave_impact=True, crra_gamma=crra
                )
                welfare.append(W_mean)
            gamma_opt = gamma_grid[np.argmax(welfare)]
            if crra == 0.5:
                interp = "Low risk aversion"
            elif crra == 1.0:
                interp = "Log utility"
            else:
                interp = "High risk aversion"
        
        print(f"{crra:<12.1f} {gamma_opt*100:<15.2f}% {interp:<30}")

def test_volatility_sensitivity():
    """Test 5: Show how volatility affects welfare (but not optimal gamma under CE)."""
    print("\n" + "="*70)
    print("TEST 5: VOLATILITY SENSITIVITY")
    print("="*70)
    
    mean_alpha = 1.05
    S = 0.0
    M0 = 1.0
    T = 50
    gamma = 0.035
    n_sims = N_SIMS_REDUCED
    
    sigmas = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    W_det = total_grants_deterministic(gamma, mean_alpha, S, M0, T)
    
    print(f"\nDeterministic welfare: W = {W_det:.4f}")
    print(f"\n{'Volatility':<12} {'E[W]':<12} {'Std[W]':<12} {'CV[W]':<12} {'Rel Diff':<12}")
    print("-" * 65)
    
    for sigma in sigmas:
        if sigma == 0:
            W_mean, W_std = W_det, 0.0
        else:
            W_mean, W_std, _ = simulate_stochastic_welfare(
                gamma, mean_alpha, sigma, S, M0, T, n_sims
            )
        
        cv = W_std / W_mean if W_mean > 0 else 0
        rel_diff = 100 * (W_mean - W_det) / W_det
        print(f"{sigma*100:<12.0f}% {W_mean:<12.4f} {W_std:<12.4f} {cv:<12.3f} {rel_diff:<12.2f}%")

def generate_verification_figure():
    """Generate figure showing CE verification and breakdown cases."""
    print("\n[INFO] Generating verification figure...")
    start_time = time.time()
    
    mean_alpha = 1.05
    sigma = 0.15
    S = 0.0
    M0 = 1.0
    T = 50
    n_sims = N_SIMS_REDUCED
    
    gamma_grid = np.linspace(0.01, 0.12, N_GAMMA_POINTS)
    
    W_det = [total_grants_deterministic(g, mean_alpha, S, M0, T) for g in gamma_grid]
    
    W_iid = []
    W_serial = []
    W_concave = []
    
    for gamma in gamma_grid:
        w, _, _ = simulate_stochastic_welfare(gamma, mean_alpha, sigma, S, M0, T, n_sims)
        W_iid.append(w)
        
        w, _, _ = simulate_stochastic_welfare(gamma, mean_alpha, sigma, S, M0, T, n_sims, 
                                               serial_corr=0.5)
        W_serial.append(w)
        
        w, _, _ = simulate_stochastic_welfare(gamma, mean_alpha, sigma, S, M0, T, n_sims,
                                               concave_impact=True, crra_gamma=1.0)
        W_concave.append(w)
    
    W_det = np.array(W_det)
    W_iid = np.array(W_iid)
    W_serial = np.array(W_serial)
    W_concave = np.array(W_concave)
    
    fig, axes = plt.subplots(1, 2, figsize=FIG_TWO_PANEL)
    
    # Panel A: CE verification (i.i.d. case)
    ax1 = axes[0]
    ax1.plot(gamma_grid * 100, W_det, color='#1f77b4', linewidth=1.5, 
             label='Deterministic $W^{det}(\\gamma; \\bar\\alpha)$')
    ax1.plot(gamma_grid * 100, W_iid, '--', color='#d62728', linewidth=1.5, 
             label='Stochastic $\\mathbb{E}[W(\\gamma)]$ (i.i.d.)')
    ax1.set_xlabel('Payout rate $\\gamma$ (\\%)')
    ax1.set_ylabel('Welfare')
    ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes, fontsize=11,
             fontweight='bold', va='top', ha='left')
    ax1.legend(fontsize=9, loc='upper right', frameon=False)
    # NO grid
    
    # Panel B: CE breakdown cases
    ax2 = axes[1]
    ax2.plot(gamma_grid * 100, W_det / W_det.max(), color='#1f77b4', linewidth=1.5, 
             label='Deterministic (baseline)')
    ax2.plot(gamma_grid * 100, W_serial / W_serial.max(), '--', color='#2ca02c', linewidth=1.5, 
             label='Serial correlation ($\\rho=0.5$)')
    ax2.plot(gamma_grid * 100, W_concave / W_concave.max(), ':', color='#9467bd', linewidth=2, 
             label='Concave impact (log utility)')
    ax2.set_xlabel('Payout rate $\\gamma$ (\\%)')
    ax2.set_ylabel('Normalized welfare')
    ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes, fontsize=11,
             fontweight='bold', va='top', ha='left')
    ax2.legend(fontsize=9, loc='lower left', frameon=False)
    # NO grid
    
    plt.tight_layout()
    save_figure('figure_ce_verification')
    
    elapsed = time.time() - start_time
    print(f"[OK] Generated figure (elapsed: {elapsed:.1f}s)")

def generate_latex_table():
    """Generate LaTeX table for CE verification."""
    mean_alpha = 1.05
    sigma = 0.15
    S = 0.0
    M0 = 1.0
    T = 50
    n_sims = N_SIMS_DEFAULT
    
    gamma_values = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
    
    latex = r"""\begin{table}[htbp]
\centering
\caption{Monte Carlo verification of certainty equivalence ($\bar\alpha = 1.05$, $\sigma = 15\%$, $T = 50$)}
\label{tab:ce-verification}
\begin{tabular}{ccccc}
\toprule
$\gamma$ & $W^{det}(\gamma; \bar\alpha)$ & $\mathbb{E}[W(\gamma)]$ & Std. Error & Rel. Diff. \\
\midrule
"""
    
    for gamma in gamma_values:
        W_det = total_grants_deterministic(gamma, mean_alpha, S, M0, T)
        W_mean, W_std, _ = simulate_stochastic_welfare(gamma, mean_alpha, sigma, S, M0, T, n_sims)
        se = W_std / np.sqrt(n_sims)
        rel_diff = 100 * (W_mean - W_det) / W_det
        
        latex += f"{gamma*100:.0f}\\% & {W_det:.3f} & {W_mean:.3f} & {se:.4f} & {rel_diff:+.2f}\\% \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}

\smallskip
\emph{Notes}: $W^{det}$ is deterministic welfare; $\mathbb{E}[W]$ is the Monte Carlo mean. Relative differences are within simulation noise, confirming CE.
\end{table}
"""
    
    with open('table_ce_verification.tex', 'w') as f:
        f.write(latex)
    
    print("[OK] Saved: table_ce_verification.tex")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    overall_start = time.time()
    
    print("="*70)
    print("MONTE CARLO VERIFICATION OF CERTAINTY-EQUIVALENCE RESULT")
    print("="*70)
    if FAST_MODE:
        print("[INFO] Running in FAST MODE - reduced accuracy for quick testing")
    else:
        print("[INFO] Running in FULL MODE - this may take 5-10 minutes")
    
    df1 = test_certainty_equivalence()
    gamma_det, gamma_stoch = test_optimal_gamma_equivalence()
    df3 = test_ce_breakdown_serial_correlation()
    test_ce_breakdown_concave_impact()
    test_volatility_sensitivity()
    
    print("\n### GENERATING OUTPUTS ###")
    generate_verification_figure()
    generate_latex_table()
    
    overall_elapsed = time.time() - overall_start
    print("\n" + "="*70)
    print(f"[OK] All verification tests complete. Total time: {overall_elapsed:.1f}s")
    print("="*70)
