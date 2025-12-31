#!/usr/bin/env python3
"""
CES/CRRA Robustness Analysis

Analyzes how optimal payout rates change under different welfare criteria:
- Linear impact (baseline): W = sum C_t
- CES/CRRA impact: W = sum u(C_t) where u(C) = C^(1-sigma)/(1-sigma)

Key finding: Comparative statics in T and alpha are preserved under concave impact,
but optimal gamma* increases (smoothing motive pushes toward higher current spending).

UPDATED: Figure styling for Journal of Public Economic Theory submission
- Removed in-figure titles (captions in LaTeX)
- Removed grid lines
- Serif fonts, frameless legends
"""

import numpy as np
from scipy.optimize import minimize_scalar
import pandas as pd
import matplotlib.pyplot as plt

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

FIG_SINGLE = (6.5, 4.5)

def save_figure(filename):
    """Save figure in PDF and PNG formats."""
    plt.savefig(f'{filename}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {filename}.pdf")

# =============================================================================
# SIMULATION-BASED WELFARE COMPUTATION
# =============================================================================

def simulate_welfare_crra(gamma, alpha, S, M0, T, sigma):
    """
    Compute welfare under CRRA utility: u(C) = C^(1-sigma)/(1-sigma) for sigma != 1
                                        u(C) = log(C) for sigma = 1
    """
    if gamma <= 0 or gamma >= 1:
        return -np.inf
    
    M = M0
    W = 0.0
    
    for t in range(T):
        R_t = alpha * M + S
        C_t = gamma * R_t
        C_t = max(C_t, 1e-10)
        
        if abs(sigma - 1.0) < 1e-10:
            u_t = np.log(C_t)
        elif sigma == 0:
            u_t = C_t
        else:
            u_t = (C_t ** (1 - sigma)) / (1 - sigma)
        
        W += u_t
        M = (1 - gamma) * R_t
    
    return W

def find_optimal_gamma_crra(alpha, S, M0, T, sigma):
    """Find optimal constant payout rate under CRRA utility."""
    res = minimize_scalar(
        lambda g: -simulate_welfare_crra(g, alpha, S, M0, T, sigma),
        bounds=(1e-4, 1.0 - 1e-4),
        method='bounded'
    )
    return res.x

# =============================================================================
# BASELINE (LINEAR) FOR COMPARISON
# =============================================================================

def H1(x, T):
    """Geometric sum."""
    if abs(1.0 - x) < 1e-12:
        return float(T)
    return (1.0 - x**T) / (1.0 - x)

def total_grants_linear(gamma, alpha, S, M0, T):
    """Total grants under linear impact (baseline)."""
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

def find_optimal_gamma_linear(alpha, S, M0, T):
    """Find optimal constant payout rate under linear impact."""
    res = minimize_scalar(
        lambda g: -total_grants_linear(g, alpha, S, M0, T),
        bounds=(1e-6, 1.0 - 1e-6),
        method='bounded'
    )
    return res.x

# =============================================================================
# ANALYSIS
# =============================================================================

def generate_crra_sensitivity_table():
    """Generate table showing optimal gamma* for different sigma values."""
    alphas = [1.03, 1.05, 1.07]
    horizons = [20, 50, 100]
    sigmas = [0.0, 0.5, 1.0, 2.0]
    M0, S = 1.0, 0.0
    
    results = []
    
    for alpha in alphas:
        r_pct = (alpha - 1) * 100
        for T in horizons:
            row = {'r': f'{r_pct:.0f}%', 'T': T}
            for sigma in sigmas:
                if sigma == 0:
                    gamma_opt = find_optimal_gamma_linear(alpha, S, M0, T) * 100
                else:
                    gamma_opt = find_optimal_gamma_crra(alpha, S, M0, T, sigma) * 100
                row[f'sigma={sigma}'] = gamma_opt
            results.append(row)
    
    df = pd.DataFrame(results)
    return df

def generate_latex_table():
    """Generate LaTeX table for CRRA sensitivity analysis."""
    alphas = [1.03, 1.05, 1.07]
    horizons = [20, 50, 100]
    sigmas = [0.0, 0.5, 1.0, 2.0]
    M0, S = 1.0, 0.0
    
    latex = r"""\begin{table}[htbp]
\centering
\caption{Optimal payout under different risk aversion levels ($S=0$, $M_0=1$)}
\label{tab:crra-sensitivity}
\begin{tabular}{llcccc}
\toprule
& & \multicolumn{4}{c}{\textbf{CRRA coefficient $\sigma$}} \\
\cmidrule(lr){3-6}
\textbf{Return $r$} & \textbf{Horizon $T$} & $\sigma=0$ & $\sigma=0.5$ & $\sigma=1$ & $\sigma=2$ \\
& & (linear) & (low RA) & (log) & (high RA) \\
\midrule
"""
    
    for alpha in alphas:
        r_pct = (alpha - 1) * 100
        for i, T in enumerate(horizons):
            if i == 0:
                row_start = f"{r_pct:.0f}\\%"
            else:
                row_start = ""
            
            gammas = []
            for sigma in sigmas:
                if sigma == 0:
                    g = find_optimal_gamma_linear(alpha, S, M0, T) * 100
                else:
                    g = find_optimal_gamma_crra(alpha, S, M0, T, sigma) * 100
                gammas.append(f"{g:.2f}\\%")
            
            latex += f"{row_start} & {T} & {' & '.join(gammas)} \\\\\n"
        
        if alpha != alphas[-1]:
            latex += "\\addlinespace\n"
    
    latex += r"""\bottomrule
\end{tabular}

\smallskip
\emph{Notes}: $\sigma = 0$ corresponds to linear (risk-neutral) impact; $\sigma = 1$ is log utility; higher $\sigma$ indicates greater risk aversion.
\end{table}
"""
    
    with open('table_crra_sensitivity.tex', 'w') as f:
        f.write(latex)
    
    print("[OK] Saved: table_crra_sensitivity.tex")
    return latex

def verify_comparative_statics():
    """Verify that comparative statics in T and alpha are preserved under CRRA."""
    print("\n" + "="*70)
    print("VERIFICATION: COMPARATIVE STATICS UNDER CRRA")
    print("="*70)
    
    M0, S = 1.0, 0.0
    sigmas = [0.0, 0.5, 1.0, 2.0]
    
    print("\n1. MONOTONICITY IN T (fixing alpha = 1.05)")
    print("-" * 50)
    alpha = 1.05
    horizons = [20, 50, 100, 200]
    
    for sigma in sigmas:
        sigma_label = "linear" if sigma == 0 else f"sigma={sigma}"
        gammas = []
        for T in horizons:
            if sigma == 0:
                g = find_optimal_gamma_linear(alpha, S, M0, T)
            else:
                g = find_optimal_gamma_crra(alpha, S, M0, T, sigma)
            gammas.append(g * 100)
        
        monotone = all(gammas[i] > gammas[i+1] for i in range(len(gammas)-1))
        status = "[OK] decreasing" if monotone else "[X] NOT monotone"
        print(f"  {sigma_label:10s}: T=20->{gammas[0]:.2f}%, T=50->{gammas[1]:.2f}%, "
              f"T=100->{gammas[2]:.2f}%, T=200->{gammas[3]:.2f}%  {status}")
    
    print("\n2. MONOTONICITY IN alpha (fixing T = 50)")
    print("-" * 50)
    T = 50
    alphas = [1.03, 1.05, 1.07, 1.10]
    
    for sigma in sigmas:
        sigma_label = "linear" if sigma == 0 else f"sigma={sigma}"
        gammas = []
        for alpha in alphas:
            if sigma == 0:
                g = find_optimal_gamma_linear(alpha, S, M0, T)
            else:
                g = find_optimal_gamma_crra(alpha, S, M0, T, sigma)
            gammas.append(g * 100)
        
        monotone = all(gammas[i] > gammas[i+1] for i in range(len(gammas)-1))
        status = "[OK] decreasing" if monotone else "[X] NOT monotone"
        print(f"  {sigma_label:10s}: alpha=1.03->{gammas[0]:.2f}%, alpha=1.05->{gammas[1]:.2f}%, "
              f"alpha=1.07->{gammas[2]:.2f}%, alpha=1.10->{gammas[3]:.2f}%  {status}")

def plot_crra_sensitivity():
    """
    Plot optimal gamma* vs horizon for different sigma values.
    Uses a finer grid for smoother curves.
    """
    alpha = 1.05
    M0, S = 1.0, 0.0
    
    # Finer grid for smoother curves
    T_grid_fine = np.arange(10, 50, 2)
    T_grid_coarse = np.arange(50, 151, 5)
    T_grid = np.concatenate([T_grid_fine, T_grid_coarse])
    
    sigmas = [0.0, 0.5, 1.0, 2.0]
    labels = [r'Linear ($\sigma=0$)', r'Low RA ($\sigma=0.5$)', 
              r'Log ($\sigma=1$)', r'High RA ($\sigma=2$)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    plt.figure(figsize=FIG_SINGLE)
    
    for sigma, label, color in zip(sigmas, labels, colors):
        gammas = []
        for T in T_grid:
            if sigma == 0:
                g = find_optimal_gamma_linear(alpha, S, M0, T)
            else:
                g = find_optimal_gamma_crra(alpha, S, M0, T, sigma)
            gammas.append(g * 100)
        plt.plot(T_grid, gammas, linewidth=1.5, label=label, color=color)
    
    # Add critical threshold
    gamma_c = (1 - 1/alpha) * 100
    plt.axhline(gamma_c, linestyle=':', color='gray', linewidth=1.0, alpha=0.7,
                label=f'$\\gamma_c = {gamma_c:.2f}\\%$')
    
    plt.xlabel('Planning horizon $T$ (years)')
    plt.ylabel('Optimal payout rate $\\gamma^*$ (%)')
    # NO title - caption handles this
    plt.legend(fontsize=9, loc='upper right', frameon=False)
    # NO grid
    plt.xlim(10, 150)
    plt.ylim(0, 25)
    plt.tight_layout()
    
    save_figure('figure_crra_sensitivity')

def analyze_smoothing_effect():
    """Analyze how much higher gamma* is under concave vs. linear impact."""
    print("\n" + "="*70)
    print("SMOOTHING EFFECT: INCREASE IN gamma* FROM LINEAR TO CRRA")
    print("="*70)
    
    alpha = 1.05
    M0, S = 1.0, 0.0
    horizons = [20, 50, 100]
    
    print(f"\nParameters: alpha = {alpha}, S = {S}, M0 = {M0}")
    print(f"\n{'T':<8} {'Linear':<12} {'sigma=0.5':<12} {'sigma=1':<12} {'sigma=2':<12}")
    print("-" * 55)
    
    for T in horizons:
        g_linear = find_optimal_gamma_linear(alpha, S, M0, T) * 100
        g_05 = find_optimal_gamma_crra(alpha, S, M0, T, 0.5) * 100
        g_1 = find_optimal_gamma_crra(alpha, S, M0, T, 1.0) * 100
        g_2 = find_optimal_gamma_crra(alpha, S, M0, T, 2.0) * 100
        
        print(f"{T:<8} {g_linear:<12.2f} {g_05:<12.2f} {g_1:<12.2f} {g_2:<12.2f}")
    
    print("\nPercentage increase from linear baseline:")
    print(f"{'T':<8} {'sigma=0.5':<12} {'sigma=1':<12} {'sigma=2':<12}")
    print("-" * 45)
    
    for T in horizons:
        g_linear = find_optimal_gamma_linear(alpha, S, M0, T)
        g_05 = find_optimal_gamma_crra(alpha, S, M0, T, 0.5)
        g_1 = find_optimal_gamma_crra(alpha, S, M0, T, 1.0)
        g_2 = find_optimal_gamma_crra(alpha, S, M0, T, 2.0)
        
        inc_05 = (g_05 - g_linear) / g_linear * 100
        inc_1 = (g_1 - g_linear) / g_linear * 100
        inc_2 = (g_2 - g_linear) / g_linear * 100
        
        print(f"{T:<8} {inc_05:<12.1f}% {inc_1:<12.1f}% {inc_2:<12.1f}%")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("CES/CRRA ROBUSTNESS ANALYSIS")
    print("="*70)
    
    print("\n### SENSITIVITY TABLE ###")
    df = generate_crra_sensitivity_table()
    print(df.to_string(index=False))
    
    print("\n### GENERATING LATEX TABLE ###")
    generate_latex_table()
    
    verify_comparative_statics()
    analyze_smoothing_effect()
    
    print("\n### GENERATING FIGURE ###")
    plot_crra_sensitivity()
    
    print("\n" + "="*70)
    print("[OK] CES/CRRA analysis complete.")
    print("="*70)
