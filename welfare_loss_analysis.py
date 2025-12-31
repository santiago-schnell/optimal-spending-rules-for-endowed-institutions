#!/usr/bin/env python3
"""
Welfare Loss Analysis: Constant-Rate vs. Time-Varying Optimal Policies

This script computes:
1. Optimal time-varying policy {gamma_t} via backward induction
2. Welfare under optimal constant rate gamma*
3. Welfare gap between the two approaches
4. Sensitivity analysis across parameters

Key insight: Under linear impact and no discounting, the optimal time-varying
policy is "bang-bang": gamma_t = 0 for t < T-1, gamma_{T-1} = 1.

UPDATED: Figure styling for Journal of Public Economic Theory submission
- Removed in-figure titles (captions in LaTeX)
- Removed grid lines
- Serif fonts, frameless legends
"""

import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import pandas as pd

# Import shared styling (or define inline if module not available)
try:
    from figure_style import apply_journal_style, save_figure, COLORS, FIG_SINGLE, FIG_TWO_PANEL
    apply_journal_style()
except ImportError:
    # Fallback: apply styling inline
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.linewidth': 0.8,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'legend.frameon': False,
        'lines.linewidth': 1.5,
        'axes.grid': False,
        'mathtext.fontset': 'cm',
    })
    COLORS = {'blue': '#1f77b4', 'orange': '#ff7f0e', 'green': '#2ca02c', 'red': '#d62728'}
    FIG_SINGLE = (6.5, 4.5)
    FIG_TWO_PANEL = (10, 4)
    def save_figure(fn, formats=['pdf', 'png']):
        for fmt in formats:
            plt.savefig(f'{fn}.{fmt}', dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {fn}.pdf")

# =============================================================================
# CORE FUNCTIONS (from calibration_analysis)
# =============================================================================

def H1(x, T):
    """Geometric sum H1(x) = (1 - x^T) / (1 - x) with robust x->1 limit."""
    if abs(1.0 - x) < 1e-12:
        return float(T)
    return (1.0 - x**T) / (1.0 - x)

def total_grants_constant(gamma, alpha, S, M0, T):
    """
    Total grants under constant payout rate gamma.
    W(gamma) = sum_{t=0}^{T-1} C_t where C_t = gamma * (alpha * M_t + S)
    """
    if gamma <= 0 or gamma >= 1:
        return -np.inf
    
    A = alpha * (1 - gamma)
    B = S * (1 - gamma)
    
    if abs(1.0 - A) < 1e-10:
        # Critical case: A = 1
        gamma_c = 1.0 - 1.0 / alpha
        return ((alpha - 1.0) * (T * M0 + (S / alpha) * T * (T - 1) / 2.0)
                + gamma_c * S * T)
    
    H1A = H1(A, T)
    denom = (1.0 - A)
    term2 = 0.0 if abs(denom) < 1e-14 else (B / denom) * (T - H1A)
    return gamma * alpha * (M0 * H1A + term2) + gamma * S * T

def find_optimal_gamma(alpha, S, M0, T):
    """Find optimal constant payout rate."""
    res = minimize_scalar(
        lambda g: -total_grants_constant(g, alpha, S, M0, T),
        bounds=(1e-6, 1.0 - 1e-6),
        method='bounded'
    )
    return res.x

# =============================================================================
# TIME-VARYING OPTIMAL POLICY
# =============================================================================

def optimal_time_varying_policy(alpha, S, M0, T):
    """
    Compute the optimal time-varying policy via backward induction.
    
    Returns:
        gamma_opt: array of optimal payout rates [gamma_0, ..., gamma_{T-1}]
        W_opt: total welfare under optimal policy
        M_path: capital path [M_0, ..., M_{T-1}]
        C_path: grant path [C_0, ..., C_{T-1}]
    """
    gamma_opt = np.zeros(T)
    gamma_opt[T-1] = 1.0  # Spend everything in final period
    
    # Simulate the capital and grant paths
    M_path = np.zeros(T)
    C_path = np.zeros(T)
    
    M_path[0] = M0
    for t in range(T):
        R_t = alpha * M_path[t] + S  # Available resources
        C_path[t] = gamma_opt[t] * R_t
        if t < T - 1:
            M_path[t+1] = (1 - gamma_opt[t]) * R_t
    
    W_opt = np.sum(C_path)
    
    return gamma_opt, W_opt, M_path, C_path

def optimal_time_varying_closed_form(alpha, S, M0, T):
    """
    Closed-form solution for optimal time-varying welfare.
    
    With gamma_t = 0 for t < T-1 and gamma_{T-1} = 1:
    W* = alpha^T * M0 + S * H1(alpha, T)
    """
    if abs(alpha - 1.0) < 1e-12:
        return M0 + S * T
    
    W_star = (alpha ** T) * M0 + S * H1(alpha, T)
    return W_star

# =============================================================================
# WELFARE LOSS COMPUTATION
# =============================================================================

def compute_welfare_loss(alpha, S, M0, T):
    """
    Compute welfare loss from constant-rate restriction.
    
    Returns:
        gamma_star: optimal constant rate
        W_constant: welfare under optimal constant rate
        W_varying: welfare under optimal time-varying policy
        loss_abs: absolute welfare loss (W_varying - W_constant)
        loss_pct: percentage welfare loss
    """
    gamma_star = find_optimal_gamma(alpha, S, M0, T)
    W_constant = total_grants_constant(gamma_star, alpha, S, M0, T)
    W_varying = optimal_time_varying_closed_form(alpha, S, M0, T)
    
    loss_abs = W_varying - W_constant
    loss_pct = 100 * loss_abs / W_varying if W_varying > 0 else 0
    
    return gamma_star, W_constant, W_varying, loss_abs, loss_pct

# =============================================================================
# ANALYSIS AND TABLES
# =============================================================================

def generate_welfare_loss_table():
    """Generate table of welfare losses across calibrations."""
    returns = [0.03, 0.05, 0.07]
    horizons = [20, 50, 100, 200]
    M0, S = 1.0, 0.0
    
    results = []
    for r in returns:
        alpha = 1.0 + r
        for T in horizons:
            gamma_star, W_const, W_vary, loss_abs, loss_pct = compute_welfare_loss(alpha, S, M0, T)
            results.append({
                'r': f'{int(r*100)}%',
                'alpha': alpha,
                'T': T,
                'gamma_star': gamma_star * 100,
                'W_constant': W_const,
                'W_varying': W_vary,
                'loss_pct': loss_pct
            })
    
    df = pd.DataFrame(results)
    return df

def generate_welfare_loss_latex_table():
    """Generate LaTeX table for welfare losses."""
    returns = [0.03, 0.05, 0.07]
    horizons = [20, 50, 100, 200]
    M0, S = 1.0, 0.0
    
    rows = []
    for r in returns:
        alpha = 1.0 + r
        row = {'Return': f'{int(r*100)}\\%'}
        for T in horizons:
            _, _, _, _, loss_pct = compute_welfare_loss(alpha, S, M0, T)
            row[f'T={T}'] = f'{loss_pct:.1f}\\%'
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    latex = r"""\begin{table}[htbp]
\centering
\caption{Welfare loss from constant-rate restriction ($S=0$, $M_0=1$)}
\label{tab:welfare-loss-constant}
\begin{tabular}{lcccc}
\toprule
\textbf{Real return $r$} & $\boldsymbol{T=20}$ & $\boldsymbol{T=50}$ & $\boldsymbol{T=100}$ & $\boldsymbol{T=200}$ \\
\midrule
"""
    for _, row in df.iterrows():
        latex += f"{row['Return']} & {row['T=20']} & {row['T=50']} & {row['T=100']} & {row['T=200']} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}

\smallskip
\emph{Notes}: Welfare loss is measured as $(W^* - W(\gamma^*))/W^*$, where $W^*$ is welfare under the optimal time-varying policy and $W(\gamma^*)$ is welfare under the optimal constant rate. The optimal time-varying policy sets $\gamma_t = 0$ for $t < T-1$ and $\gamma_{T-1} = 1$ (full investment followed by complete liquidation).
\end{table}
"""
    
    with open('table_welfare_loss.tex', 'w') as f:
        f.write(latex)
    
    print("[OK] Saved: table_welfare_loss.tex")
    return df

def plot_welfare_loss_vs_horizon():
    """Plot welfare loss as function of horizon for different return assumptions."""
    alphas = [1.03, 1.05, 1.07]
    T_grid = np.arange(5, 201, 5)
    M0, S = 1.0, 0.0
    
    plt.figure(figsize=FIG_SINGLE)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for alpha, color in zip(alphas, colors):
        r_pct = (alpha - 1.0) * 100
        losses = []
        for T in T_grid:
            _, _, _, _, loss_pct = compute_welfare_loss(alpha, S, M0, T)
            losses.append(loss_pct)
        plt.plot(T_grid, losses, linewidth=1.5, color=color, label=f'$r = {r_pct:.0f}\\%$')
    
    plt.xlabel('Planning horizon $T$ (years)')
    plt.ylabel('Welfare loss (%)')
    # NO title - caption handles this
    plt.legend(loc='lower right', frameon=False)
    # NO grid
    plt.xlim(5, 200)
    plt.ylim(0, None)
    plt.tight_layout()
    
    save_figure('figure_welfare_loss')
    plt.close()

def plot_policy_comparison():
    """Plot comparing grant paths under constant vs. time-varying policies."""
    alpha, S, M0, T = 1.05, 0.0, 1.0, 50
    
    gamma_star = find_optimal_gamma(alpha, S, M0, T)
    
    # Simulate constant policy
    M_const = np.zeros(T)
    C_const = np.zeros(T)
    M_const[0] = M0
    for t in range(T):
        R_t = alpha * M_const[t] + S
        C_const[t] = gamma_star * R_t
        if t < T - 1:
            M_const[t+1] = (1 - gamma_star) * R_t
    
    # Optimal time-varying policy
    gamma_opt, W_opt, M_vary, C_vary = optimal_time_varying_policy(alpha, S, M0, T)
    
    fig, axes = plt.subplots(1, 2, figsize=FIG_TWO_PANEL)
    
    # Panel A: Grant paths
    ax1 = axes[0]
    t_grid = np.arange(T)
    ax1.plot(t_grid, C_const, linewidth=1.5, color='#1f77b4', 
             label=f'Constant $\\gamma^* = {gamma_star*100:.1f}\\%$')
    ax1.bar(t_grid, C_vary, alpha=0.5, color='#ff7f0e', label='Time-varying (bang-bang)')
    ax1.set_xlabel('Period $t$')
    ax1.set_ylabel('Grants $C_t$')
    ax1.set_ylim(0, 0.1)  # <-- ADD THIS LINE
    ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes, fontsize=11, 
             fontweight='bold', va='top', ha='left')
    ax1.legend(fontsize=9, loc='lower left', frameon=False)
    # NO grid
    
    # Panel B: Capital paths
    ax2 = axes[1]
    ax2.plot(t_grid, M_const, linewidth=1.5, color='#1f77b4',
             label=f'Constant $\\gamma^* = {gamma_star*100:.1f}\\%$')
    ax2.plot(t_grid, M_vary, linewidth=1.5, linestyle='--', color='#ff7f0e',
             label='Time-varying (bang-bang)')
    ax2.set_xlabel('Period $t$')
    ax2.set_ylabel('Capital $M_t$')
    ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes, fontsize=11,
             fontweight='bold', va='top', ha='left')
    ax2.legend(fontsize=9, loc='center left', frameon=False)
    # NO grid
    
    # NO suptitle - caption handles this
    
    plt.tight_layout()
    save_figure('figure_policy_comparison')
    plt.close()

def analyze_why_constant_rules():
    """Analysis of why constant rules may be preferred despite welfare loss."""
    alpha, S, M0, T = 1.05, 0.0, 1.0, 50
    
    gamma_star = find_optimal_gamma(alpha, S, M0, T)
    
    # Simulate paths
    M_const = np.zeros(T)
    C_const = np.zeros(T)
    M_const[0] = M0
    for t in range(T):
        R_t = alpha * M_const[t] + S
        C_const[t] = gamma_star * R_t
        if t < T - 1:
            M_const[t+1] = (1 - gamma_star) * R_t
    
    _, _, M_vary, C_vary = optimal_time_varying_policy(alpha, S, M0, T)
    
    print("\n" + "="*70)
    print("WHY CONSTANT RULES MAY BE PREFERRED DESPITE WELFARE LOSS")
    print("="*70)
    print(f"\nParameters: alpha = {alpha}, T = {T}, S = {S}, M0 = {M0}")
    print(f"Optimal constant rate: gamma* = {gamma_star*100:.2f}%")
    
    # 1. Grant stability
    cv_const = np.std(C_const) / np.mean(C_const) if np.mean(C_const) > 0 else np.inf
    
    print(f"\n1. GRANT STABILITY (Coefficient of Variation)")
    print(f"   Constant policy CV: {cv_const:.2f}")
    print(f"   Time-varying CV:    undefined (grants = 0 until final period)")
    
    # 2. Years with positive grants
    years_const = np.sum(C_const > 0.01)
    years_vary = np.sum(C_vary > 0.01)
    
    print(f"\n2. GRANTMAKING CONTINUITY")
    print(f"   Years with grants (constant):     {years_const} / {T}")
    print(f"   Years with grants (time-varying): {years_vary} / {T}")
    
    # 3. Robustness to horizon uncertainty
    print(f"\n3. ROBUSTNESS TO HORIZON UNCERTAINTY")
    print(f"   (Welfare if actual horizon differs from planned T={T})")
    
    T_actual_grid = [30, 40, 50, 60, 70]
    print(f"   {'Actual T':<12} {'Constant':<15} {'Time-Varying':<15} {'Ratio':<12}")
    for T_actual in T_actual_grid:
        M = M0
        W_c = 0
        for t in range(T_actual):
            R = alpha * M + S
            W_c += gamma_star * R
            M = (1 - gamma_star) * R
        
        if T_actual < T:
            W_v = 0
        elif T_actual == T:
            W_v = optimal_time_varying_closed_form(alpha, S, M0, T_actual)
        else:
            W_v = optimal_time_varying_closed_form(alpha, S, M0, T)
        
        ratio = W_c / W_v if W_v > 0 else float('inf')
        ratio_str = f'{ratio:.2f}' if ratio < 100 else 'inf'
        print(f"   {T_actual:<12} {W_c:<15.2f} {W_v:<15.2f} {ratio_str:<12}")

def sensitivity_to_inflows():
    """Analyze how welfare loss changes with inflows S."""
    alpha = 1.05
    M0 = 1.0
    horizons = [20, 50, 100]
    S_ratios = [0.0, 0.02, 0.05, 0.10, 0.20]
    
    print("\n" + "="*70)
    print("WELFARE LOSS SENSITIVITY TO EXOGENOUS INFLOWS")
    print("="*70)
    
    for T in horizons:
        print(f"\n--- Horizon T = {T} ---")
        print(f"{'S/M0':<10} {'gamma*':<12} {'Loss %':<12}")
        for s_ratio in S_ratios:
            S = s_ratio * M0
            _, W_const, W_vary, _, loss_pct = compute_welfare_loss(alpha, S, M0, T)
            gamma_star = find_optimal_gamma(alpha, S, M0, T)
            print(f"{s_ratio*100:>6.0f}%    {gamma_star*100:>8.2f}%    {loss_pct:>8.2f}%")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("WELFARE LOSS ANALYSIS: CONSTANT-RATE VS. TIME-VARYING POLICIES")
    print("="*70)
    
    print("\n### TABLE: Welfare Loss by Horizon and Return ###")
    df = generate_welfare_loss_table()
    print(df.to_string(index=False))
    
    print("\n### GENERATING LATEX TABLE ###")
    generate_welfare_loss_latex_table()
    
    print("\n### GENERATING FIGURES ###")
    plot_welfare_loss_vs_horizon()
    plot_policy_comparison()
    
    analyze_why_constant_rules()
    sensitivity_to_inflows()
    
    print("\n" + "="*70)
    print("[OK] Analysis complete.")
    print("="*70)
