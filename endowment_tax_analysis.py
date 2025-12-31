# endowment_tax_analysis.py — Endowment Tax Extension Analysis
# ----------------------------------------------------------------
# Analyzes the impact of the 2025 One Big Beautiful Bill Act (OBBBA)
# endowment tax on optimal payout rates for university endowments.
#
# UPDATED: Figure styling for Journal of Public Economic Theory submission
# - Removed in-figure titles (captions in LaTeX)
# - Removed grid lines
# - Serif fonts, frameless legends
# ----------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import pandas as pd

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
FIG_TWO_PANEL = (10, 4)

def save_figure(filename):
    """Save figure in PDF and PNG formats."""
    plt.savefig(f'{filename}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {filename}.pdf")

# ===== CORE FUNCTIONS =====

def H1(x, T):
    """Geometric sum H1(x) = (1 - x^T) / (1 - x) with robust x->1 limit."""
    if abs(1.0 - x) < 1e-12:
        return float(T)
    return (1.0 - x**T) / (1.0 - x)

def total_grants(gamma, alpha, S, M0, T):
    """Compute total grants W(γ) = Σ_{t=0}^{T-1} C_t under constant payout γ."""
    if gamma <= 0 or gamma >= 1:
        return -np.inf
    
    A = alpha * (1 - gamma)
    B = S * (1 - gamma)
    
    if abs(1.0 - A) < 1e-10:
        gamma_c = 1 - 1/alpha
        return ((alpha - 1.0) * (T * M0 + (S / alpha) * T * (T - 1) / 2.0)
                + gamma_c * S * T)
    
    H1A = H1(A, T)
    denom = (1.0 - A)
    term2 = 0.0 if abs(denom) < 1e-14 else (B / denom) * (T - H1A)
    return gamma * alpha * (M0 * H1A + term2) + gamma * S * T

def find_optimal_gamma(alpha, S, M0, T):
    """Find γ* that maximizes total grants."""
    res = minimize_scalar(
        lambda g: -total_grants(g, alpha, S, M0, T),
        bounds=(1e-6, 1.0 - 1e-6),
        method='bounded'
    )
    return res.x

def gamma_critical(alpha):
    """Critical payout rate γ_c = 1 - 1/α."""
    return 1.0 - 1.0 / alpha

# ===== TAX-ADJUSTED FUNCTIONS =====

def alpha_after_tax(r, tau):
    """Compute after-tax gross return factor: α_τ = 1 + r(1 - τ)."""
    return 1.0 + r * (1.0 - tau)

def gamma_critical_tax(r, tau):
    """Critical payout rate under taxation."""
    alpha_tau = alpha_after_tax(r, tau)
    return gamma_critical(alpha_tau)

def find_optimal_gamma_tax(r, tau, S, M0, T):
    """Find optimal payout under taxation."""
    alpha_tau = alpha_after_tax(r, tau)
    return find_optimal_gamma(alpha_tau, S, M0, T)

# ===== OBBBA TAX TIERS =====

OBBBA_TIERS = {
    'Baseline': 0.00,
    'Tier 1 (1.4%)': 0.014,
    'Tier 2 (4%)': 0.04,
    'Tier 3 (8%)': 0.08
}

# ===== TABLE GENERATION =====

def generate_tax_impact_table(r=0.05, M0=1.0, S=0.0):
    """Generate Table 4: Optimal payout rates under OBBBA tax tiers."""
    horizons = [20, 50, 100, 200]
    
    rows = []
    for tier_name, tau in OBBBA_TIERS.items():
        alpha_tau = alpha_after_tax(r, tau)
        row = {'Tax tier': tier_name, 'τ': f'{tau*100:.1f}%', 'α_τ': f'{alpha_tau:.4f}'}
        for T in horizons:
            g = find_optimal_gamma(alpha_tau, S, M0, T) * 100
            row[f'T={T}'] = f'{g:.2f}%'
        rows.append(row)
    
    # Add delta row
    baseline_row = rows[0]
    tier3_row = rows[-1]
    delta_row = {'Tax tier': 'Δ(Tier 3 − Baseline)', 'τ': '', 'α_τ': ''}
    for T in horizons:
        baseline_val = float(baseline_row[f'T={T}'].replace('%', ''))
        tier3_val = float(tier3_row[f'T={T}'].replace('%', ''))
        delta = tier3_val - baseline_val
        delta_row[f'T={T}'] = f'+{delta:.2f} pp'
    rows.append(delta_row)
    
    df = pd.DataFrame(rows)
    return df

def generate_critical_threshold_table(r=0.05):
    """Generate table of critical thresholds under each tax tier."""
    rows = []
    for tier_name, tau in OBBBA_TIERS.items():
        alpha_tau = alpha_after_tax(r, tau)
        gamma_c = gamma_critical(alpha_tau) * 100
        baseline_gc = gamma_critical(1 + r) * 100
        change = gamma_c - baseline_gc
        rows.append({
            'Tax tier': tier_name,
            'τ': f'{tau*100:.1f}%',
            'α_τ': f'{alpha_tau:.4f}',
            'γ_c': f'{gamma_c:.2f}%',
            'Change': f'{change:+.2f} pp' if tau > 0 else '—'
        })
    return pd.DataFrame(rows)

# ===== FIGURE GENERATION =====

def plot_optimal_vs_horizon_by_tax_tier(r=0.05, M0=1.0, S=0.0, save=True):
    """Generate Figure: Optimal payout vs. horizon for each OBBBA tax tier."""
    T_grid = np.arange(10, 201, 2)
    
    colors = {
        'Baseline': '#1f77b4',
        'Tier 1 (1.4%)': '#2ca02c',
        'Tier 2 (4%)': '#ff7f0e',
        'Tier 3 (8%)': '#d62728'
    }
    
    linestyles = {
        'Baseline': '-',
        'Tier 1 (1.4%)': '--',
        'Tier 2 (4%)': '-.',
        'Tier 3 (8%)': ':'
    }
    
    plt.figure(figsize=FIG_SINGLE)
    
    for tier_name, tau in OBBBA_TIERS.items():
        alpha_tau = alpha_after_tax(r, tau)
        gamma_opt = [find_optimal_gamma(alpha_tau, S, M0, T) * 100 for T in T_grid]
        
        plt.plot(T_grid, gamma_opt, 
                 linewidth=1.5, 
                 color=colors[tier_name],
                 linestyle=linestyles[tier_name],
                 label=tier_name)
    
    # Add critical threshold line (baseline only, subtle)
    alpha_baseline = alpha_after_tax(r, 0.0)
    gc = gamma_critical(alpha_baseline) * 100
    plt.axhline(gc, linestyle=':', linewidth=1.0, color='gray', alpha=0.6)
    
    plt.xlim(10, 200)
    plt.ylim(0, 16)
    plt.xlabel('Planning horizon $T$ (years)')
    plt.ylabel('Optimal payout rate $\\gamma^*$ (%)')
    # NO title - caption handles this
    plt.legend(fontsize=9, loc='upper right', frameon=False)
    # NO grid
    plt.tight_layout()
    
    if save:
        save_figure('figure_endowment_tax')

def plot_tax_effect_detail(r=0.05, M0=1.0, S=0.0, save=True):
    """Generate a two-panel figure showing tax effects in detail."""
    T_grid = np.arange(30, 151, 2)
    
    colors = {
        'Baseline': '#1f77b4',
        'Tier 1 (1.4%)': '#2ca02c',
        'Tier 2 (4%)': '#ff7f0e',
        'Tier 3 (8%)': '#d62728'
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_TWO_PANEL)
    
    # Left panel: Absolute levels
    for tier_name, tau in OBBBA_TIERS.items():
        alpha_tau = alpha_after_tax(r, tau)
        gamma_opt = [find_optimal_gamma(alpha_tau, S, M0, T) * 100 for T in T_grid]
        ax1.plot(T_grid, gamma_opt, linewidth=1.5, color=colors[tier_name], label=tier_name)
    
    ax1.set_xlabel('Planning horizon $T$ (years)')
    ax1.set_ylabel('Optimal payout rate $\\gamma^*$ (%)')
    ax1.text(0.06, 0.98, '(a)', transform=ax1.transAxes, fontsize=11,
             fontweight='bold', va='top', ha='left')
    ax1.legend(fontsize=8, loc='upper right', frameon=False)
    ax1.set_ylim(0, 8)
    # NO grid
    
    # Right panel: Percentage increase relative to baseline
    baseline_gamma = {}
    for T in T_grid:
        alpha_baseline = alpha_after_tax(r, 0.0)
        baseline_gamma[T] = find_optimal_gamma(alpha_baseline, S, M0, T) * 100
    
    for tier_name, tau in list(OBBBA_TIERS.items())[1:]:  # Skip baseline
        alpha_tau = alpha_after_tax(r, tau)
        pct_increase = []
        for T in T_grid:
            g = find_optimal_gamma(alpha_tau, S, M0, T) * 100
            pct_inc = (g - baseline_gamma[T]) / baseline_gamma[T] * 100
            pct_increase.append(pct_inc)
        ax2.plot(T_grid, pct_increase, linewidth=1.5, color=colors[tier_name], label=tier_name)
    
    ax2.set_xlabel('Planning horizon $T$ (years)')
    ax2.set_ylabel('Increase in $\\gamma^*$ relative to baseline (%)')
    ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes, fontsize=11,
             fontweight='bold', va='top', ha='left')
    ax2.legend(fontsize=8, loc='upper right', frameon=False)
    ax2.axhline(0, color='gray', linewidth=0.5)
    # NO grid
    
    plt.tight_layout()
    
    if save:
        save_figure('figure_endowment_tax_detail')

# ===== WELFARE ANALYSIS =====

def compute_welfare_loss(r, tau, M0, T, S=0.0):
    """Compute welfare loss from taxation as % reduction in total grants."""
    alpha_baseline = alpha_after_tax(r, 0.0)
    gamma_baseline = find_optimal_gamma(alpha_baseline, S, M0, T)
    W_baseline = total_grants(gamma_baseline, alpha_baseline, S, M0, T)
    
    alpha_tau = alpha_after_tax(r, tau)
    gamma_tau = find_optimal_gamma(alpha_tau, S, M0, T)
    W_tau = total_grants(gamma_tau, alpha_tau, S, M0, T)
    
    loss_pct = (W_baseline - W_tau) / W_baseline * 100
    
    return {
        'tau': tau,
        'gamma_baseline': gamma_baseline,
        'gamma_tau': gamma_tau,
        'W_baseline': W_baseline,
        'W_tau': W_tau,
        'loss_pct': loss_pct
    }

def generate_welfare_loss_table(r=0.05, M0=1.0, S=0.0):
    """Generate table of welfare losses under each tax tier."""
    horizons = [20, 50, 100, 200]
    
    rows = []
    for tier_name, tau in list(OBBBA_TIERS.items())[1:]:  # Skip baseline
        row = {'Tax tier': tier_name}
        for T in horizons:
            result = compute_welfare_loss(r, tau, M0, T, S)
            row[f'T={T}'] = f'{result["loss_pct"]:.2f}%'
        rows.append(row)
    
    return pd.DataFrame(rows)

# ===== LaTeX OUTPUT =====

def generate_latex_table(r=0.05, M0=1.0, S=0.0):
    """Generate LaTeX code for Table 4."""
    horizons = [20, 50, 100, 200]
    
    latex = r"""\begin{table}[htbp]
\centering
\caption{Optimal payout rates under OBBBA endowment tax tiers ($S=0$, $M_0=1$, $r=5\%$ pre-tax)}
\label{tab:tax-impact}
\begin{tabular}{lcccc}
\toprule
\textbf{Tax tier} & $\boldsymbol{T=20}$ & $\boldsymbol{T=50}$ & $\boldsymbol{T=100}$ & $\boldsymbol{T=200}$ \\
\midrule
"""
    
    baseline_vals = {}
    for tier_name, tau in OBBBA_TIERS.items():
        alpha_tau = alpha_after_tax(r, tau)
        
        if tier_name == 'Baseline':
            latex += f"Baseline ($\\tau=0$)"
        else:
            latex += f"{tier_name.replace('Tier ', 'Tier ').replace('(', '($\\tau=').replace(')', '$)')}"
        
        for T in horizons:
            g = find_optimal_gamma(alpha_tau, S, M0, T) * 100
            latex += f" & {g:.2f}\\%"
            if tier_name == 'Baseline':
                baseline_vals[T] = g
        latex += " \\\\\n"
    
    latex += r"\midrule" + "\n"
    latex += r"$\Delta$(Tier 3 $-$ Baseline)"
    
    alpha_tier3 = alpha_after_tax(r, 0.08)
    for T in horizons:
        g_tier3 = find_optimal_gamma(alpha_tier3, S, M0, T) * 100
        delta = g_tier3 - baseline_vals[T]
        latex += f" & +{delta:.2f} pp"
    latex += " \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}

\smallskip
\emph{Notes}: Values computed by maximizing $W(\gamma)$ with $\alpha_\tau = 1 + 0.05(1-\tau)$.
\end{table}
"""
    
    return latex

# ===== MAIN =====

if __name__ == "__main__":
    print("=" * 70)
    print("ENDOWMENT TAX EXTENSION ANALYSIS — OBBBA 2025")
    print("=" * 70)
    
    r = 0.05
    M0, S = 1.0, 0.0
    
    print("\n### Critical Thresholds under Each Tax Tier ###")
    df_critical = generate_critical_threshold_table(r)
    print(df_critical.to_string(index=False))
    
    print("\n### Table 4: Optimal Payout Rates under OBBBA Tax Tiers ###")
    df_table4 = generate_tax_impact_table(r, M0, S)
    print(df_table4.to_string(index=False))
    
    print("\n### Welfare Loss from Taxation (% reduction in total grants) ###")
    df_welfare = generate_welfare_loss_table(r, M0, S)
    print(df_welfare.to_string(index=False))
    
    print("\n### Generating Figures ###")
    plot_optimal_vs_horizon_by_tax_tier(r, M0, S, save=True)
    plot_tax_effect_detail(r, M0, S, save=True)
    
    print("\n### LaTeX Table Code ###")
    print(generate_latex_table(r, M0, S))
    
    print("\n" + "=" * 70)
    print("[OK] All endowment tax analysis complete.")
    print("=" * 70)
