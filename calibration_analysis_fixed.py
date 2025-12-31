# calibration_analysis.py -- Baseline + optional discounting (beta)
# ----------------------------------------------------------------
# - Baseline remains identical (beta=1.0 by default) -- no paper changes required
# - total_grants now supports beta in (0,1] with a numerically-stable closed form
# - Table 1 note updated to reference W(gamma) in \eqref{eq:normative}
# - Optional beta robustness figure/table behind RUN_BETA_EXTRAS toggle
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

# Standard figure sizes
FIG_SINGLE = (6.5, 4.5)
FIG_WIDE = (10, 5)

def save_figure(filename):
    """Save figure in PDF and PNG formats."""
    plt.savefig(f'{filename}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {filename}.pdf")

# ===== HELPERS =====

TOL = 1e-12

def H1(x, T):
    """Geometric sum H1(x) = (1 - x^T) / (1 - x) with a robust x->1 limit."""
    if abs(1.0 - x) < 1e-12:
        return float(T)
    return (1.0 - x**T) / (1.0 - x)

def compute_A_B(gamma, alpha, S):
    A = alpha * (1 - gamma)
    B = S * (1 - gamma)
    return A, B

def gamma_critical(alpha):
    return 1.0 - 1.0 / alpha

def gamma_asymptotic(alpha, T):
    # Large-T approximation (pure endowment). Undefined if T < alpha/(alpha-1)
    return 1.0 / (T - alpha / (alpha - 1.0))

# ===== OBJECTIVE: W_beta(gamma) with beta in (0,1], baseline beta=1 =====

def _total_grants_safe_sum(gamma, alpha, S, M0, T, beta):
    """Exact finite-sum fallback: robust near singularities (fast for T <= 1e3)."""
    if gamma <= 0 or gamma >= 1:
        return -np.inf
    A, B = compute_A_B(gamma, alpha, S)
    M = M0
    total = 0.0
    for t in range(T):
        Ct = gamma * (alpha * M + S)
        total += (beta ** t) * Ct
        M = A * M + B
    return total

def total_grants(gamma, alpha, S, M0, T, beta=1.0):
    """
    Objective W_beta(gamma) = sum_{t=0}^{T-1} beta^t C_t.
    Baseline is beta=1.0 (time-neutral).
    """
    if gamma <= 0 or gamma >= 1:
        return -np.inf

    A, B = compute_A_B(gamma, alpha, S)

    # Near-singularity guardrails for numerical stability
    near_A1      = abs(1.0 - A) < 1e-10
    near_betaA1  = (beta < 1.0) and (abs(1.0 - beta * A) < 1e-10)

    if beta == 1.0:
        # Baseline: time-neutral
        if near_A1:
            gamma_c = gamma_critical(alpha)
            return ((alpha - 1.0) * (T * M0 + (S / alpha) * T * (T - 1) / 2.0)
                    + gamma_c * S * T)
        H1A = H1(A, T)
        denom = (1.0 - A)
        term2 = 0.0 if abs(denom) < 1e-14 else (B / denom) * (T - H1A)
        return gamma * alpha * (M0 * H1A + term2) + gamma * S * T

    # beta in (0,1)
    if near_A1 or near_betaA1:
        return _total_grants_safe_sum(gamma, alpha, S, M0, T, beta)

    H1_betaA = H1(beta * A, T)
    H1_beta  = H1(beta, T)
    denom = (1.0 - A)
    term2 = 0.0 if abs(denom) < 1e-14 else (B / denom) * (H1_beta - H1_betaA)
    return gamma * alpha * (M0 * H1_betaA + term2) + gamma * S * H1_beta

def objective_to_minimize(gamma, alpha, S, M0, T, beta=1.0):
    return -total_grants(gamma, alpha, S, M0, T, beta=beta)

def find_optimal_gamma(alpha, S, M0, T, beta=1.0):
    """Unconstrained optimum (Option A baseline)."""
    res = minimize_scalar(
        objective_to_minimize,
        bounds=(1e-6, 1.0 - 1e-6),
        args=(alpha, S, M0, T, beta),
        method='bounded'
    )
    return res.x

# ===== TABLES =====

def generate_baseline_table(beta=1.0):
    """Generate Table 1 with baseline calibration results."""
    returns = [0.03, 0.05, 0.07]
    horizons = [20, 30, 40, 50, 100, 200]
    M0, S = 1.0, 0.0
    rows = []

    for r in returns:
        alpha = 1.0 + r
        gc = gamma_critical(alpha) * 100.0
        row = {
            'Real return $r$': f'{int(r*100)}\\%',
            '$\\alpha$': f'{alpha:.2f}',
            '$\\gamma_c$': f'{gc:.2f}\\%'
        }
        for T in horizons:
            g = find_optimal_gamma(alpha, S, M0, T, beta=beta) * 100.0
            cell = f'{g:.2f}\\%'
            if g > gc + 1e-8:
                cell += '\\textsuperscript{\\dag}'
            row[f'$T={T}$'] = cell
        rows.append(row)

    df = pd.DataFrame(rows, columns=['Real return $r$', '$\\alpha$', '$\\gamma_c$']
                                 + [f'$T={T}$' for T in horizons])
    df.to_latex('table1_baseline.tex', index=False, escape=False)

    with open('table1_note.tex', 'w') as f:
        f.write(
            "\\emph{Note:} Values computed numerically by maximizing $W(\\gamma)$ in "
            "\\eqref{eq:normative} with no additional constraints.\n"
            "The critical payout $\\gamma_c = 1 - 1/\\alpha$ separates accumulation ($\\gamma<\\gamma_c$) from\n"
            "decumulation ($\\gamma>\\gamma_c$) regimes. Cells marked ${}^{\\dagger}$ indicate $\\gamma^*>\\gamma_c$.\n"
        )
    print("[OK] Saved: table1_baseline.tex and table1_note.tex")
    return df

def generate_asymptotic_panel_table():
    """Generate Table 2 for asymptotic approximation accuracy."""
    alphas = [1.03, 1.05, 1.07]
    Ts     = [50, 100, 200, 500]
    panelA, panelB, panelC = [], [], []
    for a in alphas:
        rowA = [f'{(a-1)*100:.0f}% (alpha={a:.02f})']
        rowB = [f'{(a-1)*100:.0f}%']
        rowC = [f'{(a-1)*100:.0f}%']
        for T in Ts:
            g_exact  = find_optimal_gamma(a, 0.0, 1.0, T) * 100.0
            g_approx = gamma_asymptotic(a, T) * 100.0
            err = abs(g_exact - g_approx) / g_exact * 100.0
            rowA.append(f'{g_exact:.2f}')
            rowB.append(f'{g_approx:.2f}')
            rowC.append(f'{err:.2f}%')
        panelA.append(rowA)
        panelB.append(rowB)
        panelC.append(rowC)

    with open('table2_panel.tex', 'w') as f:
        f.write(r"""\begin{table}[t]
\centering
\caption{Asymptotic approximation accuracy for pure endowment ($S=0$, $M_0=1$)}
\label{tab:asymptotic}
\begin{tabular}{lcccc}
\toprule
\textbf{Real return $r$} & \textbf{$T=50$} & \textbf{$T=100$} & \textbf{$T=200$} & \textbf{$T=500$}\\
\midrule
\multicolumn{5}{l}{\textbf{Panel A: Exact optimal payout $\gamma^\ast$ (\%)}}\\
""")
        for row in panelA:
            f.write(f"{row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} \\\\\n")
        f.write(r"""\addlinespace
\multicolumn{5}{l}{\textbf{Panel B: Corrected asymptotic }$\tilde\gamma=\dfrac{1}{\,T-\frac{\alpha}{\alpha-1}\,}$\textbf{ (\%)}}\\
""")
        for row in panelB:
            f.write(f"{row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} \\\\\n")
        f.write(r"""\addlinespace
\multicolumn{5}{l}{\textbf{Panel C: Relative error } $\left|\gamma^\ast-\tilde\gamma\right|/\gamma^\ast \times 100\%$}\\
""")
        for row in panelC:
            f.write(f"{row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} \\\\\n")
        f.write(r"""\bottomrule
\end{tabular}

\smallskip
\emph{Notes}: The large-$T$ approximation $\tilde\gamma$ is undefined (negative) for very short horizons when $T<\alpha/(\alpha-1)$; therefore $T=20$ is omitted.
\end{table}
""")
    print("[OK] Saved: table2_panel.tex")

def generate_inflow_table(beta=1.0):
    """Generate Table 3 for inflow sensitivity."""
    alpha = 1.05
    M0 = 1.0
    S_ratios = [0.0, 0.02, 0.05, 0.10, 0.20]
    horizons = [20, 50, 100]
    rows = []
    for ratio in S_ratios:
        S = ratio * M0
        row = {'S/M_0': f'{ratio*100:.0f}%'}
        for T in horizons:
            g = find_optimal_gamma(alpha, S, M0, T, beta=beta)
            row[f'T={T}'] = f'{g*100:.2f}%'
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv('table3_inflows.csv', index=False)
    df.to_latex('table3_inflows.tex', index=False, escape=False)
    print("[OK] Saved: table3_inflows.csv and table3_inflows.tex")
    return df

# ===== OPTIONAL: beta robustness outputs (appendix-ready) =====

def plot_beta_sensitivity():
    """Optional appendix figure: gamma* vs beta."""
    betas = np.linspace(0.95, 1.00, 26)
    alphas = [1.03, 1.05, 1.07]
    Ts = [50, 100]
    M0, S = 1.0, 0.0

    plt.figure(figsize=FIG_SINGLE)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    linestyles = ['-', '--']
    
    for a, color in zip(alphas, colors):
        for T, ls in zip(Ts, linestyles):
            gstars = [find_optimal_gamma(a, S, M0, T, beta=b)*100.0 for b in betas]
            plt.plot(betas, gstars, linewidth=1.5, linestyle=ls, color=color,
                     label=f'$r={(a-1)*100:.0f}\\%$, $T={T}$')
    
    plt.xlabel('Discount factor $\\beta$')
    plt.ylabel('Optimal payout $\\gamma^*$ (\\%)')
    # NO title
    plt.legend(fontsize=8, ncol=2, loc='lower left', frameon=False)
    # NO grid
    plt.tight_layout()
    save_figure('figure_beta_sensitivity')

def generate_beta_panel_table():
    """Optional appendix table: gamma* across beta."""
    alpha = 1.05
    M0, S = 1.0, 0.0
    betas = [1.00, 0.99, 0.97, 0.95]
    horizons = [20, 50, 100]
    rows = []
    for T in horizons:
        row = {'Horizon T': f'{T}'}
        for b in betas:
            g = find_optimal_gamma(alpha, S, M0, T, beta=b)*100.0
            row[f'$\\beta={b:.2f}$'] = f'{g:.2f}'
        rows.append(row)
    df = pd.DataFrame(rows, columns=['Horizon T'] + [f'$\\beta={b:.2f}$' for b in betas])
    df.to_latex('table_beta_panel.tex', index=False, escape=False,
                caption=r'Optimal payout $\gamma^*$ under social discounting ($\alpha=1.05$, $S=0$, $M_0=1$).',
                label='tab:beta-panel')
    print("[OK] Saved: table_beta_panel.tex")
    return df

# ===== FIGURES =====

def plot_optimal_vs_horizon_main():
    """
    CLEAN Figure for the paper (Section 4.1):
    - Exact gamma* vs T for r in {3,5,7}%
    - Horizontal gamma_c lines
    - No asymptote overlay
    - Y-axis limited to [0, 12] %
    """
    alpha_values = [1.03, 1.05, 1.07]
    T_grid = np.arange(10, 201, 5)
    M0, S = 1.0, 0.0
    y_upper = 12.0

    plt.figure(figsize=FIG_SINGLE)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for alpha, color in zip(alpha_values, colors):
        r_pct = (alpha - 1.0) * 100.0
        gamma_exact = [find_optimal_gamma(alpha, S, M0, T) * 100.0 for T in T_grid]
        plt.plot(T_grid, gamma_exact, linewidth=1.5, color=color, 
                 label=f'$r={r_pct:.0f}\\%$')
        
        # Critical line (horizontal, subtle)
        gc = gamma_critical(alpha) * 100.0
        plt.axhline(gc, linestyle=':', linewidth=1.0, color=color, alpha=0.6)

    plt.ylim(0, y_upper)
    plt.xlim(10, 200)
    plt.xlabel('Planning horizon $T$ (years)')
    plt.ylabel('Optimal payout rate $\\gamma^*$ (%)')
    # NO title
    plt.legend(fontsize=9, loc='upper right', frameon=False)
    # NO grid
    plt.tight_layout()
    save_figure('figure_gamma_vs_T')

def plot_optimal_vs_horizon_overlay():
    """
    OPTIONAL supplemental figure with asymptotic overlay.
    """
    alpha_values = [1.03, 1.05, 1.07]
    T_grid = np.arange(10, 201, 5)
    M0, S = 1.0, 0.0
    y_upper = 12.0

    plt.figure(figsize=FIG_SINGLE)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for alpha, color in zip(alpha_values, colors):
        r_pct = (alpha - 1.0) * 100.0
        gamma_exact = [find_optimal_gamma(alpha, S, M0, T) * 100.0 for T in T_grid]
        plt.plot(T_grid, gamma_exact, linewidth=1.5, color=color, 
                 label=f'Exact, $r={r_pct:.0f}\\%$')

        # Asymptote: only plot above the validity boundary
        T_bound = alpha / (alpha - 1.0)
        approx = []
        for T in T_grid:
            if T > T_bound + 5:
                val = gamma_asymptotic(alpha, T) * 100.0
                approx.append(val if val <= y_upper else np.nan)
            else:
                approx.append(np.nan)
        plt.plot(T_grid, approx, linestyle='--', linewidth=1.2, color=color, alpha=0.7,
                 label=f'Asymptotic, $r={r_pct:.0f}\\%$')

        # Critical line
        gc = gamma_critical(alpha) * 100.0
        plt.axhline(gc, linestyle=':', linewidth=1.0, color=color, alpha=0.5)

    plt.ylim(0, y_upper)
    plt.xlim(10, 200)
    plt.xlabel('Planning horizon $T$ (years)')
    plt.ylabel('Optimal payout rate $\\gamma^*$ (%)')
    # NO title
    plt.legend(fontsize=8, ncol=2, loc='upper right', frameon=False)
    # NO grid
    plt.tight_layout()
    save_figure('figure_gamma_vs_T_overlay')

def plot_inflow_sensitivity(beta=1.0):
    """Plot sensitivity to exogenous inflows."""
    alpha = 1.05
    M0 = 1.0
    S_ratios = np.linspace(0, 0.3, 30)
    horizons = [20, 50, 100]

    plt.figure(figsize=FIG_SINGLE)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for T, color in zip(horizons, colors):
        gammas = [find_optimal_gamma(alpha, s*M0, M0, T, beta=beta)*100.0 for s in S_ratios]
        plt.plot(S_ratios * 100.0, gammas, linewidth=1.5, color=color, 
                 label=f'$T = {T}$ years')

    gc = gamma_critical(alpha) * 100.0
    plt.axhline(gc, linestyle=':', linewidth=1.0, color='gray', alpha=0.7,
                label=f'$\\gamma_c = {gc:.2f}\\%$')
    
    plt.xlabel('Inflow-to-capital ratio $S/M_0$ (%)')
    plt.ylabel('Optimal payout rate $\\gamma^*$ (%)')
    # NO title
    plt.legend(fontsize=9, loc='upper left', frameon=False)
    # NO grid
    plt.tight_layout()
    save_figure('figure_inflow_sensitivity')

def plot_total_giving_vs_gamma(beta=1.0):
    """
    Figure: Total giving (normalized) vs payout gamma for different horizons.
    """
    alpha, M0, S = 1.05, 1.0, 0.0
    horizons = [20, 50, 100, 200]
    gamma_grid = np.linspace(0.0005, 0.10, 400)

    plt.figure(figsize=FIG_SINGLE)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for T, color in zip(horizons, colors):
        totals = np.array([total_grants(g, alpha, S, M0, T, beta=beta) for g in gamma_grid])
        max_val = totals.max()
        totals_norm = totals / max_val if max_val > 0 else totals
        gstar = find_optimal_gamma(alpha, S, M0, T, beta=beta)
        plt.plot(gamma_grid * 100.0, totals_norm, linewidth=1.5, color=color, 
                 label=f'$T = {T}$')
        plt.plot(gstar * 100.0, 1.0, 'o', markersize=5, color=color)

    gc = gamma_critical(alpha) * 100.0
    plt.axvline(gc, linestyle=':', linewidth=1.0, color='gray', alpha=0.7,
                label=f'$\\gamma_c={gc:.2f}\\%$')

    plt.xlim(0, 10)
    plt.ylim(0, 1.05)
    plt.xlabel('Payout rate $\\gamma$ (%)')
    plt.ylabel('Total giving (normalized)')
    # NO title
    plt.legend(fontsize=9, loc='center right', frameon=False)
    # NO grid
    plt.tight_layout()
    save_figure('figure_total_vs_gamma')

# ===== MAIN =====

if __name__ == "__main__":
    RUN_BETA_EXTRAS = False  # toggle to True to generate beta robustness outputs

    print("="*70)
    print("PHILANTHROPIC FOUNDATION CALIBRATION ANALYSIS -- Baseline + beta hooks")
    print("="*70)

    print("\n### TABLE 1: Baseline Calibration (pure endowment) ###")
    df1 = generate_baseline_table(beta=1.0)
    print(df1.to_string(index=False))

    print("\n### TABLE 2: Asymptotic Approximation (panel) ###")
    generate_asymptotic_panel_table()

    print("\n### TABLE 3: Sensitivity to Inflows ###")
    df3 = generate_inflow_table(beta=1.0)
    print(df3.to_string(index=False))

    print("\n### GENERATING FIGURES ###")
    plot_optimal_vs_horizon_main()
    plot_optimal_vs_horizon_overlay()
    plot_total_giving_vs_gamma(beta=1.0)
    plot_inflow_sensitivity(beta=1.0)

    if RUN_BETA_EXTRAS:
        print("\n### OPTIONAL: beta robustness outputs (appendix) ###")
        plot_beta_sensitivity()
        generate_beta_panel_table()

    print("\n" + "="*70)
    print("[OK] All figures and tables generated.")
    print("="*70)
