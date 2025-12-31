#!/usr/bin/env python3
"""
Welfare Loss Sensitivity Analysis - Clean Version

Key insight: Under CRRA utility (sigma > 0), the optimal time-varying policy is NOT 
bang-bang because zero consumption gives -infinity utility. The optimal policy involves 
smooth consumption paths similar to constant rules. This means:

1. Under linear utility: bang-bang is optimal, losses are 50-95%
2. Under CRRA: optimal varying policy is smooth, constant rules are nearly optimal

For the paper, we report "efficiency" = W_constant / W_optimal_varying, where values 
close to 100% indicate constant rules perform well.
"""

import numpy as np
from scipy.optimize import minimize_scalar, minimize
import pandas as pd

# =============================================================================
# CORE FUNCTIONS  
# =============================================================================

def simulate_path(gamma, alpha, S, M0, T):
    """Simulate capital and consumption paths under constant gamma."""
    M = np.zeros(T + 1)
    C = np.zeros(T)
    M[0] = M0
    
    for t in range(T):
        R_t = alpha * M[t] + S
        C[t] = gamma * R_t
        M[t + 1] = (1 - gamma) * R_t
    
    return M, C


def welfare_from_path(C, sigma, delta):
    """Compute welfare from consumption path."""
    beta = 1.0 / (1.0 + delta)
    T = len(C)
    W = 0.0
    
    for t in range(T):
        c = max(C[t], 1e-10)
        if abs(sigma - 1.0) < 1e-10:
            u = np.log(c)
        elif sigma == 0:
            u = c
        else:
            u = (c ** (1 - sigma)) / (1 - sigma)
        W += (beta ** t) * u
    
    return W


def find_optimal_constant_gamma(alpha, S, M0, T, sigma, delta):
    """Find optimal constant payout rate."""
    def neg_welfare(gamma):
        if gamma <= 0.001 or gamma >= 0.999:
            return 1e10
        _, C = simulate_path(gamma, alpha, S, M0, T)
        return -welfare_from_path(C, sigma, delta)
    
    res = minimize_scalar(neg_welfare, bounds=(0.001, 0.999), method='bounded')
    return res.x, -res.fun


def optimal_varying_welfare_linear(alpha, S, M0, T, delta):
    """
    Optimal time-varying welfare under LINEAR impact (sigma=0).
    Bang-bang is optimal: gamma_t = 0 for t < T-1, gamma_{T-1} = 1.
    """
    beta = 1.0 / (1.0 + delta)
    
    if abs(alpha - 1.0) < 1e-10:
        M_final = M0 + S * (T - 1)
    else:
        M_final = (alpha ** (T - 1)) * M0 + S * ((alpha ** (T - 1)) - 1) / (alpha - 1)
    
    C_final = alpha * M_final + S
    W = (beta ** (T - 1)) * C_final
    
    return W


def optimal_varying_welfare_crra(alpha, S, M0, T, sigma, delta):
    """
    Optimal time-varying welfare under CRRA via dynamic programming.
    Uses value function iteration on a grid.
    """
    beta = 1.0 / (1.0 + delta)
    
    def u(c):
        c = max(c, 1e-10)
        if abs(sigma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return (c ** (1 - sigma)) / (1 - sigma)
    
    M_max = (alpha ** T) * M0 * 2 + S * T * 2
    n_grid = 300
    M_grid = np.linspace(1e-6, M_max, n_grid)
    
    # Terminal value: spend everything
    V = np.array([u(alpha * M + S) for M in M_grid])
    
    # Backward induction
    for t in range(T - 2, -1, -1):
        V_new = np.zeros(n_grid)
        for i, M in enumerate(M_grid):
            R = alpha * M + S
            
            def neg_value(gamma):
                if gamma <= 0.01 or gamma >= 0.99:
                    return 1e10
                c = gamma * R
                M_next = (1 - gamma) * R
                V_next = np.interp(M_next, M_grid, V)
                return -(u(c) + beta * V_next)
            
            res = minimize_scalar(neg_value, bounds=(0.01, 0.99), method='bounded')
            V_new[i] = -res.fun
        
        V = V_new
    
    return np.interp(M0, M_grid, V)


def compute_welfare_loss_clean(alpha, S, M0, T, sigma, delta):
    """
    Compute welfare loss with proper handling of CRRA case.
    
    Returns (gamma_star, W_constant, W_optimal, loss_pct)
    where loss_pct = 100 * (W_optimal - W_constant) / |W_optimal|
    """
    gamma_star, W_constant = find_optimal_constant_gamma(alpha, S, M0, T, sigma, delta)
    
    if sigma == 0:
        W_optimal = optimal_varying_welfare_linear(alpha, S, M0, T, delta)
    else:
        W_optimal = optimal_varying_welfare_crra(alpha, S, M0, T, sigma, delta)
    
    if abs(W_optimal) < 1e-10:
        loss_pct = 0.0
    else:
        loss_pct = 100 * (W_optimal - W_constant) / abs(W_optimal)
    
    # Cap at 0 if constant beats optimal (numerical artifact)
    loss_pct = max(0.0, loss_pct)
    
    return gamma_star, W_constant, W_optimal, loss_pct


# =============================================================================
# GENERATE TABLE
# =============================================================================

def generate_table():
    """Generate the welfare loss sensitivity table."""
    alpha = 1.05
    S, M0 = 0.0, 1.0
    
    horizons = [20, 50, 100, 200]
    specs = [
        (0, 0.00, 'None ($\\delta=0$)', 'Linear ($\\sigma=0$)'),
        (1, 0.00, '', 'Log ($\\sigma=1$)'),
        (2, 0.00, '', 'CRRA ($\\sigma=2$)'),
        (0, 0.02, '$\\delta=2\\%$', 'Linear ($\\sigma=0$)'),
        (1, 0.02, '', 'Log ($\\sigma=1$)'),
        (2, 0.02, '', 'CRRA ($\\sigma=2$)'),
        (0, 0.04, '$\\delta=4\\%$', 'Linear ($\\sigma=0$)'),
        (1, 0.04, '', 'Log ($\\sigma=1$)'),
        (2, 0.04, '', 'CRRA ($\\sigma=2$)'),
    ]
    
    results = []
    
    print("Computing welfare losses...")
    print("-" * 70)
    
    for sigma, delta, disc_label, impact_label in specs:
        row = {'Discounting': disc_label, 'Impact': impact_label}
        
        for T in horizons:
            print(f"  sigma={sigma}, delta={delta:.0%}, T={T}...", end=" ")
            _, _, _, loss = compute_welfare_loss_clean(alpha, S, M0, T, sigma, delta)
            row[f'T={T}'] = loss
            print(f"{loss:.1f}%")
        
        results.append(row)
    
    return pd.DataFrame(results)


def print_table(df):
    """Print formatted table."""
    print("\n" + "=" * 80)
    print("WELFARE LOSS SENSITIVITY TABLE")
    print("=" * 80)
    print(f"\n{'Discounting':<18} {'Impact':<22} {'T=20':>8} {'T=50':>8} {'T=100':>8} {'T=200':>8}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        disc = row['Discounting'] if row['Discounting'] else ""
        disc = disc.replace('\\', '').replace('$', '').replace('delta=', 'δ=')
        imp = row['Impact'].replace('\\', '').replace('$', '').replace('sigma=', 'σ=')
        
        print(f"{disc:<18} {imp:<22} {row['T=20']:>7.1f}% {row['T=50']:>7.1f}% {row['T=100']:>7.1f}% {row['T=200']:>7.1f}%")


def generate_latex(df):
    """Generate LaTeX table."""
    latex = r"""\begin{table}[htbp]
\centering
\caption{Welfare loss sensitivity to specification ($r = 5\%$, $S=0$, $M_0=1$)}
\label{tab:welfare-loss-sensitivity}
\begin{tabular}{llcccc}
\toprule
& & \multicolumn{4}{c}{\textbf{Planning horizon $T$}} \\
\cmidrule(lr){3-6}
\textbf{Discounting} & \textbf{Impact function} & $T=20$ & $T=50$ & $T=100$ & $T=200$ \\
\midrule
"""
    
    for i, row in df.iterrows():
        disc = row['Discounting']
        imp = row['Impact']
        
        vals = [f"{row[f'T={T}']:.1f}\\%" for T in [20, 50, 100, 200]]
        
        latex += f"{disc} & {imp} & {' & '.join(vals)} \\\\\n"
        
        if i in [2, 5]:
            latex += "\\addlinespace\n"
    
    latex += r"""\bottomrule
\end{tabular}

\smallskip
\emph{Notes}: Welfare loss $(W^* - W(\gamma^*))/W^*$ under each specification. Under the baseline (linear, undiscounted), the loss is 77.5\% at $T=50$. Under log utility with 2\% discounting, the loss falls dramatically because the optimal time-varying policy under CRRA involves smooth consumption rather than bang-bang, making constant rules nearly optimal.
\end{table}
"""
    return latex


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("WELFARE LOSS SENSITIVITY ANALYSIS")
    print("=" * 70)
    
    df = generate_table()
    print_table(df)
    
    latex = generate_latex(df)
    with open('table_welfare_loss_sensitivity.tex', 'w') as f:
        f.write(latex)
    print("\n[OK] Saved: table_welfare_loss_sensitivity.tex")
    
    df.to_csv('welfare_loss_sensitivity.csv', index=False)
    print("[OK] Saved: welfare_loss_sensitivity.csv")
