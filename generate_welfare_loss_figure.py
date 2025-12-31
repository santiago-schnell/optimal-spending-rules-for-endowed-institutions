#!/usr/bin/env python3
"""
Generate figure showing welfare loss sensitivity to specification.
Visualizes Table 6 from the paper.

UPDATED: Figure styling for Journal of Public Economic Theory submission
- Removed in-figure titles (captions in LaTeX)
- Removed grid lines
- Serif fonts, frameless legends
- Removed text annotations (let caption handle explanation)
"""

import numpy as np
from scipy.optimize import minimize_scalar
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
    plt.savefig(f'{filename}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {filename}.pdf")

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
    """Optimal time-varying welfare under LINEAR impact."""
    beta = 1.0 / (1.0 + delta)
    
    if abs(alpha - 1.0) < 1e-10:
        M_final = M0 + S * (T - 1)
    else:
        M_final = (alpha ** (T - 1)) * M0 + S * ((alpha ** (T - 1)) - 1) / (alpha - 1)
    
    C_final = alpha * M_final + S
    W = (beta ** (T - 1)) * C_final
    
    return W


def optimal_varying_welfare_crra(alpha, S, M0, T, sigma, delta):
    """Optimal time-varying welfare under CRRA via dynamic programming."""
    beta = 1.0 / (1.0 + delta)
    
    def u(c):
        c = max(c, 1e-10)
        if abs(sigma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return (c ** (1 - sigma)) / (1 - sigma)
    
    M_max = (alpha ** T) * M0 * 2 + S * T * 2
    n_grid = 200
    M_grid = np.linspace(1e-6, M_max, n_grid)
    
    V = np.array([u(alpha * M + S) for M in M_grid])
    
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


def compute_welfare_loss(alpha, S, M0, T, sigma, delta):
    """Compute welfare loss from constant-rate restriction."""
    gamma_star, W_constant = find_optimal_constant_gamma(alpha, S, M0, T, sigma, delta)
    
    if sigma == 0:
        W_optimal = optimal_varying_welfare_linear(alpha, S, M0, T, delta)
    else:
        W_optimal = optimal_varying_welfare_crra(alpha, S, M0, T, sigma, delta)
    
    if abs(W_optimal) < 1e-10:
        loss_pct = 0.0
    else:
        loss_pct = 100 * (W_optimal - W_constant) / abs(W_optimal)
    
    loss_pct = max(0.0, loss_pct)
    
    return loss_pct


# =============================================================================
# GENERATE FIGURE
# =============================================================================

def generate_welfare_loss_sensitivity_figure():
    """Generate figure showing welfare loss vs horizon for different specifications."""
    alpha = 1.05
    S, M0 = 0.0, 1.0
    
    T_grid = np.array([10, 20, 30, 40, 50, 75, 100, 150, 200])
    
    specs = [
        (0, 0.00, 'Linear, no discount', '-', '#1f77b4'),
        (0, 0.02, 'Linear, $\\delta=2\\%$', '--', '#1f77b4'),
        (1, 0.00, 'Log utility, no discount', '-', '#ff7f0e'),
        (1, 0.02, 'Log utility, $\\delta=2\\%$', '--', '#ff7f0e'),
        (2, 0.02, 'CRRA $\\sigma=2$, $\\delta=2\\%$', '--', '#2ca02c'),
    ]
    
    plt.figure(figsize=FIG_SINGLE)
    
    for sigma, delta, label, ls, color in specs:
        losses = []
        print(f"Computing: {label}...")
        for T in T_grid:
            loss = compute_welfare_loss(alpha, S, M0, T, sigma, delta)
            losses.append(loss)
            print(f"  T={T}: {loss:.1f}%")
        
        plt.plot(T_grid, losses, linestyle=ls, color=color, linewidth=1.5, 
                 marker='o', markersize=4, label=label)
    
    plt.xlabel('Planning horizon $T$ (years)')
    plt.ylabel('Welfare loss (%)')
    # NO title - caption handles this
    plt.legend(fontsize=8, loc='center right', frameon=False)
    # NO grid
    plt.xlim(0, 210)
    plt.ylim(0, 100)
    
    # NO annotations - let caption handle explanation
    
    plt.tight_layout()
    save_figure('figure_welfare_loss_sensitivity')


if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING WELFARE LOSS SENSITIVITY FIGURE")
    print("=" * 60)
    generate_welfare_loss_sensitivity_figure()
