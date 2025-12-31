# Replication Package: Optimal Spending Rules for Endowed Institutions

This repository contains the replication materials for "Optimal Spending Rules for Endowed Institutions: A Finite-Horizon Theory of Intertemporal Giving" by Evgeny Havkin and Santiago Schnell.

**Version:** Revised December 2024 (post peer review)

## Requirements

- Python 3.8 or higher (tested with Python 3.10.12)

Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Repository Structure

```
├── requirements.txt                 # Python dependencies (with version bounds)
├── README.md                        # This file
├── run_all.py                       # Master execution script (NEW)
│
├── Python Scripts:
│   ├── calibration_analysis_fixed.py    # Core optimization, Tables 1–2, Figures 1–2
│   ├── robustness_analysis.py           # Multi-specification robustness, Table 3
│   ├── crra_analysis.py                 # CRRA sensitivity, Table 4, Figure 6
│   ├── welfare_loss_analysis.py         # Baseline welfare loss, Table 5, Figures 3–4
│   ├── welfare_loss_sensitivity_v2.py   # Welfare loss sensitivity, Table 6, Figure 7
│   ├── generate_welfare_loss_figure.py  # Generates Figure 7
│   ├── endowment_tax_analysis.py        # Endowment tax analysis, Tables 7–8, Figure 5
│   └── ce_verification.py               # Certainty equivalence verification (ENHANCED)
│
└── Generated Figures (PDF):
    ├── figure_gamma_vs_T.pdf              # Optimal payout vs. horizon (Figure 1)
    ├── figure_gamma_vs_T_overlay.pdf      # Exact vs. asymptotic overlay (Appendix)
    ├── figure_total_vs_gamma.pdf          # Total giving vs. payout rate (Figure 2)
    ├── figure_welfare_loss.pdf            # Welfare loss baseline (Figure 3)
    ├── figure_policy_comparison.pdf       # Policy comparison (Figure 4)
    ├── figure_endowment_tax.pdf           # Endowment tax effects (Figure 5)
    ├── figure_endowment_tax_detail.pdf    # Detailed tax effects (Appendix)
    ├── figure_crra_sensitivity.pdf        # CRRA sensitivity (Figure 6)
    ├── figure_welfare_loss_sensitivity.pdf # Welfare loss by specification (Figure 7)
    ├── figure_inflow_sensitivity.pdf      # Inflow sensitivity (Appendix)
    └── figure_ce_verification.pdf         # Certainty equivalence verification (Appendix)
```

## Reproducing Results

### Quick Start: Master Execution Script (Recommended)

The easiest way to reproduce all results:

```bash
# Full analysis (publication quality, ~15-20 minutes)
python run_all.py

# Quick verification (reduced accuracy, ~2-3 minutes)
python run_all.py --fast

# Generate figures only
python run_all.py --figures
```

The master script runs all analyses in sequence, checks for expected outputs, and reports timing.

### Manual Execution: Run Scripts Individually

```bash
python calibration_analysis_fixed.py
python robustness_analysis.py
python crra_analysis.py
python welfare_loss_analysis.py
python welfare_loss_sensitivity_v2.py
python generate_welfare_loss_figure.py
python endowment_tax_analysis.py
python ce_verification.py --fast  # Use --fast for quick verification
```

---

## Script-by-Script Guide

### 1. calibration_analysis_fixed.py
**Runtime:** ~10 seconds

**Generates:**
- **Table 1**: Baseline calibration of optimal payout rates
- **Table 2**: Asymptotic approximation accuracy
- **Figure 1**: `figure_gamma_vs_T.pdf` — Optimal γ* vs. planning horizon T
- **Figure 2**: `figure_total_vs_gamma.pdf` — Total giving vs. payout rate
- **Appendix**: `figure_gamma_vs_T_overlay.pdf`, `figure_inflow_sensitivity.pdf`

**Key equations verified:**
- Optimal payout from FOC: γ* that maximizes W(γ) = Σ C_t
- Critical threshold: γ_c = 1 − 1/α ≈ 4.76% for α = 1.05
- Asymptotic approximation: γ* ≈ 1/(T − α/(α−1)) for large T

**Sample output to verify:**
```
T=20, α=1.05: γ* = 13.19%
T=50, α=1.05: γ* = 3.56%
T=100, α=1.05: γ* = 1.33%
Critical threshold γ_c = 4.76%
```

---

### 2. robustness_analysis.py
**Runtime:** ~15 seconds

**Generates:**
- **Table 3**: Systematic robustness across 9 specification combinations
  - Welfare: Linear, Log (σ=1), CRRA (σ=2)
  - Discounting: β = 1.00, 0.98, 0.96

**Key finding:** Horizon effect (∂γ*/∂T < 0) is **fully robust** across all specifications.

**Sample output to verify:**
```
Linear, β=1.00:   T=20 → 13.19%, T=50 → 3.56%, T=100 → 1.33%
Log, β=0.98:      T=20 → 14.52%, T=50 → 4.30%, T=100 → 1.78%
CRRA σ=2, β=0.96: T=20 → 15.89%, T=50 → 5.12%, T=100 → 2.34%
```

---

### 3. crra_analysis.py
**Runtime:** ~15 seconds

**Generates:**
- **Table 4**: CRRA sensitivity analysis
- **Figure 6**: `figure_crra_sensitivity.pdf` — Optimal γ* vs. T for different σ

**Key findings:** 
- Higher risk aversion (σ) increases optimal payout at long horizons (smoothing motive)
- But *decreases* optimal payout at short horizons (crossover effect—see paper Section 3.5)

**Sample output to verify:**
```
σ=0 (linear): γ*(T=50) = 3.56%
σ=1 (log):    γ*(T=50) = 3.92%
σ=2 (CRRA):   γ*(T=50) = 4.14%
```

---

### 4. welfare_loss_analysis.py
**Runtime:** ~5 seconds

**Generates:**
- **Table 5**: Welfare loss from constant-rate restriction (baseline criterion)
- **Figure 3**: `figure_welfare_loss.pdf` — Welfare loss vs. horizon
- **Figure 4**: `figure_policy_comparison.pdf` — Constant vs. bang-bang policy paths

**Key equations:**
- Bang-bang welfare: W* = α^T · M_0 + S · H_1(α, T)
- Welfare loss: (W* − W(γ*)) / W*

**Sample output to verify:**
```
T=20, r=5%: Loss = 50.3%
T=50, r=5%: Loss = 77.5%
T=100, r=5%: Loss = 90.1%
T=200, r=5%: Loss = 95.7%
```

**Note:** These are **upper bounds** under the baseline (linear, undiscounted) criterion—an operationally infeasible benchmark. See Table 6 for realistic welfare specifications.

---

### 5. welfare_loss_sensitivity_v2.py
**Runtime:** ~2–3 minutes

**Generates:**
- **Table 6**: Welfare loss sensitivity to specification
  - Discounting: δ = 0%, 2%, 4%
  - Impact function: Linear (σ=0), Log (σ=1), CRRA (σ=2)

**Key finding:** Under CRRA utility, the optimal time-varying policy is smooth (not bang-bang), so constant rules achieve nearly optimal welfare. Losses shrink from 50–95% (baseline) to under 15% at T=50.

**Sample output to verify:**
```
Linear, δ=0:    T=50 → 77.5%
Log, δ=0:       T=50 → 11.1%
Log, δ=2%:      T=50 → 5.7%
CRRA σ=2, δ=2%: T=50 → 12.4%
```

---

### 6. generate_welfare_loss_figure.py
**Runtime:** ~2–3 minutes

**Generates:**
- **Figure 7**: `figure_welfare_loss_sensitivity.pdf` — Welfare loss by specification

Visualizes Table 6, showing that baseline losses are upper bounds.

---

### 7. endowment_tax_analysis.py
**Runtime:** ~10 seconds

**Generates:**
- **Table 7**: Optimal payout under OBBBA 2025 endowment tax tiers
- **Table 8**: Reduction in charitable capacity from taxation
- **Figure 5**: `figure_endowment_tax.pdf` — Tax impact on optimal payout
- **Appendix**: `figure_endowment_tax_detail.pdf`

**Tax tiers modeled (OBBBA 2025):**
- Tier 1: τ = 1.4% (baseline TCJA 2017)
- Tier 2: τ = 4%
- Tier 3: τ = 8%

**Key equations:**
- After-tax return: α_τ = 1 + (r − τ) = α − τ
- Charitable capacity reduction: (W_baseline − W_τ) / W_baseline

**Sample output to verify:**
```
Tax Tier 1 (1.4%): Charitable capacity reduction = 1.85% at T=50
Tax Tier 2 (4%):   Charitable capacity reduction = 5.17% at T=50
Tax Tier 3 (8%):   Charitable capacity reduction = 9.98% at T=50
```

---

### 8. ce_verification.py (Enhanced)
**Runtime:** ~5–10 minutes (full) or ~30–60 seconds (--fast)

**Generates:**
- **Appendix Figure**: `figure_ce_verification.pdf` — Monte Carlo verification
- **Table**: Volatility sensitivity analysis (NEW)

**Purpose:** Verifies Proposition 6 (certainty equivalence) via simulation.

**New feature:** Volatility sensitivity table showing mean welfare matches deterministic across σ = 10%–30%, confirming CE robustness.

**Usage:**
```bash
python ce_verification.py --fast  # Quick verification (1,000 simulations)
python ce_verification.py         # Full verification (10,000 simulations)
```

**Random seed:** `np.random.seed(42)` for reproducibility.

---

## Verification Checklist

After running all scripts, verify outputs match these key values:

| Output | Script | Key Values |
|--------|--------|------------|
| Table 1 | calibration_analysis_fixed.py | γ*(T=20, α=1.05) = 13.19% |
| Table 2 | calibration_analysis_fixed.py | Asymptotic error < 5% for T≥100 |
| Table 3 | robustness_analysis.py | Horizon effect robust across all 9 specs |
| Table 4 | crra_analysis.py | γ*(σ=1, T=50) = 3.92% |
| Table 5 | welfare_loss_analysis.py | Loss = 77.5% at T=50, r=5% (baseline) |
| Table 6 | welfare_loss_sensitivity_v2.py | Loss = 5.7% at T=50 (log, δ=2%) |
| Table 7 | endowment_tax_analysis.py | Δγ* = +0.14pp at T=50, 8% tax |
| Table 8 | endowment_tax_analysis.py | Capacity reduction = 9.98% at T=50, 8% tax |
| Figure 1 | calibration_analysis_fixed.py | Monotonic decrease in γ* with T |
| Figure 7 | generate_welfare_loss_figure.py | CRRA losses approach 0 at long horizons |

---

## Core Model Equations

The Python scripts implement the following model from the paper:

**Capital dynamics:**
```
M_{t+1} = (1 − γ)(α M_t + S)
```

**Grants:**
```
C_t = γ(α M_t + S)
```

**Welfare (baseline):**
```
W(γ) = Σ_{t=0}^{T-1} C_t
```

**Welfare (discounted CRRA):**
```
W(γ) = Σ_{t=0}^{T-1} β^t · u(C_t)
where u(C) = C^{1-σ}/(1-σ) for σ ≠ 1, u(C) = log(C) for σ = 1
```

**Critical threshold:**
```
γ_c = 1 − 1/α ≈ 4.76% for α = 1.05
```

**Bang-bang welfare (linear case):**
```
W* = α^T · M_0 + S · (α^T − 1)/(α − 1)
```

---

## Key Theoretical Results

1. **Robust finding:** Horizon effect (∂γ*/∂T < 0) holds across ALL welfare specifications
2. **Specification-dependent:** Return effect (∂γ*/∂α) sign depends on risk aversion σ
3. **CRRA crossover:** Higher σ *decreases* optimal payout at short horizons but *increases* it at long horizons
4. **Welfare loss interpretation:** Baseline 50–95% losses are upper bounds under an operationally infeasible benchmark; under CRRA with smoothing preferences, constant rules are nearly optimal
5. **Policy application:** OBBBA 2025 endowment tax (1.4%–8%) reduces charitable capacity by 2–50% depending on horizon

---

## Contact

For questions about replication, contact: santiago.schnell@dartmouth.edu

## License

MIT License. See LICENSE file for details.
