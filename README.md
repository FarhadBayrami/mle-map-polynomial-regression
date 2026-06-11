# Homework 4 — MLE and MAP
**Statistical and Mathematical Methods for AI — UniBO 2022-2023**

Implementation of Maximum Likelihood Estimation (MLE) and Maximum A Posteriori (MAP) for polynomial regression under two noise models: Gaussian and Poisson.

---

## What's implemented

### Gaussian noise model
| Component | Details |
|---|---|
| MLE | Normal Equations, Gradient Descent, SGD |
| MAP | Normal Equations (Ridge), Gradient Descent |
| Vandermonde matrix | `phi_j(x) = x^{j-1}` |

### Poisson noise model
| Component | Details |
|---|---|
| MLE | GD on negative log-likelihood |
| MAP | GD on NLL + Gaussian prior |

### Experiments
1. Train/test error vs K (varying polynomial degree)
2. Regression curves for K < K_true, K = K_true, K > K_true
3. MAP curves with different λ values
4. Relative parameter error `||θ - θ_true|| / ||θ_true||` vs K
5. Effect of dataset size N
6. Solver comparison: Normal Equations vs GD vs SGD
7. All experiments repeated for the Poisson model

---

## Setup

```bash
pip install numpy matplotlib scipy
```

## Run

```bash
python homework4_mle_map.py
```

Saves figures: `exp_varying_K.png`, `exp_mle_curves.png`, `exp_map_curves.png`, `exp_param_error.png`, `exp_vary_N.png`, `exp_poisson.png`

---

## Key equations

**MLE (Gaussian)** — equivalent to Least Squares:
```
θ_MLE = argmin  (1/2σ²) ||Φ(X)θ - Y||²
      = (Φᵀ Φ)⁻¹ Φᵀ Y
```

**MAP (Gaussian prior)** — equivalent to Ridge Regression:
```
θ_MAP = argmin  (1/2σ²)||Φ(X)θ - Y||² + (1/2σ²_θ)||θ||²
      = (Φᵀ Φ + λI)⁻¹ Φᵀ Y      where λ = σ²/σ²_θ
```

**MLE (Poisson)**:
```
θ_MLE = argmin  Σᵢ [ -yᵢ log(λᵢ) + λᵢ ]     λᵢ = Φ(xᵢ)θ
```

**MAP (Poisson + Gaussian prior)**:
```
θ_MAP = argmin  Σᵢ [ -yᵢ log(λᵢ) + λᵢ ] + (λ/2)||θ||²
```
