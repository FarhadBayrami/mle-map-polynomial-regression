<div align="center">

# 📈 MLE and MAP for Polynomial Regression
### Maximum Likelihood and Maximum A Posteriori Estimation under Gaussian and Poisson Noise Models

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="https://img.shields.io/badge/Noise%20Models-Gaussian%20%2B%20Poisson-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Solvers-Normal%20Eq%20%7C%20GD%20%7C%20SGD-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/Course-Statistical%20Methods%20for%20AI-orange?style=flat-square"/>
</p>

*From-scratch implementation of MLE and MAP polynomial regression with Gaussian and Poisson noise, comparing Normal Equations, Gradient Descent, and SGD solvers.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Methods Implemented](#-methods-implemented)
- [Key Equations](#-key-equations)
- [Experiments](#-experiments)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [References](#-references)
- [Author](#-author)

---

## 🔬 Overview

This project implements **Maximum Likelihood Estimation (MLE)** and **Maximum A Posteriori (MAP)** estimation for polynomial regression from scratch, under two different noise assumptions:

- **Gaussian noise** — leads to least squares (MLE) and ridge regression (MAP)
- **Poisson noise** — requires iterative gradient-based optimisation

All solvers are implemented without ML libraries — using only NumPy, SciPy, and Matplotlib.

---

## ⚙️ Methods Implemented

### Gaussian Noise Model

| Method | Solver | Details |
|--------|--------|---------|
| MLE | Normal Equations | Closed-form least squares solution |
| MLE | Gradient Descent | Iterative minimisation of squared error |
| MLE | SGD | Stochastic gradient descent |
| MAP | Normal Equations (Ridge) | Closed-form with L2 regularisation |
| MAP | Gradient Descent | Iterative with Gaussian prior |

### Poisson Noise Model

| Method | Solver | Details |
|--------|--------|---------|
| MLE | Gradient Descent | Minimises negative log-likelihood |
| MAP | Gradient Descent | NLL + Gaussian prior regularisation |

**Feature map:** Vandermonde matrix — `φⱼ(x) = x^(j−1)`

---

## 📐 Key Equations

**MLE — Gaussian** (equivalent to Least Squares):

| | Formula |
|-|---------|
| θ_MLE | = (ΦᵀΦ)⁻¹ Φᵀ Y |

**MAP — Gaussian prior** (equivalent to Ridge Regression):

| | Formula |
|-|---------|
| θ_MAP | = (ΦᵀΦ + λI)⁻¹ Φᵀ Y |
| where | λ = σ² / σ²_θ |

**MLE — Poisson:**

| | Formula |
|-|---------|
| θ_MLE | = argmin Σᵢ [ −yᵢ log(λᵢ) + λᵢ ] |
| where | λᵢ = Φ(xᵢ)θ |

**MAP — Poisson + Gaussian prior:**

| | Formula |
|-|---------|
| θ_MAP | = argmin Σᵢ [ −yᵢ log(λᵢ) + λᵢ ] + (λ/2)\|\|θ\|\|² |---

## 🧪 Experiments

| # | Experiment | Description |
|---|------------|-------------|
| 1 | Train/test error vs K | Effect of polynomial degree on generalisation |
| 2 | Regression curves | Underfitting (K < K_true), correct (K = K_true), overfitting (K > K_true) |
| 3 | MAP with varying λ | Effect of regularisation strength on fit |
| 4 | Parameter error vs K | `‖θ − θ_true‖ / ‖θ_true‖` as degree increases |
| 5 | Effect of dataset size N | How N affects estimation quality |
| 6 | Solver comparison | Normal Equations vs GD vs SGD |
| 7 | Poisson model | All experiments repeated under Poisson noise |

**Output figures:** `exp_varying_K.png`, `exp_mle_curves.png`, `exp_map_curves.png`, `exp_param_error.png`, `exp_vary_N.png`, `exp_poisson.png`

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run

```bash
# 1. Clone the repository
git clone https://github.com/FarhadBayrami/mle-map-polynomial-regression.git
cd mle-map-polynomial-regression

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run all experiments
python homework4_mle_map.py
```

All output figures are saved in the working directory.

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `homework4_mle_map.py` | Full implementation: MLE, MAP, solvers, all experiments |
| `REPORT.md` | Written report with analysis and discussion |
| `requirements.txt` | Python dependencies |
| `LICENSE` | MIT License |
| `CITATION.cff` | How to cite this work |
| `README.md` | Project documentation |

---

## 📚 References

1. Bishop, C.M. — *Pattern Recognition and Machine Learning*, Springer, 2006. (Chapter 3)
2. Murphy, K.P. — *Machine Learning: A Probabilistic Perspective*, MIT Press, 2012.
3. Statistical and Mathematical Methods for AI — University of Bologna, A.Y. 2022–2023.

---

## 👤 Author

**Farhad Bayrami**
MSc Student — University of Bologna
📧 [farhad.bayrami@studio.unibo.it](mailto:farhad.bayrami@studio.unibo.it)
🔗 [GitHub](https://github.com/FarhadBayrami)

---

<div align="center">
  <sub>Built with ❤️ as part of Statistical and Mathematical Methods for AI — University of Bologna · 2022–2023</sub>
</div>
