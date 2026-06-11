"""
Homework 4: MLE and MAP
Statistical and Mathematical Methods for AI — UniBO 2022-2023

Covers:
  - Gaussian noise model: MLE (= Least Squares) and MAP (= Ridge Regression)
  - Poisson noise model: MLE and MAP via Gradient Descent
  - Solvers: Normal Equations, GD, SGD
  - Experiments: varying K, N, lambda, solver comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln   # for Poisson log-likelihood
from math import factorial

# ─────────────────────────────────────────────────────────────
# 1.  DATA GENERATION
# ─────────────────────────────────────────────────────────────

def vandermonde(X: np.ndarray, K: int) -> np.ndarray:
    """
    Classical Vandermonde matrix Phi(X) in R^{N x K}.
    Column j contains x^{j-1}, j = 1..K.
    """
    N = len(X)
    Phi = np.zeros((N, K))
    for j in range(K):
        Phi[:, j] = X ** j
    return Phi


def generate_dataset(K: int, N: int, a: float, b: float,
                     sigma2: float, noise: str = "gaussian",
                     theta_true: np.ndarray = None, seed: int = 0):
    """
    Build dataset D = {X, Y}.

    Parameters
    ----------
    K        : true polynomial degree + 1  (number of coefficients)
    N        : number of training points
    a, b     : interval for uniform sampling
    sigma2   : noise variance
    noise    : 'gaussian' or 'poisson'
    theta_true: if None defaults to all-ones vector of length K

    Returns
    -------
    X, Y, Phi, theta_true
    """
    rng = np.random.default_rng(seed)
    if theta_true is None:
        theta_true = np.ones(K)

    X   = rng.uniform(a, b, N)
    Phi = vandermonde(X, K)
    f   = Phi @ theta_true   # noiseless signal

    if noise == "gaussian":
        e = rng.normal(0, np.sqrt(sigma2), N)
        Y = f + e
    elif noise == "poisson":
        # lambda of Poisson must be positive; clip to avoid zeros
        lam = np.maximum(f, 1e-6)
        Y   = rng.poisson(lam).astype(float)
    else:
        raise ValueError(f"Unknown noise type: {noise}")

    return X, Y, Phi, theta_true


# ─────────────────────────────────────────────────────────────
# 2.  SOLVERS — GAUSSIAN MODEL
# ─────────────────────────────────────────────────────────────

def mle_normal_equations(X: np.ndarray, Y: np.ndarray, K: int) -> np.ndarray:
    """
    MLE via Normal Equations:  theta = (Phi^T Phi)^{-1} Phi^T Y
    """
    Phi = vandermonde(X, K)
    return np.linalg.lstsq(Phi, Y, rcond=None)[0]


def map_normal_equations(X: np.ndarray, Y: np.ndarray, K: int,
                         lam: float, sigma2: float = 1.0) -> np.ndarray:
    """
    MAP via Normal Equations (Ridge Regression):
        theta = (Phi^T Phi + lambda * I)^{-1} Phi^T Y
    where lambda = sigma2 / sigma2_theta  (regularisation strength).
    """
    Phi = vandermonde(X, K)
    A   = Phi.T @ Phi + lam * np.eye(K)
    b   = Phi.T @ Y
    return np.linalg.solve(A, b)


def mle_gd(X: np.ndarray, Y: np.ndarray, K: int,
           lr: float = 1e-3, epochs: int = 5000,
           tol: float = 1e-8) -> np.ndarray:
    """
    MLE via full-batch Gradient Descent on the MSE loss.
    Loss = (1/2N) || Phi theta - Y ||^2
    Gradient = (1/N) Phi^T (Phi theta - Y)
    """
    Phi   = vandermonde(X, K)
    N     = len(Y)
    theta = np.zeros(K)
    for _ in range(epochs):
        grad  = (Phi.T @ (Phi @ theta - Y)) / N
        theta_new = theta - lr * grad
        if np.linalg.norm(theta_new - theta) < tol:
            break
        theta = theta_new
    return theta


def map_gd(X: np.ndarray, Y: np.ndarray, K: int, lam: float,
           lr: float = 1e-3, epochs: int = 5000,
           tol: float = 1e-8) -> np.ndarray:
    """
    MAP via full-batch Gradient Descent on regularised MSE.
    Loss = (1/2N)||Phi theta - Y||^2 + (lambda/2)||theta||^2
    """
    Phi   = vandermonde(X, K)
    N     = len(Y)
    theta = np.zeros(K)
    for _ in range(epochs):
        grad  = (Phi.T @ (Phi @ theta - Y)) / N + lam * theta
        theta_new = theta - lr * grad
        if np.linalg.norm(theta_new - theta) < tol:
            break
        theta = theta_new
    return theta


def mle_sgd(X: np.ndarray, Y: np.ndarray, K: int,
            lr: float = 1e-3, epochs: int = 50,
            batch_size: int = 16, seed: int = 1) -> np.ndarray:
    """
    MLE via mini-batch Stochastic Gradient Descent.
    """
    rng   = np.random.default_rng(seed)
    Phi   = vandermonde(X, K)
    N     = len(Y)
    theta = np.zeros(K)
    for _ in range(epochs):
        idx = rng.permutation(N)
        for start in range(0, N, batch_size):
            b_idx  = idx[start:start + batch_size]
            Phi_b  = Phi[b_idx]
            Y_b    = Y[b_idx]
            grad   = Phi_b.T @ (Phi_b @ theta - Y_b) / len(b_idx)
            theta -= lr * grad
    return theta


# ─────────────────────────────────────────────────────────────
# 3.  ERROR METRICS
# ─────────────────────────────────────────────────────────────

def predict(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    K   = len(theta)
    Phi = vandermonde(X, K)
    return Phi @ theta


def mse(theta: np.ndarray, X: np.ndarray, Y: np.ndarray) -> float:
    """Average squared error  (1/N) || f_theta(X) - Y ||^2"""
    return float(np.mean((predict(theta, X) - Y) ** 2))


def param_error(theta: np.ndarray, theta_true: np.ndarray) -> float:
    """
    Relative parameter error: || theta - theta_true_padded ||_2 / || theta_true ||_2
    theta_true is zero-padded to match len(theta).
    """
    K    = len(theta)
    Kt   = len(theta_true)
    pad  = np.zeros(K)
    pad[:Kt] = theta_true
    return float(np.linalg.norm(theta - pad) / (np.linalg.norm(theta_true) + 1e-12))


# ─────────────────────────────────────────────────────────────
# 4.  POISSON MODEL
# ─────────────────────────────────────────────────────────────

def poisson_neg_log_likelihood(theta: np.ndarray, Phi: np.ndarray,
                               Y: np.ndarray) -> float:
    """
    NLL = sum_i [ -y_i log(lambda_i) + lambda_i + log(y_i!) ]
    where lambda_i = Phi[i] @ theta  (clipped to be positive)
    """
    lam  = np.maximum(Phi @ theta, 1e-9)
    return float(np.sum(-Y * np.log(lam) + lam + gammaln(Y + 1)))


def poisson_nll_gradient(theta: np.ndarray, Phi: np.ndarray,
                         Y: np.ndarray) -> np.ndarray:
    """
    d NLL / d theta = Phi^T (1 - Y / lambda)
    """
    lam  = np.maximum(Phi @ theta, 1e-9)
    return Phi.T @ (1.0 - Y / lam)


def mle_poisson_gd(X: np.ndarray, Y: np.ndarray, K: int,
                   lr: float = 1e-4, epochs: int = 10000,
                   tol: float = 1e-8) -> np.ndarray:
    """MLE for Poisson model via GD."""
    Phi   = vandermonde(X, K)
    theta = np.ones(K) * 0.1
    for _ in range(epochs):
        grad      = poisson_nll_gradient(theta, Phi, Y) / len(Y)
        theta_new = theta - lr * grad
        if np.linalg.norm(theta_new - theta) < tol:
            break
        theta = theta_new
    return theta


def map_poisson_gd(X: np.ndarray, Y: np.ndarray, K: int, lam: float,
                   lr: float = 1e-4, epochs: int = 10000,
                   tol: float = 1e-8) -> np.ndarray:
    """MAP for Poisson model via GD (Gaussian prior on theta)."""
    Phi   = vandermonde(X, K)
    theta = np.ones(K) * 0.1
    for _ in range(epochs):
        grad      = poisson_nll_gradient(theta, Phi, Y) / len(Y) + lam * theta
        theta_new = theta - lr * grad
        if np.linalg.norm(theta_new - theta) < tol:
            break
        theta = theta_new
    return theta


# ─────────────────────────────────────────────────────────────
# 5.  EXPERIMENTS — GAUSSIAN MODEL
# ─────────────────────────────────────────────────────────────

def exp_varying_K(K_true=4, N=50, a=-1, b=1, sigma2=0.5,
                  K_range=range(1, 12), lam=0.1):
    """
    Plot training/test error vs K for MLE and MAP (Gaussian).
    """
    X_tr, Y_tr, _, theta_true = generate_dataset(K_true, N, a, b, sigma2, seed=0)
    X_te, Y_te, _, _          = generate_dataset(K_true, 200, a, b, sigma2, seed=1)

    mle_tr, mle_te = [], []
    map_tr, map_te = [], []

    for K in K_range:
        th_mle = mle_normal_equations(X_tr, Y_tr, K)
        th_map = map_normal_equations(X_tr, Y_tr, K, lam)
        mle_tr.append(mse(th_mle, X_tr, Y_tr))
        mle_te.append(mse(th_mle, X_te, Y_te))
        map_tr.append(mse(th_map, X_tr, Y_tr))
        map_te.append(mse(th_map, X_te, Y_te))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(K_range, mle_tr, "o-", label="MLE train")
    axes[0].plot(K_range, mle_te, "s--", label="MLE test")
    axes[0].axvline(K_true, color="gray", linestyle=":", label=f"K_true={K_true}")
    axes[0].set(xlabel="K", ylabel="MSE", title="MLE: Train vs Test Error")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(K_range, map_tr, "o-", label=f"MAP train (λ={lam})")
    axes[1].plot(K_range, map_te, "s--", label=f"MAP test (λ={lam})")
    axes[1].axvline(K_true, color="gray", linestyle=":", label=f"K_true={K_true}")
    axes[1].set(xlabel="K", ylabel="MSE", title="MAP: Train vs Test Error")
    axes[1].legend(); axes[1].grid(True)

    plt.suptitle(f"Varying K  (N={N}, σ²={sigma2})", fontsize=13)
    plt.tight_layout()
    plt.savefig("exp_varying_K.png", dpi=150)
    plt.show()
    print("[exp_varying_K] Figure saved.")


def exp_regression_curves(K_true=4, N=50, a=-1, b=1, sigma2=0.5,
                           K_list=(2, 4, 8), lam_list=(0.001, 0.1, 10.0)):
    """
    For each K in K_list, plot the MLE and MAP regression curves.
    """
    X_tr, Y_tr, _, theta_true = generate_dataset(K_true, N, a, b, sigma2, seed=0)
    X_te, Y_te, _, _          = generate_dataset(K_true, 200, a, b, sigma2, seed=1)
    x_plot = np.linspace(a, b, 300)

    # ── MLE curves ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(K_list), figsize=(5 * len(K_list), 4))
    for ax, K in zip(axes, K_list):
        th = mle_normal_equations(X_tr, Y_tr, K)
        y_hat = predict(th, x_plot)
        ax.scatter(X_tr, Y_tr, s=15, alpha=0.6, label="Train", color="steelblue")
        ax.scatter(X_te, Y_te, s=15, alpha=0.3, label="Test",  color="orange")
        ax.plot(x_plot, y_hat, "r-", lw=2, label=f"MLE K={K}")
        ax.set(title=f"MLE  K={K}", xlabel="x", ylabel="y")
        ax.legend(fontsize=8); ax.grid(True)
    plt.suptitle("MLE regression curves", fontsize=13)
    plt.tight_layout()
    plt.savefig("exp_mle_curves.png", dpi=150)
    plt.show()

    # ── MAP curves (K = K_true, varying lambda) ─────────────────
    fig, axes = plt.subplots(1, len(lam_list), figsize=(5 * len(lam_list), 4))
    for ax, lam in zip(axes, lam_list):
        th = map_normal_equations(X_tr, Y_tr, K_true + 3, lam)
        y_hat = predict(th, x_plot)
        ax.scatter(X_tr, Y_tr, s=15, alpha=0.6, label="Train", color="steelblue")
        ax.scatter(X_te, Y_te, s=15, alpha=0.3, label="Test",  color="orange")
        ax.plot(x_plot, y_hat, "r-", lw=2, label=f"MAP λ={lam}")
        ax.set(title=f"MAP  λ={lam}  K={K_true+3}", xlabel="x", ylabel="y")
        ax.legend(fontsize=8); ax.grid(True)
    plt.suptitle("MAP regression curves (K > K_true)", fontsize=13)
    plt.tight_layout()
    plt.savefig("exp_map_curves.png", dpi=150)
    plt.show()
    print("[exp_regression_curves] Figures saved.")


def exp_param_error(K_true=4, N=50, a=-1, b=1, sigma2=0.5,
                    K_range=range(4, 15), lam_list=(0.001, 0.01, 0.1, 1.0)):
    """
    Relative parameter error Err(theta) vs K for MLE and MAP.
    """
    X_tr, Y_tr, _, theta_true = generate_dataset(K_true, N, a, b, sigma2, seed=0)

    mle_errs = []
    for K in K_range:
        th = mle_normal_equations(X_tr, Y_tr, K)
        mle_errs.append(param_error(th, theta_true))

    plt.figure(figsize=(8, 5))
    plt.plot(K_range, mle_errs, "ko-", label="MLE")

    for lam in lam_list:
        map_errs = []
        for K in K_range:
            th = map_normal_equations(X_tr, Y_tr, K, lam)
            map_errs.append(param_error(th, theta_true))
        plt.plot(K_range, map_errs, "s--", label=f"MAP λ={lam}")

    plt.axvline(K_true, color="gray", linestyle=":", label=f"K_true={K_true}")
    plt.xlabel("K"); plt.ylabel("Relative parameter error")
    plt.title("Parameter error vs K"); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig("exp_param_error.png", dpi=150)
    plt.show()
    print("[exp_param_error] Figure saved.")


def exp_vary_N(K_true=4, a=-1, b=1, sigma2=0.5, K_fit=7,
               N_list=(20, 50, 100, 300, 1000), lam=0.1):
    """
    Compare MLE vs MAP test error as N grows (K fixed > K_true).
    """
    mle_te, map_te = [], []
    for N in N_list:
        X_tr, Y_tr, _, _ = generate_dataset(K_true, N,   a, b, sigma2, seed=0)
        X_te, Y_te, _, _ = generate_dataset(K_true, 500, a, b, sigma2, seed=2)
        th_mle = mle_normal_equations(X_tr, Y_tr, K_fit)
        th_map = map_normal_equations(X_tr, Y_tr, K_fit, lam)
        mle_te.append(mse(th_mle, X_te, Y_te))
        map_te.append(mse(th_map, X_te, Y_te))

    plt.figure(figsize=(7, 4))
    plt.semilogx(N_list, mle_te, "o-", label="MLE test")
    plt.semilogx(N_list, map_te, "s--", label=f"MAP test (λ={lam})")
    plt.xlabel("N (log scale)"); plt.ylabel("Test MSE")
    plt.title(f"Effect of N  (K_fit={K_fit}, K_true={K_true})")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig("exp_vary_N.png", dpi=150)
    plt.show()
    print("[exp_vary_N] Figure saved.")


def exp_solver_comparison(K_true=4, N=100, a=-1, b=1, sigma2=0.5,
                          K_fit=6, lam=0.1):
    """
    Compare Normal Equations, GD, SGD for MLE and MAP.
    """
    X_tr, Y_tr, _, _ = generate_dataset(K_true, N,   a, b, sigma2, seed=0)
    X_te, Y_te, _, _ = generate_dataset(K_true, 500, a, b, sigma2, seed=2)

    results = {}
    results["MLE NE"]  = mle_normal_equations(X_tr, Y_tr, K_fit)
    results["MLE GD"]  = mle_gd(X_tr, Y_tr, K_fit, lr=1e-2, epochs=10000)
    results["MLE SGD"] = mle_sgd(X_tr, Y_tr, K_fit, lr=1e-2, epochs=200, batch_size=16)
    results["MAP NE"]  = map_normal_equations(X_tr, Y_tr, K_fit, lam)
    results["MAP GD"]  = map_gd(X_tr, Y_tr, K_fit, lam, lr=1e-2, epochs=10000)

    print("\n── Solver Comparison ──────────────────────────────────────")
    print(f"{'Solver':<12}  {'Train MSE':>12}  {'Test MSE':>12}")
    print("-" * 40)
    for name, th in results.items():
        print(f"{name:<12}  {mse(th, X_tr, Y_tr):>12.6f}  {mse(th, X_te, Y_te):>12.6f}")
    print("───────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────
# 6.  EXPERIMENTS — POISSON MODEL
# ─────────────────────────────────────────────────────────────

def exp_poisson(K_true=3, N=80, a=0.5, b=3.0,
                K_range=range(1, 10), lam_list=(0.01, 0.1, 1.0)):
    """
    Poisson noise model: MLE and MAP via GD, varying K.
    Note: a > 0 so that lambda = f(x) stays positive.
    """
    X_tr, Y_tr, _, theta_true = generate_dataset(
        K_true, N, a, b, sigma2=1.0, noise="poisson", seed=42
    )
    X_te, Y_te, _, _ = generate_dataset(
        K_true, 300, a, b, sigma2=1.0, noise="poisson", seed=99
    )

    mle_tr_errs, mle_te_errs = [], []
    for K in K_range:
        th = mle_poisson_gd(X_tr, Y_tr, K, lr=1e-4, epochs=5000)
        mle_tr_errs.append(mse(th, X_tr, Y_tr))
        mle_te_errs.append(mse(th, X_te, Y_te))

    plt.figure(figsize=(8, 5))
    plt.plot(K_range, mle_tr_errs, "o-", label="Poisson MLE train")
    plt.plot(K_range, mle_te_errs, "s--", label="Poisson MLE test")

    for lam in lam_list:
        map_te_errs = []
        for K in K_range:
            th = map_poisson_gd(X_tr, Y_tr, K, lam, lr=1e-4, epochs=5000)
            map_te_errs.append(mse(th, X_te, Y_te))
        plt.plot(K_range, map_te_errs, "--", label=f"Poisson MAP test λ={lam}")

    plt.axvline(K_true, color="gray", linestyle=":", label=f"K_true={K_true}")
    plt.xlabel("K"); plt.ylabel("MSE")
    plt.title("Poisson model: MLE vs MAP  (train/test error vs K)")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig("exp_poisson.png", dpi=150)
    plt.show()
    print("[exp_poisson] Figure saved.")


# ─────────────────────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Homework 4 — MLE and MAP")
    print("=" * 60)

    # ── Gaussian experiments ─────────────────────────────────
    print("\n[1] Varying K — train/test error")
    exp_varying_K(K_true=4, N=50, sigma2=0.5, K_range=range(1, 14), lam=0.1)

    print("\n[2] Regression curves (MLE and MAP)")
    exp_regression_curves(K_true=4, N=50, sigma2=0.5,
                          K_list=(2, 4, 9), lam_list=(0.001, 0.1, 10.0))

    print("\n[3] Parameter error vs K")
    exp_param_error(K_true=4, N=50, sigma2=0.5,
                    K_range=range(4, 15),
                    lam_list=(0.001, 0.01, 0.1, 1.0))

    print("\n[4] Effect of N")
    exp_vary_N(K_true=4, sigma2=0.5, K_fit=8,
               N_list=(20, 50, 100, 300, 1000), lam=0.1)

    print("\n[5] Solver comparison (NE vs GD vs SGD)")
    exp_solver_comparison(K_true=4, N=100, sigma2=0.5, K_fit=6, lam=0.1)

    # ── Poisson experiments ──────────────────────────────────
    print("\n[6] Poisson noise model")
    exp_poisson(K_true=3, N=80, a=0.5, b=3.0,
                K_range=range(1, 10),
                lam_list=(0.01, 0.1, 1.0))

    print("\nAll experiments complete. PNG figures saved in the working directory.")
