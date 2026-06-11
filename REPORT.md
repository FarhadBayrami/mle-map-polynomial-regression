# Homework 4 Report — MLE and MAP
**Statistical and Mathematical Methods for AI — UniBO 2022-2023**  
**Student:** Farhad Bayrami

---

## 1. Problem Setup

Given a dataset $\mathcal{D} = \{X, Y\}$ with $x_i \in \mathbb{R}$ and $y_i \in \mathbb{R}$, we model:

$$y_i = \theta_1 + \theta_2 x_i + \cdots + \theta_K x_i^{K-1} + e_i$$

The feature map uses the classical Vandermonde matrix:

$$\Phi(X)_{ij} = \phi_j(x_i) = x_i^{j-1}, \quad i=1\ldots N,\; j=1\ldots K$$

So the model becomes $Y = \Phi(X)\theta + e$.

---

## 2. MLE — Gaussian Noise

Assuming $e_i \sim \mathcal{N}(0, \sigma^2)$, the conditional likelihood is:

$$p_\theta(y_i | x_i) = \mathcal{N}(f_\theta(x_i), \sigma^2)$$

Maximising the log-likelihood is equivalent to minimising the least squares loss:

$$\theta_{\text{MLE}} = \arg\min_\theta \frac{1}{2\sigma^2} \|\Phi(X)\theta - Y\|_2^2$$

**Closed-form solution (Normal Equations):**

$$\theta_{\text{MLE}} = (\Phi^T \Phi)^{-1} \Phi^T Y$$

---

## 3. MAP — Gaussian Prior

Assuming a Gaussian prior $\theta \sim \mathcal{N}(0, \sigma^2_\theta I)$ and applying Bayes' theorem:

$$\theta_{\text{MAP}} = \arg\min_\theta \sum_{i=1}^N -\log p(y_i|x_i,\theta) - \log p(\theta)$$

This gives the Ridge Regression problem:

$$\theta_{\text{MAP}} = \arg\min_\theta \frac{1}{2\sigma^2}\|\Phi(X)\theta - Y\|^2 + \frac{\lambda}{2}\|\theta\|^2$$

where $\lambda = \sigma^2 / \sigma^2_\theta$ controls the regularisation strength.

**Closed-form solution:**

$$\theta_{\text{MAP}} = (\Phi^T\Phi + \lambda I)^{-1}\Phi^T Y$$

---

## 4. Solvers Implemented

Three optimisation methods were implemented and compared:

| Solver | Update rule |
|---|---|
| Normal Equations (NE) | Direct closed-form matrix solve |
| Gradient Descent (GD) | $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$ (full batch) |
| SGD | Same but over random mini-batches of size $B$ |

For GD on the MLE loss:
$$\nabla_\theta \mathcal{L} = \frac{1}{N}\Phi^T(\Phi\theta - Y)$$

For MAP (GD):
$$\nabla_\theta \mathcal{L} = \frac{1}{N}\Phi^T(\Phi\theta - Y) + \lambda\theta$$

---

## 5. Experiments — Gaussian Model

### 5.1 Training/Test Error vs K

- For $K < K_{\text{true}}$: underfitting — both train and test error are high.
- For $K = K_{\text{true}}$: best generalisation.
- For $K > K_{\text{true}}$: MLE overfits (test error rises sharply); MAP remains stable thanks to regularisation.

### 5.2 Regression Curves

- MLE with $K \gg K_{\text{true}}$ produces highly oscillatory fits (Runge's phenomenon).
- MAP with appropriate $\lambda$ suppresses oscillations even at large $K$.
- Large $\lambda$ leads to underfitting (over-smoothed curves).

### 5.3 Parameter Error vs K

For $K > K_{\text{true}}$, $\theta_{\text{true}}$ is zero-padded and:

$$\text{Err}(\theta) = \frac{\|\theta - \theta_{\text{true}}\|_2}{\|\theta_{\text{true}}\|_2}$$

- MLE parameter error grows quickly with $K$ (ill-conditioning of $\Phi^T\Phi$).
- MAP controls this growth; smaller $\lambda$ → closer to MLE behaviour, larger $\lambda$ → more shrinkage toward zero.

### 5.4 Effect of N

- With more training data, MLE becomes more reliable even for large $K$ (variance decreases).
- MAP advantage shrinks as $N$ grows — consistent with the asymptotic equivalence of MLE and MAP for large datasets.

### 5.5 Solver Comparison

All three solvers converge to the same solution when properly tuned:
- Normal Equations: exact, fast for small $K$, numerically unstable for very large $K$.
- GD: reliable, slower, requires tuning of learning rate.
- SGD: fastest per epoch, noisier convergence, needs more epochs.

---

## 6. Poisson Noise Model

When observations follow $y_i \sim \text{Poi}(\lambda_i)$ with $\lambda_i = f_\theta(x_i)$:

$$p_\theta(y_i | x_i) = \frac{\lambda_i^{y_i} e^{-\lambda_i}}{y_i!}$$

### MLE (Poisson)

$$\theta_{\text{MLE}} = \arg\min_\theta \sum_{i=1}^N \left[ -y_i \log\lambda_i + \lambda_i \right]$$

Gradient:
$$\nabla_\theta \mathcal{L} = \frac{1}{N}\Phi^T\left(1 - \frac{Y}{\lambda}\right)$$

### MAP (Poisson + Gaussian Prior)

$$\theta_{\text{MAP}} = \arg\min_\theta \sum_{i=1}^N \left[ -y_i \log\lambda_i + \lambda_i \right] + \frac{\lambda}{2}\|\theta\|^2$$

Both solved via Gradient Descent (no closed form due to the Poisson NLL).

### Observations
- Poisson MLE behaves similarly to Gaussian MLE in terms of overfitting at large $K$.
- MAP regularisation is equally effective at controlling variance.
- The Poisson model is more appropriate when $y_i$ are count data (non-negative integers).

---

## 7. Conclusions

| Setting | MLE | MAP |
|---|---|---|
| $K = K_{\text{true}}$ | Good fit | Good fit (λ matters less) |
| $K > K_{\text{true}}$, small N | Overfits | Stable with right λ |
| $K > K_{\text{true}}$, large N | Recovers | ≈ MLE |
| Poisson noise | Works via GD | Works via GD + prior |

Key takeaway: **MAP = MLE + regularisation**. When $K$ is unknown and $N$ is limited, MAP with cross-validated $\lambda$ is the safer choice.
