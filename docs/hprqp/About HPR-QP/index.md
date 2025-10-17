# About HPR-QP

HPR-QP is a GPU-accelerated solver for large-scale convex composite quadratic programming (CCQP). It is a GPU-based dual Halpern–Peaceman–Rachford solver built on the restricted Wolfe dual with symmetric Gauss–Seidel, range-space updates, and adaptive restart.

## Problem statement


**Problem form.** The CCQP solved by HPR-QP is:

```{math}
\begin{aligned}
\min_{x \in \mathbb{R}^n}\quad & \tfrac12\langle x, Qx\rangle + \langle c, x \rangle + \phi(x) \\
\text{s.t.}\quad & Ax \in \mathcal{K}, \\
\end{aligned}
```

where $Q:\mathbb{R}^n\to\mathbb{R}^n$ is a self-adjoint positive semidefinite linear operator, 
$c\in\mathbb{R}^n$ is a given vector, and $\phi:\mathbb{R}^n\to(-\infty,+\infty]$ is a proper, closed, and convex function.  
Here, $A:\mathbb{R}^n\to\mathbb{R}^m$ is a linear operator, and $\mathcal{K}$ is a simple polyhedral set:


```{math}
\mathcal{K} := \{\, y \in \mathbb{R}^m \mid -\infty \le l_i \le y_i \le u_i \le +\infty,\; 1 \le i \le m \,\}.
```

A key feature of our approach is that it does not require an explicit matrix representation of $Q$,  
which makes the proposed method particularly suitable for large-scale or matrix-free settings—e.g., when $Q$ is defined implicitly via Kronecker products or structured operators.

In particular, CCQP includes the classical convex QP (CQP) as an important special case:


```{math}
\begin{aligned}
\min_{x \in \mathbb{R}^n}\quad & \tfrac12\langle x, Qx\rangle + \langle c, x \rangle + \delta_{\mathcal{C}}(x) \\
\text{s.t.}\quad & Ax \in \mathcal{K}, \\
\end{aligned}
```

Here, $\(\delta_{\mathcal{C}}(\cdot)\)$ is the indicator function of the box constraint set $\mathcal{C}$:

```{math}
\mathcal{C} := \{\, x \in \mathbb{R}^n \mid L \le x \le U \,\},
\qquad
L \in (\mathbb{R} \cup \{-\infty\})^n,\quad
U \in (\mathbb{R} \cup \{+\infty\})^n.
```

**Dual form.** The novel restricted Wolfe dual problem is:

```{math}
\begin{aligned}
\min_{(y,w,z) \in \mathbb{R}^m \times \mathbb{R}^n \times \mathbb{R}^n} \quad
& \tfrac{1}{2}\langle w, Qw\rangle
  + \delta_{\mathcal{K}}^*(-y)
  + \phi^*(-z) \\
\text{s.t.} \quad
& -Qw + A^*y + z = c, \\
& w \in \mathcal{W}.
\end{aligned}
```

where $\mathcal{W}:=Range(Q)$, the range space of $Q$.



## Dual HPR method for Solving CCQP

HPR-QP is based on the Halpern–Peaceman–Rachford (HPR) method for convex composite quadratic programming (CCQP). Its a general HPR framework with semi-proximal terms for solving the restricted Wolfe dual problem. The base algorithm appears below, followed by its convergence guarantees and complexity properties, which in turn motivate the algorithmic enhancements described later.


### Base algorithm

Let $\sigma > 0$ be a given penalty parameter.  
Define the augmented Lagrangian function $L_{\sigma}(y,w,z; x)$ associated with problem (1.5) for any  
$(y,w,z,x) \in \mathbb{R}^m \times \mathcal{W} \times \mathbb{R}^n \times \mathbb{R}^n$ as follows:

```{math}
L_{\sigma}(y,w,z; x)
= \tfrac{1}{2}\langle w, Qw\rangle
  + \delta_{\mathcal{K}}^*(-y)
  + \phi^*(-z)
  + \langle x, -Qw + A^*y + z - c\rangle
  + \tfrac{\sigma}{2}\| -Qw + A^*y + z - c \|^2.
```

We make the following assumption:

**Assumption 1** *There exists a vector $(y^*, w^*, z^*, x^*) \in \mathbb{R}^m \times \mathcal{W} \times \mathbb{R}^n \times \mathbb{R}^n$ satisfying the KKT system.*


```{math}
\begin{array}{|l|}
\hline
\textbf{Algorithm 1: A dual HPR method for solving the restricted-Wolfe dual problem} \\ \hline
\textbf{Input:}\ 
\text{Let } \mathcal{S}_y \text{ and } \mathcal{S}_w \text{ be two self-adjoint positive semidefinite linear operators on } \mathbb{R}^m \text{ and } \mathcal{W}, \\
\text{respectively, such that } \mathcal{S}_y + A A^* \text{ is positive definite. Denote } u = (y, w, z, x), \ \bar{u} = (\bar{y}, \bar{w}, \bar{z}, \bar{x}). \\
\text{Let } u^0 = (y^0, w^0, z^0, x^0) \in \mathcal{U}, \ \text{and set } \sigma > 0. \\ 
\textbf{for } k = 0,1,2,\ldots \ \textbf{do} \\ 
\quad \text{Step 1: } \ \bar{z}^{k+1} = \arg\min_{z \in \mathbb{R}^n} \{ L_\sigma(y^k, w^k, z; x^k) \}; \\[3pt]
\quad \text{Step 2: } \ \bar{x}^{k+1} = x^k + \sigma(-Qw^k + A^*y^k + \bar{z}^{k+1} - c); \\[3pt]
\quad \text{Step 3-1: } \ \bar{w}^{k+\frac{1}{2}} = \arg\min_{w \in \mathcal{W}} 
   \{ L_\sigma(y^k, w, \bar{z}^{k+1}; \bar{x}^{k+1}) + \tfrac{\sigma}{2}\| w - w^k \|_{\mathcal{S}_w}^2 \}; \\[3pt]
\quad \text{Step 3-2: } \ \bar{y}^{k+1} = \arg\min_{y \in \mathbb{R}^m}
   \{ L_\sigma(y, \bar{w}^{k+\frac{1}{2}}, \bar{z}^{k+1}; \bar{x}^{k+1}) + \tfrac{\sigma}{2}\| y - y^k \|_{\mathcal{S}_y}^2 \}; \\[3pt]
\quad \text{Step 3-3: } \ \bar{w}^{k+1} = \arg\min_{w \in \mathcal{W}}
   \{ L_\sigma(\bar{y}^{k+1}, w, \bar{z}^{k+1}; \bar{x}^{k+1}) + \tfrac{\sigma}{2}\| w - w^k \|_{\mathcal{S}_w}^2 \}; \\[3pt]
\quad \text{Step 4: } \ \hat{u}^{k+1} = 2\bar{u}^{k+1} - u^k; \\[3pt]
\quad \text{Step 5: } \ u^{k+1} = \tfrac{1}{k+2}u^0 + \tfrac{k+1}{k+2}\hat{u}^{k+1}; \\ 
\textbf{end for} \\ 
\textbf{Output:}\ \text{Iteration sequence } \{ \bar{u}^k \}. \\ \hline
\end{array}
```
**Remark** In Algorithm 1, the updates for $\bar{w}^{k+\frac{1}{2}}$ and $\bar{w}^{k+1}$ for $k \ge 0$ are restricted to the subspace $\mathcal{W} = \mathrm{Range}(Q)$. Although it may seem more straightforward to update $\bar{w}^{k+\frac{1}{2}}$ and $\bar{w}^{k+1}$ in the full space $\mathbb{R}^n$, doing so—particularly under a linearized ADMM framework—necessitates a proximal operator with a larger spectral norm, such as  
$\mathcal{S}_w = \lambda_1(Q^2 + Q/\sigma)I_n - (Q^2 + Q/\sigma)$, to ensure convergence.  
A proximal operator with a large spectral norm typically results in slower convergence.  
By contrast, restricting the update to $\mathcal{W}$ allows *HPR-QP* to employ a proximal operator with a smaller spectral norm, namely  
$\mathcal{S}_w = Q(\lambda_1(Q)I_n - Q)$, which accelerates convergence while preserving theoretical guarantees.


### An Easy-to-Implement Dual HPR Method

While computing $\bar{w}^{k+\frac{1}{2}}$ and $\bar{w}^{k+1}$ within $\mathrm{Range}(Q)$ may seem costly, these updates can be implemented efficiently **without explicit projection**. For small-scale or structured cases, one may simply set $\mathcal{S}_w = 0$ and use direct or preconditioned conjugate gradient solvers.  For large-scale general CCQP problems, HPR-QP adopts the proximal operator
```{math}
\mathcal{S}_w = Q(\lambda_Q I_n - Q),
```
where $\lambda_Q > 0$ satisfies $\lambda_Q \ge \lambda_1(Q)$. Then, for $k \ge 0$, the updates of $$ become
```{math}
\begin{aligned}
Q\bar{w}^{k+\frac{1}{2}} &= \frac{1}{1+\sigma\lambda_Q}
  Q\big(\sigma\lambda_Q w^k + \bar{x}^{k+1} + \sigma(-Qw^k + A^*y^k + \bar{z}^{k+1} - c)\big),\\[3pt], \bar{w}^{k+\frac{1}{2}}\in \mathcal{W}.
Q\bar{w}^{k+1} &= \frac{1}{1+\sigma\lambda_Q}
  Q\big(\sigma\lambda_Q w^k + \bar{x}^{k+1} + \sigma(-Qw^k + A^*\bar{y}^{k+1} + \bar{z}^{k+1} - c)\big), \bar{w}^{k+1}\in \mathcal{W}.
\end{aligned}
```
This design allows efficient updates via a **shadow sequence**, avoiding explicit projection onto $\mathrm{Range}(Q)$ and reducing computational overhead.

```{math}
\begin{array}{|l|}
\hline
\textbf{Algorithm 2: An easy-to-implement dual HPR method}\\ \textbf{for the restricted-Wolfe dual problem} \\ \hline
\textbf{Input:}\ 
\text{Let } \mathcal{S}_w = Q(\lambda_Q I_n - Q) \text{ with } \lambda_Q \ge \lambda_1(Q), \text{ and let } \mathcal{S}_y \succeq 0 \text{ such that } \mathcal{S}_y + A A^* \succ 0. \\ 
\text{Denote } u_Q = (y, w_Q, z, x),\ \bar{u}_Q = (\bar{y}, \bar{w}_Q, \bar{z}, \bar{x}).\ \text{Let } u_Q^0 = (y^0, w_Q^0, z^0, x^0) \text{ and set } \sigma > 0. \\ 
\textbf{for } k = 0,1,2,\ldots \ \textbf{do} \\ 
\quad \text{Step 1: } \ \bar{z}^{k+1} = \arg\min_{z \in \mathbb{R}^n} \{ L_\sigma(y^k, w_Q^k, z; x^k) \}; \\[3pt]
\quad \text{Step 2: } \ \bar{x}^{k+1} = x^k + \sigma(-Qw_Q^k + A^*y^k + \bar{z}^{k+1} - c); \\[3pt]
\quad \text{Step 3-1: } \ \bar{w}_Q^{k+\frac{1}{2}} = \tfrac{1}{1+\sigma\lambda_Q}
  (\sigma\lambda_Q w_Q^k + \bar{x}^{k+1} + \sigma(-Qw_Q^k + A^*y^k + \bar{z}^{k+1} - c)); \\[3pt]
\quad \text{Step 3-2: } \ \bar{y}^{k+1} = \arg\min_{y \in \mathbb{R}^m}
  \{ L_\sigma(y, \bar{w}_Q^{k+\frac{1}{2}}, \bar{z}^{k+1}; \bar{x}^{k+1}) + \tfrac{\sigma}{2}\|y - y^k\|_{\mathcal{S}_y}^2 \}; \\[3pt]
\quad \text{Step 3-3: } \ \bar{w}_Q^{k+1} = \tfrac{1}{1+\sigma\lambda_Q}
  (\sigma\lambda_Q w_Q^k + \bar{x}^{k+1} + \sigma(-Qw_Q^k + A^*\bar{y}^{k+1} + \bar{z}^{k+1} - c)); \\[3pt]
\quad \text{Step 4: } \ \hat{u}_Q^{k+1} = 2\bar{u}_Q^{k+1} - u_Q^k; \\[3pt]
\quad \text{Step 5: } \ u_Q^{k+1} = \tfrac{1}{k+2}u_Q^0 + \tfrac{k+1}{k+2}\hat{u}_Q^{k+1}; \\ 
\textbf{end for} \\ 
\textbf{Output:}\ \text{Iteration sequence } \{ \bar{u}_Q^k \}. \\ \hline
\end{array}
```


**Theorem 1** *Suppose Assumption 1 holds.  
Let $\{\bar{u}_Q^k\} = \{(y^k, w_Q^k, z^k, x^k)\}$ and $\{u_Q^k\} = \{(y^k, w_Q^k, z^k, x^k)\}$ be the sequences generated by Algorithm&nbsp;2,  
and let $u^* = (y^*, w^*, z^*, x^*)$ be a solution to the KKT system&nbsp;(2.1).  
Then, for all $k \ge 0$, the following complexity bounds hold with $R_0 = \|u_Q^0 - u^*\|_{\mathcal{M}}$:*

```{math}
\|\bar{u}_Q^{k+1} - u_Q^k\|_{\mathcal{M}} \le \frac{R_0}{k+1},
```

```{math}
\|\mathcal{R}(\bar{u}_Q^{k+1})\|
\le \left(\frac{\sigma\|A_Q^*\| + 1}{\sqrt{\sigma}} + \|\sqrt{\mathcal{S} + \hat{\mathcal{S}}_{\mathrm{sGS}}}\|\right)\frac{R_0}{k+1},
```

```{math}
-\frac{\|x^*\|}{\sqrt{\sigma}} \cdot \frac{R_0}{k+1}
\;\le\;
h(\bar{y}^{k+1}, \bar{w}_Q^{k+1}, \bar{z}^{k+1})
\;\le\;
\left(3R_0 + \frac{\|x^*\|}{\sqrt{\sigma}}\right)\frac{R_0}{k+1}.
```



```{math}
\begin{array}{|l|}
\hline
\textbf{Algorithm 3: HPR-QP — A dual HPR method for the CCQP problem} \ \hline
\textbf{Input:}\
\text{Let } \mathcal{S}_w \text{ be defined as in (2.8), and let } \mathcal{S}_y \text{ be a self-adjoint positive semidefinite linear operator on } \mathbb{R}^m\
\text{such that } \mathcal{S}_y + A A^* \text{ is positive definite. Let } u_Q = (y, w_Q, z, x),\ \bar{u}_Q = (\bar{y}, \bar{w}*Q, \bar{z}, \bar{x}),\
\text{and initial point } u_Q^{0,0} = (y^{0,0}, w_Q^{0,0}, z^{0,0}, x^{0,0}) \in \mathbb{R}^m \times \mathbb{R}^n \times \mathbb{R}^n \times \mathbb{R}^n.\
\textbf{Initialization:}\
\text{Set outer loop counter } r=0,\ \text{total iteration counter } k=0,\ \text{and initial penalty parameter } \sigma_0 > 0.\
\textbf{repeat} \
\quad \text{Initialize inner loop: set inner counter } t = 0; \
\quad \textbf{repeat} \
\quad\quad \bar{z}*Q^{r,t+1} = \arg\min*{z\in\mathbb{R}^n} L*{\sigma_r}(y^{r,t}, w_Q^{r,t}, z; x^{r,t}); \
\quad\quad \bar{x}^{r,t+1} = x^{r,t} + \sigma_r(-Q w_Q^{r,t} + A^* y^{r,t} + \bar{z}_Q^{r,t+1} - c); \
\quad\quad \bar{w}*Q^{r,t+\frac{1}{2}} = \frac{1}{1 + \sigma_r \lambda_Q} \Big( \sigma_r \lambda_Q w_Q^{r,t} + \bar{x}^{r,t+1} + \sigma_r(-Q w_Q^{r,t} + A^* y^{r,t} + \bar{z}*Q^{r,t+1} - c) \Big); \
\quad\quad \bar{y}^{r,t+1} = \arg\min*{y \in \mathbb{R}^m} \Big{ L*{\sigma_r}(y, \bar{w}_Q^{r,t+\frac{1}{2}}, \bar{z}*Q^{r,t+1}; \bar{x}^{r,t+1}) + \tfrac{\sigma_r}{2}|y - y^{r,t}|*{\mathcal{S}_y}^2 \Big}; \
\quad\quad \bar{w}_Q^{r,t+1} = \frac{1}{1 + \sigma_r \lambda_Q} \Big( \sigma_r \lambda_Q w_Q^{r,t} + \bar{x}^{r,t+1} + \sigma_r(-Q w_Q^{r,t} + A^* \bar{y}^{r,t+1} + \bar{z}_Q^{r,t+1} - c) \Big); \
\quad\quad \hat{u}_Q^{r,t+1} = 2\bar{u}_Q^{r,t+1} - u_Q^{r,t}; \
\quad\quad u_Q^{r,t+1} = \tfrac{1}{t+2} u_Q^{r,0} + \tfrac{t+1}{t+2} \hat{u}_Q^{r,t+1}; \
\quad\quad t = t + 1,\ k = k + 1; \
\quad \textbf{until restart or termination criteria are met;} \
\quad \textbf{Restart:}\
\text{Set } \tau_r = t,\ u_Q^{r+1,0} = \bar{u}*Q^{r,\tau_r}; \
\quad \sigma*{r+1} = \textbf{SigmaUpdate}(\bar{u}_Q^{r,\tau_r}, u_Q^{r,0}, \mathcal{S}_y, \mathcal{S}_w, A, Q); \
\quad r = r + 1; \
\textbf{until termination criteria are met;} \
\textbf{Output:}\ { \bar{u}_Q^{r,t} }. \ \hline
\end{array}
```



### Restart strategy

Restarting plays a central role in Halpern iterations. The complexity bound depends on the initial anchor distance $R_0$. As the iterates move closer to the solution, keeping a far-away anchor becomes inefficient. Resetting the anchor to the current iterate tightens the bound and improves late-stage progress.



This motivates the merit function

```{math}
R_{r,t} := \| w^{r,t} - w^* \|_{\mathcal{M}}, 
\quad \forall r \geq 0,\; t \geq 0,
```

where $w^*$ is any solution of the KKT system. Since $w^*$ is unknown, the practical surrogate

```{math}
\tilde{R}_{r,t} := \| w^{r,t} - \hat{w}^{r,t+1} \|_{\mathcal{M}}
```

is employed in defining restart rules. The following criteria are commonly adopted:

1. **Sufficient decay:**

```{math}
\tilde{R}_{r,t+1} \leq \alpha_1 \tilde{R}_{r,0},
```

2. **Necessary decay + no local progress:**

```{math}
\tilde{R}_{r,t+1} \leq \alpha_2 \tilde{R}_{r,0}, 
\quad \text{and} \quad
\tilde{R}_{r,t+1} > \tilde{R}_{r,t};
```

3. **Long inner loop:**

```{math}
t \geq \alpha_3 k;
```

where $0 < \alpha_1 < \alpha_2 < 1$ and $0 < \alpha_3 < 1$. When any criterion is met, the inner loop restarts with $w^{r+1,0}=\bar{w}^{r,\tau_r}$ and an updated $\sigma_{r+1}$.



### Update rules for $\sigma$

The penalty parameter $\sigma$ is updated at restart points to tighten the bound and reduce residuals. Ideally, $\sigma$ is chosen to minimize the weighted distance to the solution:



```{math}
\sigma_{r+1} := \arg\min_\sigma \| w^{r+1,0} - w^* \|_{\mathcal{M}}^2,
```

where $w^*$ is any solution of the KKT system.  
Substituting the definition of $\mathcal{M}$ leads to the closed-form expression

```{math}
\sigma_{r+1} =
\sqrt{
\frac{\| x^{r+1,0} - x^* \|^2}
{\| y^{r+1,0} - y^* \|_{\mathcal{T}_1}^2 + \| A^*(y^{r+1,0} - y^*) \|^2}
}.
```

Since $(x^*,y^*)$ are unknown, observable progress is used instead:



```{math}
\Delta_x := \| \bar{x}^{r,\tau_r} - x^{r,0} \|, 
\quad
\Delta_y := \sqrt{ \| \bar{y}^{r,\tau_r} - y^{r,0} \|_{\mathcal{T}_1}^2 + \| A^*(\bar{y}^{r,\tau_r} - y^{r,0}) \|^2 },
```

which yields the implementable update rule

```{math}
\sigma_{r+1} = \frac{\Delta_x}{\Delta_y}.
```

Several special cases of $\mathcal{T}_1$ are listed below:

1. **Case $\mathcal{T}_1 = 0$.**  
   This case occurs when $l_c = u_c = b$, which arises in applications with special structure in $A$. The $y$-update then reduces to solving the linear system

   ```{math}
   A A^* \bar{y}^{r,t+1} = \frac{1}{\sigma_r} \big( b - A(\bar{x}^{r,t+1} + \sigma_r(\bar{z}^{r,t+1} - c)) \big),
   ```

   which is computationally affordable in practice. In this case, the update rule simplifies to

   ```{math}
   \sigma_{r+1} = \frac{\| \bar{x}^{r,\tau_r} - x^{r,0} \|}{\| A^*(\bar{y}^{r,\tau_r} - y^{r,0}) \|}.
   ```

2. **Case $\mathcal{T}_1 = \lambda_A I_m - A A^*$ with $\lambda_A \geq \|A\|_2^2$.**  
   This choice applies when $l_c \neq u_c$ or when solving the system in case 1 is too expensive. The $y$-update takes the form

   ```{math}
   \bar{y}^{r,t+1} = \frac{1}{\sigma_r \lambda_A} \Big( \Pi_{\mathcal{K}}(R_y) - R_y \Big),
   ```

   where $R_y := A(2\bar{x}^{r,t+1} - x^{r,t}) - \sigma_r \lambda_A y^{r,t}$. In this setting, the update for $\sigma$ becomes

   ```{math}
   \sigma_{r+1} = \frac{\| \bar{x}^{r,\tau_r} - x^{r,0} \|}{\sqrt{\lambda_A} \; \| \bar{y}^{r,\tau_r} - y^{r,0} \|}.
   ```
 

Note that $\Delta_x$ and $\Delta_y$ may deviate significantly from the true quantities.


## GPU Implementation

We first present the update formulas for each subproblem in HPR-LP. Specifically, for any $r\ge 0$ and $t\ge 0$, the update of $z^{r,t+1}$ is:

```{math}
z^{r,t+1}
= \arg\min_{z\in\mathbb{R}^n}\{L_{\sigma_r}(y^{r,t}, z; x^{r,t})\}
= \frac{1}{\sigma_r}\!\left(\Pi_{\mathcal{C}}\!\big(x^{r,t}+\sigma_r(A^*y^{r,t}-c)\big)
 - \big(x^{r,t}+\sigma_r(A^*y^{r,t}-c)\big)\right).
```

The update of $x^{r,t+1}$ is:


```{math}
x^{r,t+1}
= x^{r,t} + \sigma_r\!\left(A^*y^{r,t} + z^{r,t+1} - c\right)
= \Pi_{\mathcal{C}}\!\big(x^{r,t}+\sigma_r(A^*y^{r,t}-c)\big).
```

For general LP problems, set $\mathcal{T}_1=\lambda I_m- AA^*$ with $\lambda \ge \lambda_1(AA^*)$ in HPR-LP. The update for $y^{r,t+1}$ is:

```{math}
y^{r, t+1} \;=\; \frac{1}{\sigma_r \lambda_A}\Big( \Pi_{\mathcal{K}}(R_y) - R_y \Big),
```

where $R_y := A\big(2x^{r, t+1} - x^{r,t}\big) - \sigma_r \lambda_A y^{r,t}.$ Combining these relations shows $z^{r,t+1}$ need not be computed at every step; it is only required when checking termination. Each step reduces to SpMV, vector operations, and simple projections, with per-iteration cost $O(\mathrm{nnz}(A))$.

On GPUs, these operations are mapped to custom CUDA kernels. Matrix–vector products use **`cusparseSpMV()`** with **`CUSPARSE_SPMV_CSR_ALG2`** for deterministic results.


