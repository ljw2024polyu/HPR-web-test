# About HPR-LP

HPR-LP is a GPU-accelerated solver for linear programming based on the Halpern Peaceman–Rachford method with adaptive restart
<!-- ```{toctree}
:maxdepth: 1
:caption: About HPR-LP

Problem statement
HPR method for LP
Algorithmic enhancements
Implementations
``` -->

## Problem statement



We consider the following general form of LP:

$$
\begin{aligned}
\min_{x \in \mathbb{R}^n} & \;\; \langle c, x \rangle \\
\text{s.t.} \;\; & Ax \in \mathcal{K}, \\
& x \in \mathcal{C},
\end{aligned}

$$

where $c \in \mathbb{R}^n$ is the objective vector, $A \in \mathbb{R}^{m \times n}$ is the constraint matrix,  
$\mathcal{K} := \{ s \in \mathbb{R}^m : l_c \leq s \leq u_c \}$ with bounds $l_c \in (\mathbb{R} \cup \{-\infty\})^m$ and $u_c \in (\mathbb{R} \cup \{\infty\})^m$,  
and $\mathcal{C} := \{ x \in \mathbb{R}^n : l_v \leq x \leq u_v \}$ with bounds $l_v \in (\mathbb{R} \cup \{-\infty\})^n$ and $u_v \in (\mathbb{R} \cup \{\infty\})^n$.  

The corresponding dual problem is:

$$
\begin{aligned}
\min_{y \in \mathbb{R}^m, \; z \in \mathbb{R}^n} & \;\; \delta_{\mathcal{K}}^*(-y) + \delta_{\mathcal{C}}^*(-z) \\
\text{s.t.} \;\; & A^* y + z = c,
\end{aligned}

$$

where $\delta_S^*(\cdot)$ denotes the convex conjugate of the indicator function $\delta_S(\cdot)$ associated with a closed convex set $S$.



## HPR method for LP

We introduce a Halpern-Peaceman-Rachford (HPR) method for solving linear programming problems. We first describe the base algorithm and then discuss its convergence guarantees and complexity properties, which motivate subsequent algorithmic enhancements.

### Base algorithm

For any $(y, z, x) \in \mathbb{R}^m \times \mathbb{R}^n \times \mathbb{R}^n$, the augmented Lagrangian of the dual problem is

```{math}
L_\sigma(y, z; x) := \delta_{\mathcal{K}}^*(-y) + \delta_{\mathcal{C}}^*(-z)
+ \langle x, A^* y + z - c \rangle
+ \frac{\sigma}{2} \| A^* y + z - c \|^2,
```

where $\sigma > 0$ is a penalty parameter. For notational convenience, let $w := (y, z, x) \in \mathbb{W} := \mathbb{R}^m \times \mathbb{R}^n \times \mathbb{R}^n$. Then, an HPR method with semi-proximal terms for solving the above problems is summarized in Algorithm 1.

```{math}
\begin{array}{|l|}
\hline
\textbf{Algorithm 1: An HPR method with semi-proximal terms} \\ \hline
\textbf{Input:}\
\text{Set the penalty parameter }\sigma>0.\ \\
\text{Let }\mathcal{T}_1:\mathbb{R}^m\to\mathbb{R}^m\ \text{be a self-adjoint positive semidefinite linear operator such that } \\
\mathcal{T}_1+\sigma A A^*\ \text{is positive definite. } 
\text{Denote } w=(y,z,x)\ \text{and }\bar{w}=(\bar{y},\bar{z},\bar{x}).\ \\
\text{Choose an initial point } w^0=(y^0,z^0,x^0)\in\mathbb{R}^m\times\mathbb{R}^n\times\mathbb{R}^n. \\ 
\textbf{for } k = 0,1,\ldots \ \textbf{do} \\ 
\quad \text{Step 1: } \ \bar{z}^{k+1} = \arg\min_{z \in \mathbb{R}^n} L_\sigma(y^k, z; x^k); \\ 
\quad \text{Step 2: } \ \bar{x}^{k+1} = x^k + \sigma(A^* y^k + \bar{z}^{k+1} - c); \\ 
\quad \text{Step 3: } \ \bar{y}^{k+1} = \arg\min_{y \in \mathbb{R}^m} 
   \left\{ L_\sigma(y, \bar{z}^{k+1}; \bar{x}^{k+1}) 
   + \tfrac{\sigma}{2}\|y-y^k\|_{\mathcal{T}_1}^2 \right\}; \\ 
\quad \text{Step 4: } \ \hat{w}^{k+1} = 2\bar{w}^{k+1} - w^k; \\ 
\quad \text{Step 5: } \ w^{k+1} = \tfrac{1}{k+2} w^0 + \tfrac{k+1}{k+2} \hat{w}^{k+1}; \\ 
\textbf{end for} \\ 
\textbf{Output:} \text{Iteration sequence } \{ \bar{w}^k \}. \\ \hline
\end{array}
```





**Remark** Steps 1–3 correspond to the Douglas–Rachford (DR) method. Adding Step 4 (relaxation) yields the Peaceman–Rachford (PR) method, and Step 5 introduces Halpern iteration with step size $1/(k+2)$. Together, Algorithm 1 is an accelerated preconditioned ADMM (pADMM) with parameter $\alpha = 2$.

A pair $(y^*, z^*) \in \mathbb{R}^m \times \mathbb{R}^n$ is an optimal solution to the corresponding dual problem if there exists $x^* \in \mathbb{R}^n$ such that $(y^*, z^*, x^*)$ satisfies the following KKT system:

```{math}
\begin{aligned}
0 &\in A x^* - \partial \delta_{\mathcal{K}}^*(-y^*), \\
0 &\in x^* - \partial \delta_{\mathcal{C}}^*(-z^*), \\
&\quad A^* y^* + z^* - c = 0.
\end{aligned}
```

We make the following assumption:

**Assumption 1** *There exists a vector $(y^*, z^*, x^*) \in \mathbb{R}^m \times \mathbb{R}^n \times \mathbb{R}^n$ satisfying the KKT system above.*

Under Assumption 1, solving the primal–dual pair is equivalent to finding a point $w^* \in \mathbb{R}^m \times \mathbb{R}^n \times \mathbb{R}^n$ such that $0 \in \mathcal{T} w^*$, where the maximal monotone operator $\mathcal{T}$ is defined as

```{math}
\mathcal{T} w =
\begin{pmatrix}
- \partial \delta_{\mathcal{K}}^*(-y) + A x \\
- \partial \delta_{\mathcal{C}}^*(-z) + x \\
c - A^* y - z
\end{pmatrix},
\quad
\forall w = (y, z, x) \in \mathbb{R}^m \times \mathbb{R}^n \times \mathbb{R}^n.
```

The global convergence of Algorithm 1 is established in the following proposition.

**Proposition 2**. *Suppose that Assumption 1 holds.  
Then the sequence $\{\bar{w}^k\} = \{(\bar{y}^k, \bar{z}^k, \bar{x}^k)\}$ generated by the HPR method with semi-proximal terms in Algorithm 1 converges to a point $w^* = (y^*, z^*, x^*)$, where $(y^*, z^*)$ solves the dual problem and $x^*$ solves the primal problem.*

Next, consider the self-adjoint positive semidefinite linear operator $\mathcal{M} : \mathbb{R}^m \times \mathbb{R}^n \times \mathbb{R}^n \to \mathbb{R}^m \times \mathbb{R}^n \times \mathbb{R}^n$ defined by

```{math}
\mathcal{M} =
\begin{bmatrix}
\sigma A A^* + \sigma \mathcal{T}_1 & 0 & A \\
0 & 0 & 0 \\
A^* & 0 & \tfrac{1}{\sigma} I_n
\end{bmatrix},
```

where $I_n$ denotes the $n \times n$ identity matrix. To analyze the complexity of the HPR method with semi-proximal terms, we consider the KKT residual and the objective error. The residual mapping associated with the KKT system is given by

```{math}
\mathcal{R}(w) =
\begin{pmatrix}
A x - \Pi_{\mathcal{K}}(A x - y) \\
x - \Pi_{\mathcal{C}}(x - z) \\
c - A^* y - z
\end{pmatrix},
\quad
\forall w = (y, z, x) \in \mathbb{R}^m \times \mathbb{R}^n \times \mathbb{R}^n.
```

Furthermore, let $\{(\bar{y}^k, \bar{z}^k)\}$ be the sequence generated by Algorithm 1. We define

the objective error as

```{math}
h(\bar{y}^{k+1}, \bar{z}^{k+1})
:= \delta_{\mathcal{K}}^*(-\bar{y}^{k+1}) + \delta_{\mathcal{C}}^*(-\bar{z}^{k+1})
- \delta_{\mathcal{K}}^*(-y^*) - \delta_{\mathcal{C}}^*(-z^*),
\quad \forall k \geq 0,
```

where $(y^*, z^*)$ is the limit point of the sequence $\{(\bar{y}^k, \bar{z}^k)\}$.  
The complexity of the HPR method with semi-proximal terms is summarized in the following theorem.

**Theorem 3**  *Suppose that Assumption 1 holds. Let $\{w^k\} = \{(y^k, z^k, x^k)\}$ and $\{\bar{w}^k\} = \{(\bar{y}^k, \bar{z}^k, \bar{x}^k)\}$ be two sequences generated by the HPR method with semi-proximal terms in Algorithm 1, and let $w^* = (y^*, z^*, x^*)$ be its limit point. Define $R_0 = \|w^0 - w^*\|_{\mathcal{M}}$. Then for all $k \geq 0$, the following iteration complexity bounds hold:*

```{math}
\| \bar{w}^k - \hat{w}^{k+1} \|_{\mathcal{M}} \leq \frac{R_0}{k+1},
```

```{math}
\| \mathcal{R}(\bar{w}^{k+1}) \|
\leq \left( \frac{\sigma(\|A\| + \| \sqrt{\mathcal{T}_1} \|) + 1}{\sqrt{\sigma}} \right) \frac{R_0}{k+1},
```

```{math}
- \frac{1}{\sqrt{\sigma}} \| x^* \| \frac{R_0}{k+1}
\;\; \leq \;\;
h(\bar{y}^{k+1}, \bar{z}^{k+1})
\;\; \leq \;\;
\left( 3 R_0 + \frac{1}{\sqrt{\sigma}} \| x^* \| \right) \frac{R_0}{k+1}.
```


<!-- The results above establish that the HPR method enjoys global convergence and an $O(1/k)$ complexity rate in terms of both KKT residuals and objective error. These properties motivate the use of restart strategies and adaptive parameter updates, which will be reviewed in the next subsection. -->

## Algorithmic enhancements

Several enhancements have been proposed to improve the performance of the HPR method for solving LP and CCQP.  
In particular, restart strategies and adaptive updates of the penalty parameter $\sigma$, motivated by the $O(1/k)$ complexity results in Theorem 3, have proven effective.  
For completeness, we summarize the HPR-LP framework in Algorithm 2.

```{math}
\begin{array}{|l|}
\hline
\textbf{Algorithm 2  HPR-LP: A Halpern Peaceman-Rachford method} \\ \textbf{for the problem} \\ \hline
\textbf{Input:} \ \mathcal{T}_1:\mathbb{R}^m\to\mathbb{R}^m\ \text{be a self-adjoint positive semidefinite linear operator such that}\\ \mathcal{T}_1+\sigma AA^* \text{ is positive definite}. \text{ Denote }w=(y,z,x),\ \bar{w}=(\bar{y},\bar{z},\bar{x}).\ \text{Choose an initial point}\\ w^{0,0}=(y^{0,0},z^{0,0},x^{0,0})\in\mathbb{R}^m\times\mathbb{R}^n\times\mathbb{R}^n. \\ 
\textbf{Initialization:}\ \text{Set the outer loop counter }r=0,\ \text{the total loop counter }k=0,\ \\ \text{and the initial penalty parameter }\sigma_0>0. \\ 
\textbf{repeat} \\ 
\quad \text{initialize the inner loop: set inner loop counter }t=0; \\ 
\quad \textbf{repeat} \\ 
\quad\quad \bar{z}^{r,t+1}=\arg\min_{z\in\mathbb{R}^n}\{L_{\sigma_r}(y^{r,t},z;x^{r,t})\}; \\ 
\quad\quad \bar{x}^{r,t+1}=x^{r,t}+\sigma_r(A^*y^{r,t}+\bar{z}^{r,t+1}-c); \\ 
\quad\quad \bar{y}^{r,t+1}=\arg\min_{y\in\mathbb{R}^m}\{L_{\sigma_r}(y,\bar{z}^{r,t+1};\bar{x}^{r,t+1})+\tfrac{\sigma_r}{2}\|y-y^{r,t}\|_{\mathcal{T}_1}^2\}; \\ 
\quad\quad \hat{w}^{r,t+1}=2\bar{w}^{r,t+1}-w^{r,t}; \\ 
\quad\quad w^{r,t+1}=\tfrac{1}{t+2}w^{r,0}+\tfrac{t+1}{t+2}\hat{w}^{r,t+1}; \\ 
\quad\quad t=t+1,\ k=k+1; \\ 
\quad \textbf{until}\ \text{one of the restart criteria holds or termination criteria hold} \\ 
\quad \textbf{restart the inner loop: }\tau_r=t,\ w^{r+1,0}=w^{r,\tau_r}, \\ 
\quad \sigma_{r+1}=\textbf{SigmaUpdate}(\bar{w}^{r,\tau_r},w^{r,0},\mathcal{T}_1,A),\ r=r+1; \\ 
\textbf{until}\ \text{termination criteria hold} \\ 
\textbf{Output:}\ \{\bar{w}^{r,t}\}. \\ \hline
\end{array}
```





### Restart strategy

Restarting has been recognized as particularly important for Halpern iterations.  
As noted in Theorem 3, the complexity bound depends on the weighted distance $R_0$ between the initial point and the optimal solution. Consequently, as the iterates approach optimality, continuing to reference a distant initial anchor becomes counterproductive, whereas resetting the anchor to the current iterate helps reduce the bound and refocus the iteration near the solution.  

This observation motivates the merit function

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

where $0 < \alpha_1 < \alpha_2 < 1$ and $0 < \alpha_3 < 1$.  
When any criterion is met, the inner loop is restarted at iteration $(r+1)$ with $w^{r+1,0} = \bar{w}^{r,\tau_r}$ and an updated $\sigma_{r+1}$.


### Update rules for $\sigma$

Another important enhancement of HPR methods concerns the update of the penalty parameter  $\sigma$.  
The update strategy is motivated by the complexity results of the HPR method in Algorithm 1.  At a high level, the goal is to select $\sigma$ at each restart to tighten the complexity bound and thereby reduce the KKT residuals in subsequent iterations. Specifically, the ideal update is defined as the minimizer of the weighted distance to the optimal solution:

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

Since the optimal solution $(x^*, y^*)$ is unknown, practical implementations approximate these terms using the observed progress within each outer loop:

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
   This choice applies when $l_c \neq u_c$ or when solving the equation in case 1 directly is expensive. The $y$-update takes the form

   ```{math}
   \bar{y}^{r,t+1} = \frac{1}{\sigma_r \lambda_A} \Big( \Pi_{\mathcal{K}}(R_y) - R_y \Big),
   ```

   where $R_y := A(2\bar{x}^{r,t+1} - x^{r,t}) - \sigma_r \lambda_A y^{r,t}$. In this setting, the update for $\sigma$ becomes

   ```{math}
   \sigma_{r+1} = \frac{\| \bar{x}^{r,\tau_r} - x^{r,0} \|}{\sqrt{\lambda_A} \; \| \bar{y}^{r,\tau_r} - y^{r,0} \|}.
   ```
 

It is worth noting that the approximations $\Delta_x$ and $\Delta_y$ may deviate significantly from the true quantities.  


## GPU Implementation

We first present the update formulas for each subproblem in HPR-LP. Specifically, for any $r\ge 0$ and $t\ge 0$, the update of $z^{r,t+1}$ is:

```{math}
z^{r,t+1}
= \arg\min_{z\in\mathbb{R}^n}\{L_{\sigma_r}(y^{r,t}, z; x^{r,t})\}
= \frac{1}{\sigma_r}\!\left(\Pi_{\mathcal{C}}\!\big(x^{r,t}+\sigma_r(A^*y^{r,t}-c)\big)
 - \big(x^{r,t}+\sigma_r(A^*y^{r,t}-c)\big)\right).
```

Next, the update of $x^{r,t+1}$ is:

```{math}
x^{r,t+1}
= x^{r,t} + \sigma_r\!\left(A^*y^{r,t} + z^{r,t+1} - c\right)
= \Pi_{\mathcal{C}}\!\big(x^{r,t}+\sigma_r(A^*y^{r,t}-c)\big).
```

For general LP problems, set $\mathcal{T}_1=\lambda I_m- AA^*$ with $\lambda \ge \lambda_1(AA^*)$ in HPR-LP. The update for $y^{r,t+1}$ is:

```{math}
y^{r, t+1} \;=\; \frac{1}{\sigma_r \lambda_A}\Big( \Pi_{\mathcal{K}}(R_y) - R_y \Big),
```

where $R_y := A\big(2x^{r, t+1} - x^{r,t}\big) - \sigma_r \lambda_A y^{r,t}.$ By combining these relations, we observe that it is not necessary to compute $z^{r,t+1}$ at every iteration. Instead, $z^{r,t+1}$ is only evaluated when checking the termination criteria. Each update mainly involves matrix–vector multiplications, vector additions, and projections. The per-iteration complexity is $O(\mathrm{nnz}(A))$, where $\mathrm{nnz}(A)$ is the number of nonzeros in $A$.

To fully utilize GPUs, we implement custom CUDA kernels for the above updates. For matrix–vector multiplications, we use **`cusparseSpMV()`** from cuSPARSE, which employs the **`CUSPARSE_SPMV_CSR_ALG2`** algorithm to ensure deterministic results.
