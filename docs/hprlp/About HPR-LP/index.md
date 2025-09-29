
# About HPR-LP

```{admonition} TL;DR
:class: tip
**HPR-LP** is a **GPU-accelerated** linear programming solver based on the **Halpern Peaceman–Rachford** (HPR) method with **adaptive restart**—scalable, deterministic, and robust on large sparse LPs.
```

```{div} badges
:class: hpr-badges
- 🚀 **GPU Ready** (cuSPARSE SpMV)
- 📦 **Julia / Python interfaces**
- ♻️ **Adaptive restart & σ-tuning**
- 📊 **Deterministic SpMV** (CSR_ALG2)
```

```{toctree}
:hidden:
:maxdepth: 1

theory/index
```

```{div}
:class: cta-row
[Get Started](../get-started/index){.btn .btn-primary}
[GitHub · PolyU-IOR/HPR-LP](https://github.com/PolyU-IOR/HPR-LP){.btn .btn-secondary}
```

---

## What is HPR-LP?

**HPR-LP** implements a preconditioned, accelerated ADMM-like scheme derived from **Halpern-regularized Peaceman–Rachford** splitting. In practice: *few primitives per iteration (SpMV, axpy, projections), easy GPU parallelization, and steady progress via restart*.

```{admonition} Problem form (LP)
For sparse \(A\in\mathbb{R}^{m\times n}\),
\[
\min_{x\in\mathbb{R}^n}\ \langle c,x\rangle
\quad\text{s.t.}\quad
Ax\in\mathcal{K},\ \ x\in\mathcal{C},
\]
where \(\mathcal{K}=\{s:l_c\le s\le u_c\}\) and \(\mathcal{C}=\{x:l_v\le x\le u_v\}\).
```

---

## Key features

- **GPU-accelerated**: core ops are SpMV + projections → natural parallelism.
- **Halpern-PR with restart**: complexity-guided \(O(1/k)\) scheme with anchor resets.
- **Deterministic kernels**: `cusparseSpMV(..., CUSPARSE_SPMV_CSR_ALG2)` for stable results.
- **Box/interval constraints first-class**: simple, fast projections for \(\mathcal{C},\mathcal{K}\).
- **Scales with sparsity**: per-iteration cost \(O(\mathrm{nnz}(A))\).

---

## Quick start

```{tab-set}
```{tab-item} Julia
```bash
julia --project -e 'import Pkg; Pkg.instantiate()'
julia --project demo/demo_JuMP.jl
```
*Minimal JuMP example*:
```julia
# Build a small LP, call HPR-LP via JuMP, and print summary.
```
```

```{tab-item} Python
```bash
pip install hprlp  # if published
python examples/quickstart.py
```
*Minimal example*:
```python
# Build CSR A and bounds (lc, uc), (lv, uv), then solver.solve(...)
```
```
```

> See **Interfaces** and **Examples** for more options and parameters.

---

## How it works (intuition)

Each iteration does three things:  
1) **Projection/prox** for box constraints; 2) **SpMV** for coupling; 3) **Halpern step** with **restart** to move the anchor and avoid late-stage drag.  
The penalty \(\sigma\) is updated at restarts using an observable progress proxy to tighten subsequent residual bounds.

```{dropdown} Show theory (KKT, operator, complexity) ▾
:open: false
- KKT system, maximal-monotone operator \(\mathcal{T}\), Algorithm 1/2 under \(\mathcal{M}\)-metric.  
- Global convergence and \(O(1/k)\) rates (KKT residual and objective error).  
- Restart criteria and practical \(\sigma\) update rules.  
**→** Full details in {doc}`theory/index`.
```

---

## Performance at a glance

- **Throughput**: iteration time dominated by SpMV; ideal for very large sparse LPs.  
- **Stability**: Halpern + restart reduces late-stage stagnation.  
- **Determinism**: fixed cuSPARSE algorithm → reproducible runs.

> Full benchmarks (datasets, tolerances, hardware) are in **Benchmarks**.

---

## FAQ

**Q1. Why restart?**  
A distant anchor slows late stages; restart resets the anchor to the current iterate and tightens the bound \(R_0\).

**Q2. How is \(\sigma\) updated?**  
A closed-form target is approximated with observable progress \((\Delta_x,\Delta_y)\) between restarts.

**Q3. What constraints are supported?**  
Box/interval sets are first-class; other sets can be added via projections or prox operators.

---

## Cite us

If you use HPR-LP in your research, please cite:

*Chen, K., Sun, D., Yuan, Y., Zhang, G., Zhao, X. “HPR-LP: An implementation of a Halpern Peaceman–Rachford method for linear programming.” arXiv:2408.12179 (2025).*

