# Releases
This page shows only the version tag and a one-line summary. Click any version to expand full notes and downloads.

---

<details id="v012">
<summary><code>v0.1.2</code> — SpMV & kernel rewrites; +11% Mittelmann, +7% MIP2017 (SGM10, 1e-8)</summary>

**Release date**: 2025-09-27

**Highlights**
- SpMV rewrites: added preprocessing and buffer preallocation to avoid redundant work between iterations.  
- CUDA kernel refactors: reduced memory traffic and improved occupancy.  
- Under SGM10 (1e-8 accuracy): +11% on Mittelmann’s LP benchmark set and +7% on MIP2017 large-scale LP relaxations (vs. v0.1.1).

**Downloads**
- [HPR-LP_v012_bug_fixed.zip](https://github.com/PolyU-IOR/HPR-LP/releases/download/v0.1.2/HPR-LP_v012_bug_fixed.zip)  
- [Source code (zip)](https://github.com/PolyU-IOR/HPR-LP/archive/refs/tags/v0.1.2.zip)  
- [Source code (tar.gz)](https://github.com/PolyU-IOR/HPR-LP/archive/refs/tags/v0.1.2.tar.gz)

</details>

---

<details id="v011">
<summary><code>v0.1.1</code> — Model reformulation + adaptive restart/penalty + fused kernels</summary>

**Release date**: 2025-09-09

**Highlights**
- Reformulated the problem model for better stability and consistency.  
- Added adaptive restart and automatic penalty update.  
- Fused several CUDA kernels to reduce memory traffic and improve performance.  
- Removed `sigma` and `sigma_fixed` for a cleaner interface.

**Downloads**
- [Source code (zip)](https://github.com/PolyU-IOR/HPR-LP/archive/refs/tags/v0.1.1.zip)  
- [Source code (tar.gz)](https://github.com/PolyU-IOR/HPR-LP/archive/refs/tags/v0.1.1.tar.gz)

</details>

---

<details id="v010">
<summary><code>v0.1.0</code> — First public release: Julia GPU-accelerated HPR-LP solver</summary>

**Release date**: 2025-07-04

**Highlights**
- First public release of **HPR-LP**.  
- GPU-accelerated LP solver in Julia implementing the Halpern–Peaceman–Rachford (HPR) method.  
- Supported model formulation:
```{math}
\begin{aligned}
\min_{x \in \mathbb{R}^n} \quad & (c, x) \\
\text{s.t.}\quad
& A_1 x = b_1, \\
& A_2 x \ge b_2, \\
& l \le x \le u.
\end{aligned}
```

**Downloads**
- [Source code (zip)](https://github.com/PolyU-IOR/HPR-LP/archive/refs/tags/v0.1.0.zip)  
- [Source code (tar.gz)](https://github.com/PolyU-IOR/HPR-LP/archive/refs/tags/v0.1.0.tar.gz)

</details>
