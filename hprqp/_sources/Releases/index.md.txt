# Releases

This page lists all public releases of **HPR-LP**.  


---


## v0.1.2  (2025-09-27)

**Highlights**
- SpMV rewrites. Add a preprocessing step and buffer preallocation to avoid redundant work between iterations.  
- Kernel rewrites. Several CUDA kernels were refactored to reduce memory traffic and improve occupancy.
- In terms of SGM10 (1e-8 accuracy), 11% faster for Mittelmann's LP benchmark set and 7% faster for MIP2017 large-scale LP relaxations (compared to v0.1.1).  


**Downloads**
- [HPR-LP_v012_bug_fixed.zip](https://github.com/PolyU-IOR/HPR-LP/releases/download/v0.1.2/HPR-LP_v012_bug_fixed.zip)
- [Source code (zip)](https://github.com/PolyU-IOR/HPR-LP/archive/refs/tags/v0.1.2.zip)  
- [Source code (tar.gz)](https://github.com/PolyU-IOR/HPR-LP/archive/refs/tags/v0.1.2.tar.gz)

## v0.1.1  (2025-09-09)

**Highlights**
- Reformulated the problem model for better stability and consistency.  
- Added adaptive restart and automatic penalty update.  
- Rewritten and fused several CUDA kernels to reduce memory traffic and improve performance.  
- Removed `sigma` and `sigma_fixed` parameters for a cleaner interface.

**Downloads**
- [Source code (zip)](https://github.com/PolyU-IOR/HPR-LP/archive/refs/tags/v0.1.1.zip)  
- [Source code (tar.gz)](https://github.com/PolyU-IOR/HPR-LP/archive/refs/tags/v0.1.1.tar.gz)

---

## v0.1.0  (2025-07-04)

**Highlights**
- First public release of **HPR-LP**.  
- GPU-accelerated LP solver in Julia implementing the Halpern–Peaceman–Rachford (HPR) method.  
- Supported model formulation:
  ```{math}
  \begin{aligned}
  \min_{x \in \mathbb{R}^n} \quad & (c, x) \\
  \text{s.t.} \quad 
  & A_1 x = b_1, \\
  & A_2 x \ge b_2, \\
  & l \le x \le u.
  \end{aligned}
  ```

**Downloads**
- [Source code (zip)](https://github.com/PolyU-IOR/HPR-LP/archive/refs/tags/v0.1.0.zip)  
- [Source code (tar.gz)](https://github.com/PolyU-IOR/HPR-LP/archive/refs/tags/v0.1.0.tar.gz)

---

