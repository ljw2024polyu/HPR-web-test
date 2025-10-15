# Julia interfaces (HPR-QP)

This page shows how to use the **Julia interface** of HPR-QP: run demos (MPS / MAT), build custom convex quadratic programs (CQP), and solve QAP/LASSO instances.

---

## Usage 1: Test instances in **MPS** (and **MAT** for QAP/LASSO)

### Setting Data and Result Paths
Before running the scripts, modify `demo/run_single_file.jl` or `demo/run_dataset.jl` to set your **data path** and **result path**.

### Running a Single Instance
To test a single instance:
```bash
julia --project demo/run_single_file.jl
```

### Running All Instances in a Directory
To process all files in a directory:
```bash
julia --project demo/run_dataset.jl
```

### Notes
- **QAP instances (MAT):** The `.mat` file should contain matrices **A**, **B**, **S**, **T**.  
  See `HPR-QP_QAP_LASSO/demo/demo_QAP.jl` for how to generate such files. (Also refer to **Section 4.5** of the paper.)
- **LASSO instances (MAT):** The `.mat` file should contain matrix **A** and vector **b**.

---

## Usage 2: Define your **CQP** model in Julia scripts

### CQP model form
```{math}
\begin{aligned}
\min_{x \in \mathbb{R}^n}\quad & \tfrac{1}{2} x^\top Q x + c^\top x \\
\text{s.t.}\quad & A x \;\le\; b,\quad G x \;=\; h,\quad \ell \;\le\; x \;\le\; u,
\end{aligned}
```
where \(Q \succeq 0\).

### Example 1: Build & export a CQP with **JuMP**, then solve via HPR-QP
```bash
julia --project demo/demo_JuMP.jl
```
The script:
- Builds a CQP model in **JuMP**  
- Saves the model as an **MPS** file  
- Uses **HPR-QP** to solve it

> **Remark (sanity check):** If a model may be infeasible/unbounded, you can check it with **HiGHS** first.
```julia
using JuMP, HiGHS

# Read a model from file (or create it elsewhere)
mps_file_path = "xxx"  # your file path
model = read_from_file(mps_file_path)

# Set HiGHS as the optimizer (for feasibility/boundedness checks)
set_optimizer(model, HiGHS.Optimizer)

# Solve it
optimize!(model)
```

### Example 2: Define a small **CQP** directly in Julia (no JuMP)
```bash
julia --project demo/demo_QAbc.jl
```
This example constructs a toy CQP in pure Julia and calls HPR-QP directly.

### Example 3: Generate a random **LASSO** instance in Julia
```bash
julia --project demo/demo_LASSO.jl
```
LASSO objective (reference):
```{math}
\min_{x}\ \tfrac12\|Ax-b\|_2^2 + \lambda \|x\|_{\ell_1}.
```

---

## Note on first-time execution performance

The first run of a script (or the first file in a batch) may be slower due to Julia’s **JIT compilation**.

```{tip}
**Tip for better performance:** Run from **VS Code** or the **Julia REPL** to amortize compilation.
```

**Start a Julia REPL in the project:**
```bash
julia --project
```
Then run any demo inside REPL:
```julia
include("demo/demo_QAbc.jl")
```

```{admonition} CAUTION
If you see:

`Error: Error during loading of extension AtomixCUDAExt of Atomix, use Base.retry_load_extensions() to retry.`

This is usually transient. Wait a few moments; the extension typically loads successfully on its own.
```

---

## Common options (quick reference)

Typical options shared in demos/configs include:

- `time_limit`: maximum runtime in seconds (e.g., `3600`)
- `stoptol`: stopping tolerance (e.g., `1e-6`)
- `max_iter`: maximum iteration cap
- `check_iter`: residual check interval (e.g., `150`)
- `use_gpu`: whether to use GPU
- `device_number`: GPU device id when `use_gpu = true`
- `print_frequency`: log printing frequency (use `-1` for auto)

(See demo scripts’ headers for the exact list used by HPR-QP.)
