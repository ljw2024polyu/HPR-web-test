# C interfaces

This page shows the **C interface** of HPR-LP with a minimal, task-oriented guide. Use the CLI to solve `.mps` files, or link your C/C++ program against `libhprlp` for programmatic access.

---

## Usage

### Solving from MPS Files
Use the prebuilt executable in `build/`. The default settings enable GPU, standard scalings, and a 3600-second time limit.

```bash
# Show help
./build/solve_mps_file -h

# Run with default settings
./build/solve_mps_file -i data/model.mps

# Run with custom settings
./build/solve_mps_file -i data/model.mps --tol 1e-4 --time-limit 3600
```

### Using as a Library
Link against `libhprlp.a` (static) or `libhprlp.so` (shared) and include the public headers under `include/`.  
See the `examples/` directory for small, self-contained demos (C and C++) showing compilation flags and basic API calls.

For more details, check `examples/c(cpp)/README.md` on integrating HPR-LP-C into external projects (include paths, link order, and CUDA libs).

---

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --input <path>` | Path to input MPS file | *required* |
| `--device <id>` | CUDA device ID | 0 |
| `--max-iter <N>` | Maximum iterations | unlimited |
| `--tol <eps>` | Stopping tolerance | 1e-4 |
| `--time-limit <sec>` | Time limit in seconds | 3600 |
| `--check-iter <N>` | Convergence check interval | 150 |
| `--ruiz <true/false>` | Ruiz scaling | true |
| `--pock <true/false>` | Pock-Chambolle scaling | true |
| `--bc <true/false>` | Bounds/cost scaling | true |
| `-h, --help` | Show help message | - |

> Tip: If the runtime can’t find `libhprlp.so`, set `LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH` (or install to a system path).

---

## Language Interface Installation

**Python (pip only):**
```bash
cd HPR-LP-C/bindings/python
python -m pip install .         # or: python -m pip install -e .
```

**Julia Interface:**
```bash
cd bindings/julia
bash install.sh
```

**MATLAB Interface:**
```bash
cd bindings/matlab
bash install.sh
```

See the respective README files in `bindings/` for usage, version notes, and troubleshooting.

---

## Language Interfaces
HPR-LP-C provides native interfaces for multiple languages:

- **Python**: pybind11 bindings (see `bindings/python/README.md`)  
- **MATLAB**: MEX interface with OOP wrapper (see `bindings/matlab/README.md`)  
- **Julia**: Native Julia wrapper (see `bindings/julia/README.md`)  
- **C/C++**: Direct API usage (see `examples/c/` and `examples/cpp/` for minimal build/run demos)
