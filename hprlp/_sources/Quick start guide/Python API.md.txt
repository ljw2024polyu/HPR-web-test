# Python API

This page explains how to install the **Python version** of HPR-LP.

---


## 1. Prerequisites

- **Python** ≥ 3.12  
- **CUDA Toolkit** (≥ 12.4 recommended)  

Quick checks:
```bash
python --version
nvidia-smi
```

---


## 2. Clone the Repository

```bash

```

Or download and extract the ZIP file from GitHub.

---


## 3. Install Dependencies

Install PyTorch **first** (choose the right CUDA wheel for your system):
```bash
pip install torch torchvision
```

Then install the remaining packages:
```bash
pip install -r requirements.txt
```

## Next Steps

- See the [Interfaces](../Interfaces/Python interfaces.md) section for python usage.