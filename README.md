# Adaptive Orbital Direct-to-Device: Rethinking Satellite IoT Architectures


[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-PDF-red)](paper.pdf)
[![Contact](https://img.shields.io/badge/contact-hana.pasandi%40gmail.com-brightgreen)](mailto:hana.pasandi@gmail.com)

## Overview
This repository contains the simulation code and analysis tools for the paper "Adaptive Orbital Direct-to-Device: Rethinking Satellite IoT Architectures" by Pasandi et al. Our work presents an adaptive orbital D2D architecture that can switch among LEO, GEO, and hybrid paths based on real-time Doppler, network load, and application QoS.

## Key Results
- 3x capacity improvement: 936 vs 312 nodes per gateway at 95% PDR
- 62 percent tail latency reduction: 95th percentile 342 ms vs 892 ms
- 31 percent energy savings through retransmission avoidance
- 87 percent of LEO passes exceed Doppler tolerance based on TLE analysis

## Reproducibility Notice
For reproducibility, we ran all experiments inside a Python virtual environment (venv). Using venv isolates dependencies and makes sure you install the same package versions that we used. Please follow the steps below before running any scripts.

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### 1. Clone the repository
```bash
git clone https://github.com/hanabhp/adaptive_orbital_d2d.git
cd adaptive_orbital_d2d
```

### 2. Create and activate a virtual environment
macOS or Linux
```bash
python3 -m venv venv
source venv/bin/activate
# your shell prompt should now start with (venv)
```

Windows Command Prompt
```bat
python -m venv venv
venv\Scripts\activate
:: your prompt should now start with (venv)
```

Windows PowerShell
```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.env\Scripts\Activate.ps1
# your prompt should now start with (venv)
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify your environment
```bash
python -V
python -c "import numpy, matplotlib; print('Dependencies installed successfully')"
```

### 5. Optional: record exact versions
```bash
pip freeze > requirements-lock.txt
```

## Running the Simulations

### Option A: Quick reproduction
macOS or Linux
```bash
chmod +x run_all.sh
./run_all.sh
```

Windows
```bash
python main_simulation.py
python generate_figures.py
python analyze_results.py
```

This workflow will run simulations for all architectures, find capacity limits at 95 percent PDR, generate all figures, create summary reports in the results directory, and write figures to the figures directory. Expected runtime is about 10 to 15 minutes on a modern laptop or desktop.

### Option B: Run individual components
Main simulation only
```bash
python main_simulation.py
```

Generate figures only
```bash
python generate_figures.py
```

Analyze results only
```bash
python analyze_results.py
```

Verify paper claims
```bash
python verify_paper_claims.py
```

## Expected Output
- results/simulation_results.csv
- results/capacity_results.json
- results/evaluation_summary.json
- figures/figure_1_static_limitations.pdf
- figures/figure_2_adaptive_performance.pdf

## Repository Structure
```text
adaptive_orbital_d2d/
├── main_simulation.py        # Core simulation engine
├── generate_figures.py       # Figure generation
├── analyze_results.py        # Statistical analysis
├── verify_paper_claims.py    # Reproducibility verification
├── requirements.txt          # Python dependencies
├── run_all.sh                # One click evaluation script
├── configs/                  # Configuration files
│   └── default_config.json   # Default parameters
├── results/                  # Generated results (auto created)
├── figures/                  # Generated figures (auto created)
└── docs/                     # Additional documentation
```

## Random Seed and Determinism
All simulations use fixed seeds to improve reproducibility.
```python
import numpy as np, random
np.random.seed(42)
random.seed(42)
```

## Troubleshooting

ModuleNotFoundError
- Make sure the virtual environment is activated. You should see (venv) at the start of your prompt.
- macOS or Linux: run `source venv/bin/activate`
- Windows Command Prompt: run `venv\Scripts\activate`
- Windows PowerShell: run `.env\Scripts\Activate.ps1`

Inconsistent results compared to the paper
- Confirm Python version with `python --version` and make sure it is 3.8 or higher.
- Check that dependencies are installed with `pip list`.
- Pull the latest code with `git pull`.

Out of memory
- At least 4 GB RAM is recommended.
- Run components individually instead of the full run_all.sh.
- Reduce dataset size or figure resolution if needed.

Permission denied when running run_all.sh
- On macOS or Linux, run `chmod +x run_all.sh` and try again.

## Citation
```bibtex
@inproceedings{pasandi2025adaptive,
  title={Adaptive Orbital Direct-to-Device: Rethinking Satellite IoT Architectures},
  author={Pasandi, Hannaneh B. and Others},
  booktitle={Under Review},
  year={2025}
}
```

## License
MIT License. See the LICENSE file for details.

## Contact
- Issues and questions: https://github.com/hanabhp/adaptive_orbital_d2d/issues
- Email: h.pasandi@berkeley.edu
