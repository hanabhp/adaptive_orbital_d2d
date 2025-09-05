# Adaptive Orbital Direct-to-Device: Rethinking Satellite IoT Architectures

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-PDF-red)](paper.pdf)
[![Contact](https://img.shields.io/badge/contact-hana.pasandi%40gmail.com-brightgreen)](mailto:hana.pasandi@gmail.com)




# Adaptive Orbital Direct-to-Device: Rethinking Satellite IoT Architectures

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview

This repository contains the simulation code and analysis tools for the paper **"Adaptive Orbital Direct-to-Device: Rethinking Satellite IoT Architectures"** by Pasandi et al.

Our work presents, to our knowledge, the first adaptive orbital D2D architecture that dynamically switches among LEO, GEO, and hybrid paths based on real-time Doppler, load, and application QoS. The key contributions include:

- **3× capacity improvement** (936 vs 312 nodes per gateway at 95% PDR)
- **62% tail latency reduction** (95th percentile: 342ms vs 892ms)  
- **31% energy savings** via retransmission avoidance
- **87% of LEO passes exceed Doppler tolerance** (validated through TLE analysis)

## Key Results from Paper

### Table 1: Architecture Comparison
| Architecture | Latency | Reliability | Energy | Adaptive | Capacity |
|--------------|---------|-------------|---------|----------|----------|
| Static LEO (Sparse) | Excellent | Poor | Good | No | 312 nodes |
| Static LEO (Dense) | Good | Poor | Moderate | No | 156 nodes |
| Static GEO | Poor | Excellent | Poor | No | 498 nodes |
| **Adaptive (This Work)** | **Excellent** | **Excellent** | **Excellent** | **Yes** | **936 nodes** |

### Key Findings
- LEO satellites at 340-2000km altitude offer 20-50ms latency but suffer Doppler shifts up to ±21.7kHz
- GEO satellites at 35,786km provide stable 250-300ms latency but require 15-20dB higher link budget
- 73% of transmission opportunities could benefit from alternative orbital selection
- Commercial LoRa gateways limited to 12 concurrent channels (Semtech SX1302)

## Repository Structure

```
adaptive-orbital-d2d/
│
├── main_simulation.py       # Core simulation engine
├── generate_figures.py      # Figure generation for paper
├── analyze_results.py       # Results analysis and validation
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── .gitignore             # Git ignore rules
│
├── configs/               # Simulation configuration files
│   └── default_config.json
│
├── results/              # Simulation results (generated)
│   ├── simulation_results.csv
│   ├── capacity_results.json
│   └── evaluation_summary.json
│
├── figures/              # Generated figures (generated)
│   ├── figure_1_static_limitations.pdf
│   └── figure_2_adaptive_performance.pdf
│
├── docs/                # Additional documentation
│   ├── REPORT.md       # Detailed technical report
│   └── ALGORITHMS.md   # Algorithm descriptions
│
└── tests/              # Unit tests
    └── test_simulation.py
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/adaptive-orbital-d2d.git
cd adaptive-orbital-d2d
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Quick Start

### Optional: local virtual environment (for reviewers)

If you prefer a fully isolated run, you can use a Python virtual environment.  
**Note:** The one-click scripts (`sim-arch.sh` or `run_all.sh`) already handle env setup automatically. This section is only for **manual** runs.

**macOS / Linux**
```bash
# from the repo root
python3 -m venv .venv
source .venv/bin/activate          # prompt should show: (.venv)
python -m pip install -U pip
pip install -r requirements.txt

# sanity: confirm you're using the venv's Python (not system/conda)
python -c "import sys; print(sys.executable)"


### Running the Complete Evaluation

To reproduce all paper results:

```bash
python main_simulation.py
```

This will:
- Run simulations for all architectures (LEO-Sparse, LEO-Dense, GEO, Adaptive)
- Find capacity limits at 95% PDR
- Generate detailed performance metrics
- Save results to `results/` directory

Expected runtime: ~10-15 minutes on a modern CPU

### Generating Paper Figures

```bash
python generate_figures.py
```

This creates:
- Figure 1: Static orbital architecture limitations
- Figure 2: Adaptive orbital selection performance
- Supplementary analysis figures

### Analyzing Results

```bash
python analyze_results.py
```

This provides:
- Statistical validation of results
- Confidence intervals
- Performance comparisons
- LaTeX-formatted tables for the paper

## Detailed Usage

### Custom Simulation Scenarios

```python
from main_simulation import AdaptiveOrbitalSimulator, SimulationConfig

# Custom configuration
config = SimulationConfig(
    simulation_duration=7200,  # 2 hours
    num_nodes_list=[100, 200, 300, 400, 500],
    gateway_channels=12  # Hardware limitation
)

# Run simulation
simulator = AdaptiveOrbitalSimulator(config)
results = simulator.run_scenario(
    num_nodes=300,
    deployment_type="dense",
    architecture="Adaptive"
)

print(f"PDR: {results['pdr']:.2%}")
print(f"95th percentile latency: {results['p95_latency_ms']:.1f} ms")
```

### Modifying Parameters

Edit `configs/default_config.json` to adjust:
- Network topology
- LoRa parameters (SF, BW, TX power)
- Orbital characteristics
- QoS thresholds

### Running Specific Experiments

```bash
# Test scalability only
python main_simulation.py --experiment scalability

# Test energy efficiency
python main_simulation.py --experiment energy

# Test latency distribution
python main_simulation.py --experiment latency
```

## Key Algorithms

### Adaptive Orbital Selection (Algorithm 1)

The core algorithm evaluates:
1. **Doppler feasibility** for LEO satellites
2. **Gateway congestion** levels
3. **QoS requirements** per application class
4. **Fallback to GEO** for best-effort delivery

```python
def adaptive_orbital_selection(node, packet):
    # Check Doppler tolerance for LEO
    if abs(doppler_shift) > bandwidth / (2^(SF+2)):
        mark_LEO_infeasible()
    
    # Evaluate gateway queue delay
    if queue_delay > max_delay - prop_delay:
        exclude_congested_gateways()
    
    # Select minimum-delay feasible path
    return min(feasible_options, key=lambda x: x.total_delay)
```

## Validation

Our simulation is validated against:
- **847 real packets** from FossaSat-2 CubeSat
- **96.8% Doppler prediction accuracy**
- **TLE ephemeris data** from Space-Track.org

## Performance Metrics

The simulator tracks:
- **Packet Delivery Ratio (PDR)**
- **End-to-end latency** (median, 95th percentile)
- **Energy consumption** per successfully delivered message
- **Gateway utilization** and congestion levels
- **Orbital selection distribution**

## Reproducibility

All random seeds are fixed for reproducibility:
```python
np.random.seed(42)
random.seed(42)
```

To verify results match the paper:
```bash
python verify_paper_claims.py
```

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{pasandi2025adaptive,
  title={Adaptive Orbital Direct-to-Device: Rethinking Satellite IoT Architectures},
  author={Pasandi, Hannaneh B. and Hosseini, Mohammad and Dorabi, Sina and Fraire, Juan A. and Rousseau, Franck},
  booktitle={pre-print},
  year={2025}
}
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Contact

- **Hannaneh B. Pasandi** - h.pasandi@berkeley.edu
- **Mohammad Hosseini** - m.hosseini@sbu.ac.ir
- **Sina Dorabi** - sina.dorabi@usi.ch
- **Juan A. Fraire** - juan.fraire@inria.fr
- **Franck Rousseau** - franck.rousseau@imag.fr

## Acknowledgments

We thank the anonymous reviewers for their valuable feedback and the FossaSat team for providing real satellite data for validation.

---

**Contact:** hana.pasandi@gmail.com