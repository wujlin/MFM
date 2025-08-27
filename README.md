# Mixed-Feedback Model for Emotional Bubble Dynamics

This repository contains the computational framework for the paper *"Feedback Competition Drives Emotional Bubbles in Information Ecosystems During Health Crises"*. We develop a Mixed-Feedback Model (MFM) to analyze the formation mechanisms of emotional bubbles during the COVID-19 pandemic.

The framework integrates mean-field theory, phase transition analysis, and agent-based network simulations to understand the nonlinear dynamics of collective emotional responses in heterogeneous information environments.

## Framework Components

### Core Implementation (`src/`)

- **`Meanfield_model.py`**: Mixed-feedback model implementation with self-consistent equation solver
- **`ParameterSpace_scan.py`**: Parameter space exploration for critical phase boundaries
- **`Simulation_scan.py`**: Agent-based network simulations for theoretical validation
- **`Config.py`**: Unified parameter configuration with multi-precision computing modes

### Phase Transition Analysis

- **`Poisson_correlation_analysis.py`**: Phase transition analysis for Poisson degree networks
- **`Powerlaw_correlation_analysis.py`**: Phase transition analysis for power-law degree networks
- **`analyze_scan_results_*.py`**: Systematic analysis and visualization of scanning results

### Figure Generation (`src/Figure*.py`)

- **`Figure1.py`**: Classification performance and mixed-feedback framework overview
- **`Figure2.py`**: Empirical evidence of emotional polarization dynamics
- **`Figure3.py`**: Individual psychological thresholds driving emotional bubble formation
- **`Figure4.py`**: Phase diagram analysis in psychological threshold space
- **`Figure5.py`**: Critical phenomena and state-space compression at phase transitions

### Data Analysis (`notebooks/`)

- **`Text_mining.ipynb`**: Sentiment analysis on 294,434 Weibo posts
- **`Empirical study.ipynb`**: Empirical study of Long-COVID discourse
- **`Network_analysis.ipynb`**: Multi-layer network topology analysis
- **`ABM.ipynb`**: Agent-based modeling implementation
- **`Prediction.ipynb`**: Machine learning for state transition prediction
- **`Counterfactual_analysis.ipynb`**: Counterfactual analysis validation

## Methodological Approach

### Mixed-Feedback Theory
The framework models dual feedback mechanisms in information ecosystems:
- **Mainstream media negative feedback**: $p_{risk}^{mainstream} = \frac{1-X_H+X_L}{2}$
- **We-media positive feedback**: $p_{risk}^{wemedia} = X_H$
- **Psychological threshold dynamics**: $\phi$ (low arousal threshold) and $\theta$ (high arousal threshold)

### Phase Transition Analysis
- Jacobian eigenvalue analysis for critical point identification
- Correlation length divergence: $\xi \sim |r-r_c|^{-\nu}$
- Power-law scaling behavior validation for second-order transitions

### Network Simulation Framework
- Three-layer architecture: mainstream media, we-media, and public individuals
- Threshold-driven state transition rules
- Microscopic dynamics validation of macroscopic theoretical predictions

## Requirements

- Python 3.8+
- Dependencies: NumPy, SciPy, NetworkX, pandas, scikit-learn, matplotlib, seaborn
- Dataset: 294,434 Weibo posts during COVID-19 (2020-2023)

## Quick Start

```bash
# Install dependencies
pip install numpy scipy networkx pandas scikit-learn matplotlib seaborn

# Run mixed-feedback model
python src/Meanfield_model.py

# Parameter space exploration
python src/ParameterSpace_scan.py

# Network simulation validation
python src/Simulation_scan.py

# Generate key figures
python src/Figure3.py  # Psychological threshold mechanisms
python src/Figure4.py  # Phase transition analysis
python src/Figure5.py  # Critical phenomena
```

## Data Availability

All code and data supporting the conclusions are publicly available. The computational framework includes data preprocessing scripts, machine learning models, theoretical calculations, and network simulation codes for full reproducibility.