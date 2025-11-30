# Neural Oscillations Simulation: Kuramoto Model for Alpha-Band Rhythms in Visual Attention Networks

## Overview

This project implements a computational simulation of neural synchronization using the Kuramoto model, specifically targeting alpha-band oscillations (8-12 Hz) that play a crucial role in visual attention mechanisms. The simulation demonstrates how varying attentional demands modulate phase synchrony between neural populations, providing insights into the fundamental mechanisms of cortical communication and information processing.

## 🧠 Neuroscience Background

### Alpha-Band Oscillations in Visual Attention

Alpha-band oscillations (8-12 Hz) are among the most prominent rhythms in the human brain, particularly in the visual cortex. Their functional roles include:

- **Inhibitory Control**: Alpha oscillations implement "inhibitory pulsing" that gates information flow in cortical networks
- **Attention Modulation**: Increased alpha synchrony correlates with suppression of irrelevant visual information
- **Cortical Idling**: High alpha power during rest states reflects default network activity
- **Top-Down Control**: Frontal-parietal networks use alpha rhythms to regulate sensory processing

### The Kuramoto Model in Neuroscience

The Kuramoto model provides a mathematical framework for studying synchronization phenomena in coupled oscillator systems, making it ideal for modeling:

- Neural population dynamics
- Phase synchronization in cortical networks
- Transition from desynchronized to synchronized states
- Frequency locking under various coupling conditions

## 📐 Mathematical Foundations

### Kuramoto Model Equations

The dynamics of each neural oscillator follow the differential equation:
dθᵢ/dt = ωᵢ + (K/N) × Σⱼ sin(θⱼ - θᵢ)

Where:
- **θᵢ** = phase of oscillator *i* (radians)
- **ωᵢ** = natural frequency of oscillator *i* (radians/second)
- **K** = global coupling strength (modulated by attention)
- **N** = total number of oscillators
- **sin(θⱼ - θᵢ)** = coupling function that drives phase alignment

### Synchronization Metrics

#### Order Parameter (R)
The degree of global phase synchronization is quantified by:
R(t) = |(1/N) × Σⱼ exp(iθⱼ(t))|

- **R = 0**: Complete phase incoherence (perfect desynchronization)
- **R = 1**: Perfect phase locking (all oscillators move together)
- **0 < R < 1**: Partial synchronization with phase clusters

#### Phase Locking Value (PLV)
Pairwise phase consistency between oscillators *i* and *j*:
PLVᵢⱼ = |(1/T) × Σₜ exp[i(θᵢ(t) - θⱼ(t))]|

PLV measures stable phase relationships between oscillator pairs, with values ranging from 0 (no phase locking) to 1 (perfect phase locking).

### Frequency Analysis

Power spectral density is computed using the Fast Fourier Transform (FFT):
P(f) = |FFT{sin(θₐᵥₑ(t))}|²

Where θₐᵥₑ(t) is the average phase across the neural population.

## 🎯 Scientific Objectives

This simulation addresses several key questions in computational neuroscience:

1. **How does attentional load affect neural synchronization?**
2. **What are the dynamics of phase transitions in cortical networks?**
3. **How do alpha rhythms implement information gating in visual processing?**
4. **What coupling strengths produce optimal synchronization for information transfer?**

## 🚀 Features

### Core Simulation
- **Kuramoto Model Implementation**: Numerical integration of coupled oscillator dynamics using SciPy's ODE solver
- **Alpha-Band Focus**: Natural frequencies centered at 10 Hz with physiological variability
- **Attention Modulation**: Three coupling strengths simulating different attentional loads
- **Large-Scale Networks**: Configurable number of oscillators (default: 50)

### Analysis Capabilities
- **Order Parameter Tracking**: Real-time monitoring of global synchrony evolution
- **Phase Locking Matrices**: Pairwise synchronization analysis across the network
- **Power Spectral Analysis**: Frequency domain characterization of neural dynamics
- **Statistical Summaries**: Quantitative comparison of synchrony across conditions

### Visualization Suite
- **Phase Dynamics**: Temporal evolution of oscillator phases
- **Synchrony Evolution**: Order parameter trajectories across attention conditions
- **Coherence Matrices**: Heatmaps of functional connectivity
- **Spectral Profiles**: Frequency power distributions with alpha-band highlighting


## ⚙️ Installation & Setup

### 1. Clone and Setup Environment

```bash
# Create project directory
mkdir neural_oscillations
cd neural_oscillations

# Create virtual environment (recommended)
python3 -m venv venv

# Activate environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### What to Expect

The simulation will:

1. Initialize 50 neural oscillators with alpha-band frequencies
2. Simulate three attention conditions (low, medium, high load)
3. Compute synchrony metrics (order parameter and PLV)
4. Generate visualizations and save them as PNG files
5. Display interactive plots (close to continue)

### Output Files

- kuramoto_synchrony.png - Main results showing phase dynamics, synchrony evolution, and coherence matrices
- alpha_power_spectrum.png - Frequency analysis confirming alpha-band oscillations

## Customization

Edit parameters in kuramoto_model.py:

python
# In the simulate_attention_modulation() function
model = KuramotoModel(
    n_oscillators=50,           # Number of oscillators
    natural_freq_mean=10.0,     # Center frequency (Hz)
    natural_freq_std=1.5,       # Frequency spread
    coupling_strength=0.3       # Base coupling strength
)

# Attention conditions
conditions = {
    'Low Load (K=0.2)': 0.2,
    'Medium Load (K=0.5)': 0.5,
    'High Load (K=1.0)': 1.0
}



## References

- Kuramoto, Y. (1984). Chemical Oscillations, Waves, and Turbulence
- Buzsáki, G. (2006). Rhythms of the Brain
- Klimesch, W. (2012). Alpha-band oscillations and attention

## License

MIT License - Feel free to use and modify for research and educational purposes.

## Author


Neural Oscillations Research Project
