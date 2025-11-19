

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import seaborn as sns

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


class KuramotoModel:
    

    def __init__(self, n_oscillators=50, natural_freq_mean=10.0,
                 natural_freq_std=1.0, coupling_strength=0.5):
        
        self.n = n_oscillators
        self.omega = np.random.normal(natural_freq_mean * 2 * np.pi,
                                      natural_freq_std * 2 * np.pi, n_oscillators)
        self.K = coupling_strength

    def derivatives(self, theta, t, K):
        
        dtheta = np.zeros(self.n)
        for i in range(self.n):
            dtheta[i] = self.omega[i] + (K / self.n) * np.sum(
                np.sin(theta - theta[i])
            )
        return dtheta

    def simulate(self, duration=5.0, dt=0.001, K=None):
        
        if K is None:
            K = self.K

        t = np.arange(0, duration, dt)
        theta0 = np.random.uniform(0, 2 * np.pi, self.n)

        theta = odeint(self.derivatives, theta0, t, args=(K,))

        return t, theta.T

    def compute_order_parameter(self, theta):
        
        z = np.mean(np.exp(1j * theta), axis=0)
        return np.abs(z)

    def compute_phase_coherence(self, theta):
        
        plv = np.zeros((self.n, self.n))

        for i in range(self.n):
            for j in range(i + 1, self.n):
                phase_diff = theta[i, :] - theta[j, :]
                plv[i, j] = np.abs(np.mean(np.exp(1j * phase_diff)))
                plv[j, i] = plv[i, j]

        return plv


def simulate_attention_modulation(duration=5.0, dt=0.001):
    
    # Initialize model
    model = KuramotoModel(n_oscillators=50, natural_freq_mean=10.0,
                          natural_freq_std=1.5, coupling_strength=0.3)

    # Simulate three attention conditions
    conditions = {
        'Low Load (K=0.2)': 0.2,
        'Medium Load (K=0.5)': 0.5,
        'High Load (K=1.0)': 1.0
    }

    results = {}

    for label, K in conditions.items():
        print(f"Simulating: {label}")
        t, theta = model.simulate(duration=duration, dt=dt, K=K)
        r = model.compute_order_parameter(theta)

        # Make sure we have enough timepoints for the "last second" selection
        n_last = int(1.0 / dt)
        if theta.shape[1] < n_last:
            plv_theta = theta
        else:
            plv_theta = theta[:, -n_last:]

        plv = model.compute_phase_coherence(plv_theta)  # Last second or all

        results[label] = {
            't': t,
            'theta': theta,
            'order_param': r,
            'plv': plv
        }

    return model, results


def visualize_results(model, results):
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    conditions = list(results.keys())
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    # Row 1: Phase trajectories
    for idx, (label, color) in enumerate(zip(conditions, colors)):
        ax = fig.add_subplot(gs[0, idx])
        theta = results[label]['theta']
        t = results[label]['t']

        # Plot subset of oscillators
        n_plot = min(2000, theta.shape[1])
        for i in range(0, model.n, 5):
            ax.plot(t[:n_plot], np.sin(theta[i, :n_plot]), alpha=0.3,
                    linewidth=0.5, color=color)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('sin(θ)')
        ax.set_title(f'{label}\nPhase Dynamics')
        ax.set_xlim([0, min(2, t[-1])])

    # Row 2: Order parameter over time
    ax_order = fig.add_subplot(gs[1, :])
    for label, color in zip(conditions, colors):
        t = results[label]['t']
        r = results[label]['order_param']
        ax_order.plot(t, r, label=label, color=color, linewidth=2)

    ax_order.set_xlabel('Time (s)', fontsize=12)
    ax_order.set_ylabel('Order Parameter (R)', fontsize=12)
    ax_order.set_title('Phase Synchrony Across Attention Conditions',
                      fontsize=14, fontweight='bold')
    ax_order.legend(loc='lower right', fontsize=10)
    ax_order.set_ylim([0, 1])
    ax_order.grid(alpha=0.3)

    # Row 3: Phase coherence matrices
    for idx, (label, color) in enumerate(zip(conditions, colors)):
        ax = fig.add_subplot(gs[2, idx])
        plv = results[label]['plv']

        im = ax.imshow(plv, cmap='RdYlBu_r', vmin=0, vmax=1,
                      aspect='auto', interpolation='nearest')
        ax.set_xlabel('Oscillator')
        ax.set_ylabel('Oscillator')
        ax.set_title(f'{label}\nPhase Locking Value')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('PLV', rotation=270, labelpad=15)

    plt.suptitle('Neural Alpha-Band Oscillations: Kuramoto Model Simulation',
                 fontsize=16, fontweight='bold', y=0.995)

    return fig


def analyze_synchrony_statistics(results):
    
    print("\n" + "=" * 60)
    print("SYNCHRONY ANALYSIS")
    print("=" * 60)

    for label in results.keys():
        r = results[label]['order_param']
        plv = results[label]['plv']

        # Statistics: use last second if available
        t = results[label]['t']
        dt = t[1] - t[0] if len(t) > 1 else 0.001
        n_last = int(1.0 / dt) if dt > 0 else min(1000, len(r))
        r_mean = np.mean(r[-n_last:])  # Last second
        r_std = np.std(r[-n_last:])
        if plv.size > 1:
            plv_mean = np.mean(plv[np.triu_indices_from(plv, k=1)])
        else:
            plv_mean = float(plv)

        print(f"\n{label}:")
        print(f"  Mean Order Parameter: {r_mean:.3f} ± {r_std:.3f}")
        print(f"  Mean PLV: {plv_mean:.3f}")
        print(f"  Final Synchrony: {r[-1]:.3f}")


def power_spectrum_analysis(model, results):
   
    import matplotlib.pyplot as plt  # local import safe
    from scipy.fft import fft, fftfreq

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    for idx, (label, color) in enumerate(zip(results.keys(), colors)):
        theta = results[label]['theta']
        t = results[label]['t']
        dt = t[1] - t[0] if len(t) > 1 else 0.001

        # Convert phase to signal
        signal = np.mean(np.sin(theta), axis=0)

        # Compute power spectrum
        n = len(signal)
        fft_vals = fft(signal)
        freq = fftfreq(n, dt)

        # Positive frequencies only
        mask = (freq > 0) & (freq < 20)
        power = np.abs(fft_vals[mask]) ** 2

        axes[idx].plot(freq[mask], power, color=color, linewidth=2)
        axes[idx].axvspan(8, 12, alpha=0.2, color='gray', label='Alpha Band')
        axes[idx].set_xlabel('Frequency (Hz)')
        axes[idx].set_ylabel('Power')
        axes[idx].set_title(label)
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)

    plt.tight_layout()
    return fig


# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("NEURAL OSCILLATIONS SIMULATION")
    print("Kuramoto Model for Alpha-Band Rhythms in Visual Attention")
    print("=" * 60)

    # Run simulation
    model, results = simulate_attention_modulation(duration=5.0, dt=0.001)

    # Analyze results
    analyze_synchrony_statistics(results)

    # Create visualizations
    print("\nGenerating visualizations...")
    fig1 = visualize_results(model, results)
    fig2 = power_spectrum_analysis(model, results)

    # Save figures
    fig1.savefig('kuramoto_synchrony.png', dpi=300, bbox_inches='tight')
    fig2.savefig('alpha_power_spectrum.png', dpi=300, bbox_inches='tight')
    print("\nFigures saved:")
    print("  - kuramoto_synchrony.png")
    print("  - alpha_power_spectrum.png")

    plt.show()

    print("\nSimulation complete!")
