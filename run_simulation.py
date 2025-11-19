

from kuramoto_model import (
    simulate_attention_modulation,
    visualize_results,
    power_spectrum_analysis,
    analyze_synchrony_statistics,
)
import matplotlib.pyplot as plt
import os

def main():
    """Main execution function."""

    print("\n" + "=" * 70)
    print(" " * 15 + "NEURAL OSCILLATIONS SIMULATION")
    print(" " * 5 + "Kuramoto Model for Alpha-Band Rhythms in Visual Attention")
    print("=" * 70 + "\n")

    # Create outputs directory if it doesn't exist
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
        print("Created 'outputs' directory for saving figures.\n")

    # Run the simulation
    print("Step 1: Running simulation with three attention conditions...")
    print("  - Low attentional load (K=0.2)")
    print("  - Medium attentional load (K=0.5)")
    print("  - High attentional load (K=1.0)")
    print()

    model, results = simulate_attention_modulation()

    # Analyze results
    print("\nStep 2: Computing synchrony statistics...")
    analyze_synchrony_statistics(results)

    # Generate visualizations
    print("\nStep 3: Generating visualizations...")

    fig1 = visualize_results(model, results)
    print("  ✓ Created main synchrony visualization")

    fig2 = power_spectrum_analysis(model, results)
    print("  ✓ Created power spectrum analysis")

    # Save figures
    print("\nStep 4: Saving figures...")
    fig1.savefig('outputs/kuramoto_synchrony.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: outputs/kuramoto_synchrony.png")

    fig2.savefig('outputs/alpha_power_spectrum.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: outputs/alpha_power_spectrum.png")

    # Display plots
    print("\nStep 5: Displaying interactive plots...")
    print("  (Close the plot windows to exit)")
    print("\n" + "=" * 70)
    print("Simulation complete! Check the 'outputs' folder for saved figures.")
    print("=" * 70 + "\n")

    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        print("Please check that all dependencies are installed correctly.")
        print("Run: pip install -r requirements.txt")