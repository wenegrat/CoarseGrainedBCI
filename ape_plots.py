"""
Plotting functions for Available Potential Energy (APE) analysis

This module contains visualization functions for energy timeseries data.
"""

import matplotlib.pyplot as plt


#+++ Plot energy timeseries
def plot_energy_timeseries(ds, APE, TPE, RPE, KE=None):
    """Plot APE and energy components over time"""
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Plot 1: TPE and RPE
    ax1 = axes[0]
    ax1.plot(ds.time, TPE, label='Total PE', linewidth=2)
    ax1.plot(ds.time, RPE, label='Reference PE', linewidth=2)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Potential Energy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Potential Energy Components')

    # Plot 2: APE and KE
    ax2 = axes[1]
    ax2.plot(ds.time, APE, label='APE', linewidth=2, color='red')
    if KE is not None:
        ax2_twin = ax2.twinx()
        ax2_twin.plot(ds.time, KE, label='KE', linewidth=2, color='blue', alpha=0.7)
        ax2_twin.set_ylabel('Kinetic Energy', color='blue')
        ax2_twin.tick_params(axis='y', labelcolor='blue')
        ax2_twin.legend(loc='upper right')

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Available Potential Energy', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Available Potential Energy and Kinetic Energy')

    # Plot 3: Normalized energy budget
    ax3 = axes[2]
    if KE is not None:
        total_energy = APE + KE
        total_energy_norm = total_energy / total_energy[0]

        ax3.plot(ds.time, APE / APE[0], label='APE (normalized)', linewidth=2, color='red')
        ax3.plot(ds.time, KE / KE[0], label='KE (normalized)', linewidth=2, color='blue')
        ax3.plot(ds.time, total_energy_norm, label='Total (APE + KE, normalized)',
                linewidth=2, color='black', linestyle='--')

        ax3.set_xlabel('Time')
        ax3.set_ylabel('Normalized Energy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Energy Budget: APE to KE Conversion')

    plt.tight_layout()
    return fig
#---
