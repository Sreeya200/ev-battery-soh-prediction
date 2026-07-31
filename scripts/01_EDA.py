# =============================================================================
# 01_EDA.py — Exploratory Data Analysis
# =============================================================================
# Project : Predicting EV Battery State of Health Using Machine Learning
# Author  : Sreeya Kannamala (23094968) | sk24aee@herts.ac.uk
# Module  : 7PAM2002 — Data Science Project
# Dataset : NASA PCoE Li-Ion Battery Aging Dataset
# =============================================================================
#
# What this script does:
#   1. Loads the NASA battery dataset (.mat files)
#   2. Extracts discharge capacity from each cycle
#   3. Computes State of Health (SoH) = current capacity / initial capacity
#   4. Produces 4 EDA plots:
#      - Capacity degradation over cycles (all 4 batteries)
#      - SoH decline over cycles
#      - Distribution of capacity measurements
#      - Total capacity loss per battery
#
# How to run:
#   python scripts/01_EDA.py
#
# Output:
#   outputs/EDA_Battery_SoH.png
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io
import os

# =============================================================================
# SETTINGS
# =============================================================================

# Where the .mat files are stored (see data/README.md to download)
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# Where to save the output plot
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Battery cell names — must match .mat filenames
BATTERY_NAMES = ['B0005', 'B0006', 'B0007', 'B0018']

# Colours for each battery in plots
COLOURS = {
    'B0005': '#2196F3',
    'B0006': '#E91E63',
    'B0007': '#4CAF50',
    'B0018': '#FF9800',
}

# End-of-life threshold: battery is retired when capacity drops to 1.4 Ah
# This is a 30% fade from the rated 2 Ah capacity
EOL_THRESHOLD = 1.4


# =============================================================================
# FUNCTION: LOAD CAPACITY FROM .MAT FILE
# =============================================================================

def load_capacity(battery_name):
    """
    Opens a NASA .mat file and extracts discharge capacity per cycle.

    The NASA dataset uses nested MATLAB structures. Each 'cycle' has a
    'type' (charge / discharge / impedance) and 'data' with measurements.
    We only want 'discharge' cycles, and we read the final capacity value
    recorded in each — this represents how much charge the battery delivered.

    Args:
        battery_name (str): e.g. 'B0005'

    Returns:
        capacities (list of float): one value per discharge cycle (in Ah)
    """
    mat_path = os.path.join(DATA_DIR, f'{battery_name}.mat')
    mat = scipy.io.loadmat(mat_path, simplify_cells=True)
    cycles = mat[battery_name]['cycle']

    capacities = []
    for cycle in cycles:
        if cycle['type'] == 'discharge':
            cap = cycle['data']['Capacity']
            # Take the last reading in the cycle
            capacities.append(float(cap) if np.isscalar(cap) else float(cap[-1]))

    return capacities


# =============================================================================
# FUNCTION: COMPUTE STATE OF HEALTH
# =============================================================================

def compute_soh(capacities):
    """
    Computes SoH as a fraction of the initial (first cycle) capacity.

    SoH = 1.0 means the battery is at 100% of its original capacity.
    SoH = 0.7 means only 70% remains — this is end of life (30% fade).

    Args:
        capacities (list of float): discharge capacity per cycle

    Returns:
        soh (list of float): SoH values between 0 and 1
    """
    initial = capacities[0]
    return [c / initial for c in capacities]


# =============================================================================
# LOAD ALL 4 BATTERIES
# =============================================================================

print("Loading NASA battery data...")

all_data = {}
for name in BATTERY_NAMES:
    try:
        caps = load_capacity(name)
        all_data[name] = {
            'capacities': caps,
            'soh':    compute_soh(caps),
            'cycles': list(range(1, len(caps) + 1))
        }
        print(f"  {name}: {len(caps)} discharge cycles")
    except FileNotFoundError:
        print(f"  ERROR: {name}.mat not found — please see data/README.md")

if not all_data:
    raise SystemExit("No data loaded. Download the dataset first (see data/README.md).")


# =============================================================================
# GENERATE EDA PLOTS
# =============================================================================

print("\nGenerating EDA plots...")

fig = plt.figure(figsize=(14, 10), facecolor='white')
fig.suptitle(
    'Exploratory Data Analysis — NASA PCoE Li-Ion Battery Aging Dataset\n'
    'Sreeya Kannamala (23094968) | 7PAM2002 Data Science Project',
    fontsize=13, fontweight='bold', y=0.99
)
gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

# ── Plot 1: Capacity Degradation ──────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
for name, d in all_data.items():
    ax1.plot(d['cycles'], d['capacities'], color=COLOURS[name],
             label=name, linewidth=1.5, alpha=0.85)
ax1.axhline(EOL_THRESHOLD, color='red', linestyle='--',
            linewidth=1.2, label=f'EOL ({EOL_THRESHOLD} Ah)')
ax1.set_xlabel('Cycle Number', fontsize=10)
ax1.set_ylabel('Capacity (Ah)', fontsize=10)
ax1.set_title('Battery Capacity Degradation Over Cycles',
              fontsize=11, fontweight='bold')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_facecolor('#f9f9f9')

# ── Plot 2: State of Health ────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
for name, d in all_data.items():
    soh_pct = [s * 100 for s in d['soh']]
    ax2.plot(d['cycles'], soh_pct, color=COLOURS[name],
             label=name, linewidth=1.5, alpha=0.85)
ax2.axhline(70, color='red', linestyle='--', linewidth=1.2,
            label='EOL threshold (70%)')
ax2.fill_between(range(0, 200), 0, 70, alpha=0.05, color='red')
ax2.set_xlabel('Cycle Number', fontsize=10)
ax2.set_ylabel('State of Health (%)', fontsize=10)
ax2.set_title('State of Health (SoH) Decline Over Cycles',
              fontsize=11, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_facecolor('#f9f9f9')
ax2.set_ylim(60, 105)

# ── Plot 3: Capacity Distribution ─────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
for name, d in all_data.items():
    ax3.hist(d['capacities'], bins=20, alpha=0.55,
             color=COLOURS[name], label=name, edgecolor='white')
ax3.axvline(EOL_THRESHOLD, color='red', linestyle='--',
            linewidth=1.2, label='EOL threshold')
ax3.set_xlabel('Capacity (Ah)', fontsize=10)
ax3.set_ylabel('Frequency', fontsize=10)
ax3.set_title('Distribution of Capacity Measurements\nAcross All Batteries',
              fontsize=11, fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_facecolor('#f9f9f9')

# ── Plot 4: Capacity Loss Summary ─────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
names  = list(all_data.keys())
inits  = [all_data[n]['capacities'][0]  for n in names]
finals = [all_data[n]['capacities'][-1] for n in names]
drops  = [i - f for i, f in zip(inits, finals)]
colors = [COLOURS[n] for n in names]

bars = ax4.bar(names, drops, color=colors, edgecolor='white', width=0.5)
for bar, drop, init in zip(bars, drops, inits):
    ax4.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.003,
             f'{(drop/init)*100:.1f}%',
             ha='center', va='bottom', fontsize=9, fontweight='bold')
ax4.set_xlabel('Battery ID', fontsize=10)
ax4.set_ylabel('Total Capacity Drop (Ah)', fontsize=10)
ax4.set_title('Total Capacity Loss Per Battery\n(% fade shown above bars)',
              fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_facecolor('#f9f9f9')

# Footer
fig.text(0.5, 0.01,
    'Data: NASA PCoE Battery Aging Dataset | '
    'https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip',
    ha='center', fontsize=8, color='gray')

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
out = os.path.join(OUTPUT_DIR, 'EDA_Battery_SoH.png')
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
print(f"Plot saved: {out}")
plt.show()
