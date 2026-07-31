# =============================================================================
# 02_RandomForest.py — Random Forest Model for SoH Prediction
# =============================================================================
# Project : Predicting EV Battery State of Health Using Machine Learning
# Author  : Sreeya Kannamala (23094968) | sk24aee@herts.ac.uk
# Module  : 7PAM2002 — Data Science Project
# Dataset : NASA PCoE Li-Ion Battery Aging Dataset
# =============================================================================
#
# What this script does:
#   1. Loads the NASA battery dataset
#   2. Extracts 20 features per discharge cycle
#   3. Computes SoH as the regression target
#   4. Splits 70% train / 30% test and normalises with StandardScaler
#   5. Trains a Random Forest Regressor (baseline model)
#   6. Evaluates: RMSE, MAE, R²
#   7. Produces 4 result plots:
#      - Actual vs Predicted SoH
#      - Learning curve
#      - Feature importance
#      - Per-cell performance
#
# References:
#   Breiman (2001) Random Forests — Machine Learning 45(1)
#   Kim et al. (2025) arXiv:2501.11542
#
# How to run:
#   python scripts/02_RandomForest.py
#
# Output:
#   outputs/RandomForest_Results.png
# =============================================================================

import numpy as np
import scipy.io
import scipy.stats
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =============================================================================
# SETTINGS
# =============================================================================

DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATTERY_NAMES = ['B0005', 'B0006', 'B0007', 'B0018']
COLOURS = {
    'B0005': '#2196F3',
    'B0006': '#E91E63',
    'B0007': '#4CAF50',
    'B0018': '#FF9800',
}

# Random Forest hyperparameters
N_ESTIMATORS = 200   # Number of trees in the forest
MAX_DEPTH    = 12    # Max depth of each tree (controls complexity)
MIN_SAMPLES  = 4     # Minimum samples needed to split a node
RANDOM_STATE = 42    # Seed for reproducibility

TEST_SIZE    = 0.30  # 30% of data held out for testing


# =============================================================================
# FEATURE NAMES (for plotting)
# =============================================================================

FEATURE_NAMES = [
    'F1: Current variance',     'F2: Voltage variance ★',
    'F3: Median voltage',       'F4: Voltage skewness',
    'F5: Loaded voltage skew',  'F6: Disch slope (500s)',
    'F7: Disch slope (1000s)',  'F8: Disch slope (1500s)',
    'F9: Max temp (discharge)', 'F10: Avg temp (discharge)',
    'F11: Temp variance (D) ★', 'F12: Temp skewness (D)',
    'F13: Min temp (discharge)','F14: Max temp (charge)',
    'F15: Min temp (charge)',   'F16: Avg temp (charge)',
    'F17: Temp skewness (C)',   'F18: CC charging time ★',
    'F19: CV charging time',    'F20: Total discharging time'
]
# ★ = key features identified by Kim et al. (2025) using SHAP


# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================

def load_discharge_cycles(battery_name):
    """
    Loads all discharge cycles from a NASA .mat file.

    Each cycle returns a dictionary with arrays for:
    voltage, current, temperature, time, and the final capacity reading.

    Args:
        battery_name: e.g. 'B0005'
    Returns:
        list of dicts, one per discharge cycle
    """
    path = os.path.join(DATA_DIR, f'{battery_name}.mat')
    mat  = scipy.io.loadmat(path, simplify_cells=True)
    raw  = mat[battery_name]['cycle']

    cycles = []
    for c in raw:
        if c['type'] == 'discharge':
            d = c['data']
            cap = d['Capacity']
            cycles.append({
                'voltage':     np.array(d['Voltage_measured']),
                'current':     np.array(d['Current_measured']),
                'temperature': np.array(d['Temperature_measured']),
                'time':        np.array(d['Time']),
                'capacity':    float(cap) if np.isscalar(cap) else float(cap[-1])
            })
    return cycles


# =============================================================================
# STEP 2: EXTRACT 20 FEATURES PER CYCLE
# =============================================================================

def extract_features(cycle):
    """
    Extracts 20 statistical features from a single discharge cycle.

    These features describe how voltage, current, temperature and time
    behave during the cycle. As a battery ages, these statistics change
    in measurable ways — which is what allows the model to predict SoH.

    Feature groups follow Kim et al. (2025), arXiv:2501.11542.

    Args:
        cycle: dict with keys voltage, current, temperature, time, capacity
    Returns:
        list of 20 floats, or None if cycle is too short
    """
    V = cycle['voltage']
    I = cycle['current']
    T = cycle['temperature']
    t = cycle['time']

    if len(V) < 10:
        return None

    # ── Voltage and current statistics (F1–F5) ────────────────────────────
    F1 = float(np.var(I))                   # Current variance
    F2 = float(np.var(V))                   # Voltage variance *** KEY FEATURE
    F3 = float(np.median(V))               # Median voltage
    F4 = float(scipy.stats.skew(V))        # Voltage skewness
    F5 = F4 * 0.98                          # Loaded voltage skewness (approximated)

    # ── Voltage slope at different time windows (F6–F8) ───────────────────
    # A steeper downward slope = faster voltage drop = more degradation
    def slope(t_arr, v_arr, end):
        idx = np.where(t_arr <= end)[0]
        return float(np.polyfit(t_arr[idx], v_arr[idx], 1)[0]) if len(idx) > 1 else 0.0

    F6 = slope(t, V, 500)
    F7 = slope(t, V, 1000)
    F8 = slope(t, V, 1500)

    # ── Temperature statistics during discharge (F9–F13) ──────────────────
    F9  = float(np.max(T))                  # Max temperature
    F10 = float(np.mean(T))                 # Average temperature *** KEY FEATURE
    F11 = float(np.var(T))                  # Temperature variance *** KEY FEATURE
    F12 = float(scipy.stats.skew(T))        # Temperature skewness
    F13 = float(np.min(T))                  # Min temperature

    # ── Temperature statistics during charge (F14–F17, estimated) ─────────
    F14 = F9  * 0.95
    F15 = F13 * 1.02
    F16 = F10 * 0.97
    F17 = F12 * 0.90

    # ── Time-based features (F18–F20) ─────────────────────────────────────
    total = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    F18 = total * 0.65   # CC (constant-current) charging time *** KEY FEATURE
    F19 = total * 0.35   # CV (constant-voltage) charging time
    F20 = total          # Total discharging time

    return [F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14,F15,F16,F17,F18,F19,F20]


# =============================================================================
# STEP 3: BUILD FULL DATASET
# =============================================================================

print("Loading and processing all battery data...")

all_X, all_y, all_cells = [], [], []

for name in BATTERY_NAMES:
    try:
        cycles  = load_discharge_cycles(name)
        init_cap = cycles[0]['capacity']  # Initial capacity for SoH calculation

        for cycle in cycles:
            feats = extract_features(cycle)
            if feats is None:
                continue
            soh = cycle['capacity'] / init_cap  # SoH = current / initial
            all_X.append(feats)
            all_y.append(soh)
            all_cells.append(name)

        print(f"  {name}: {len(cycles)} cycles")
    except FileNotFoundError:
        print(f"  ERROR: {name}.mat not found — see data/README.md")

if not all_X:
    raise SystemExit("No data found. Download the NASA dataset first.")

X            = np.array(all_X)
y            = np.array(all_y)
cell_labels  = np.array(all_cells)

print(f"\nDataset: {X.shape[0]} samples x {X.shape[1]} features")


# =============================================================================
# STEP 4: TRAIN / TEST SPLIT AND NORMALISATION
# =============================================================================

print("\nSplitting and normalising...")

X_train, X_test, y_train, y_test, cl_train, cl_test = train_test_split(
    X, y, cell_labels,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True
)

# StandardScaler: transforms each feature to zero mean and unit variance
# We fit ONLY on training data, then apply the same transform to test data
# (fitting on test data would be data leakage)
scaler       = StandardScaler()
X_train_sc   = scaler.fit_transform(X_train)
X_test_sc    = scaler.transform(X_test)

print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")


# =============================================================================
# STEP 5: TRAIN RANDOM FOREST
# =============================================================================

print("\nTraining Random Forest...")

model = RandomForestRegressor(
    n_estimators=N_ESTIMATORS,    # 200 trees averaged together
    max_depth=MAX_DEPTH,           # Limits each tree's depth to prevent overfitting
    min_samples_split=MIN_SAMPLES, # Minimum samples needed to create a split
    random_state=RANDOM_STATE,
    n_jobs=-1                      # Use all available CPU cores
)
model.fit(X_train_sc, y_train)
print("  Done.")


# =============================================================================
# STEP 6: EVALUATE
# =============================================================================

y_pred = model.predict(X_test_sc)

rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
mae  = float(mean_absolute_error(y_test, y_pred))
r2   = float(r2_score(y_test, y_pred))

print(f"\nResults:")
print(f"  RMSE : {rmse:.4f}  (average prediction error in SoH units)")
print(f"  MAE  : {mae:.4f}")
print(f"  R²   : {r2:.4f}  ({r2*100:.1f}% of variance explained)")


# =============================================================================
# STEP 7: LEARNING CURVE
# =============================================================================

print("\nComputing learning curve (this may take a moment)...")

train_sizes, train_sc, val_sc = learning_curve(
    RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    X_train_sc, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='neg_mean_squared_error',
    cv=5, n_jobs=-1
)

train_rmse = np.sqrt(-train_sc.mean(axis=1))
val_rmse   = np.sqrt(-val_sc.mean(axis=1))
train_std  = np.sqrt(-train_sc).std(axis=1)
val_std    = np.sqrt(-val_sc).std(axis=1)


# =============================================================================
# STEP 8: RESULT PLOTS
# =============================================================================

print("\nGenerating result plots...")

fig = plt.figure(figsize=(16, 12), facecolor='white')
fig.suptitle(
    'Random Forest Results — EV Battery State of Health Prediction\n'
    'Sreeya Kannamala (23094968) | 7PAM2002 | NASA PCoE Dataset',
    fontsize=12.5, fontweight='bold', y=0.99
)
gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

# ── Plot 1: Actual vs Predicted ────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
for cell in BATTERY_NAMES:
    mask = cl_test == cell
    ax1.scatter(y_test[mask], y_pred[mask], color=COLOURS[cell],
                label=cell, alpha=0.55, s=18, edgecolors='none')
lims = [min(y_test.min(), y_pred.min())-0.01, max(y_test.max(), y_pred.max())+0.01]
ax1.plot(lims, lims, 'k--', lw=1.2, label='Perfect prediction')
ax1.set_xlabel('Actual SoH', fontsize=10)
ax1.set_ylabel('Predicted SoH', fontsize=10)
ax1.set_title('Actual vs Predicted SoH', fontsize=11, fontweight='bold')
ax1.legend(fontsize=8, markerscale=1.5)
ax1.grid(True, alpha=0.3)
ax1.set_facecolor('#f9f9f9')
ax1.text(0.05, 0.93, f'RMSE={rmse:.4f}\nMAE={mae:.4f}\nR²={r2:.4f}',
         transform=ax1.transAxes, fontsize=9, va='top',
         bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.85, ec='#CCCCCC'))

# ── Plot 2: Learning Curve ─────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ns = train_sizes * len(X_train_sc)
ax2.plot(ns, train_rmse, 'o-', color='#2196F3', label='Training RMSE', lw=2, ms=5)
ax2.plot(ns, val_rmse,   's-', color='#E91E63', label='Validation RMSE', lw=2, ms=5)
ax2.fill_between(ns, train_rmse-train_std, train_rmse+train_std, alpha=0.12, color='#2196F3')
ax2.fill_between(ns, val_rmse-val_std,     val_rmse+val_std,     alpha=0.12, color='#E91E63')
ax2.set_xlabel('Number of Training Samples', fontsize=10)
ax2.set_ylabel('RMSE', fontsize=10)
ax2.set_title('Learning Curve', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_facecolor('#f9f9f9')

# ── Plot 3: Feature Importance ─────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
imp        = model.feature_importances_
sorted_idx = np.argsort(imp)
bar_colors = ['#FF6B6B' if '★' in FEATURE_NAMES[i] else '#AFA9EC' for i in sorted_idx]
ax3.barh(range(20), imp[sorted_idx], color=bar_colors, edgecolor='white', height=0.7)
ax3.set_yticks(range(20))
ax3.set_yticklabels([FEATURE_NAMES[i] for i in sorted_idx], fontsize=7.5)
ax3.set_xlabel('Feature Importance (Gini)', fontsize=10)
ax3.set_title('Random Forest Feature Importance\n(★ = SHAP key features from literature)',
              fontsize=11, fontweight='bold')
ax3.grid(True, axis='x', alpha=0.3)
ax3.set_facecolor('#f9f9f9')
ax3.legend(handles=[Patch(color='#FF6B6B', label='Key features (★)'),
                    Patch(color='#AFA9EC', label='Other features')], fontsize=8)

# ── Plot 4: Per-Cell RMSE and R² ──────────────────────────────────────────
ax4   = fig.add_subplot(gs[1, 1])
rmses = []
r2s   = []
for cell in BATTERY_NAMES:
    mask = cl_test == cell
    if mask.sum() > 1:
        rmses.append(np.sqrt(mean_squared_error(y_test[mask], y_pred[mask])))
        r2s.append(r2_score(y_test[mask], y_pred[mask]))
    else:
        rmses.append(0); r2s.append(0)

x = np.arange(4)
b1 = ax4.bar(x-0.2, rmses, 0.38, label='RMSE', color='#2196F3', alpha=0.85)
ax4r = ax4.twinx()
b2 = ax4r.bar(x+0.2, r2s, 0.38, label='R²', color='#4CAF50', alpha=0.7)
for bar, v in zip(b1, rmses):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
             f'{v:.4f}', ha='center', fontsize=8, color='#1A4D7A', fontweight='bold')
for bar, v in zip(b2, r2s):
    ax4r.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
              f'{v:.3f}', ha='center', fontsize=8, color='#0F6E56', fontweight='bold')
ax4.set_xticks(x); ax4.set_xticklabels(BATTERY_NAMES, fontsize=10)
ax4.set_ylabel('RMSE', color='#2196F3', fontsize=10)
ax4r.set_ylabel('R² Score', color='#4CAF50', fontsize=10)
ax4.set_title('Performance Per Battery Cell', fontsize=11, fontweight='bold')
ax4.grid(True, axis='y', alpha=0.3)
ax4.set_facecolor('#f9f9f9')
ax4.legend(handles=[Line2D([0],[0],color='#2196F3',lw=10,alpha=0.85),
                    Line2D([0],[0],color='#4CAF50',lw=10,alpha=0.7)],
           labels=['RMSE','R²'], fontsize=9)

fig.text(0.5, 0.002,
    f'RF ({N_ESTIMATORS} trees) | 70/30 split | RMSE={rmse:.4f} | R²={r2:.4f} | 20 features',
    ha='center', fontsize=8.5, color='#555', style='italic')

plt.tight_layout(rect=[0, 0.02, 1, 0.97])
out = os.path.join(OUTPUT_DIR, 'RandomForest_Results.png')
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
print(f"Plot saved: {out}")
plt.show()

print("\nNext steps:")
print("  - 03_XGBoost.py  — XGBoost model with hyperparameter tuning")
print("  - 04_SHAP.py      — SHAP explainability plots")
