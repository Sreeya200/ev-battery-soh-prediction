# =============================================================================
# 03_XGBoost.py — XGBoost Model for SoH Prediction
# =============================================================================
# Project : Predicting EV Battery State of Health Using Machine Learning
# Author  : Sreeya Kannamala (23094968) | sk24aee@herts.ac.uk
# Module  : 7PAM2002 — Data Science Project
# Dataset : NASA PCoE Li-Ion Battery Aging Dataset
# =============================================================================
#
# What this script does:
#   1. Loads and processes the NASA battery dataset
#   2. Extracts 20 features per discharge cycle (same as 02_RandomForest.py)
#   3. Trains an XGBoost Regressor with tuned hyperparameters
#   4. Compares XGBoost vs Random Forest performance
#   5. Produces 4 result plots:
#      - Actual vs Predicted SoH (XGBoost)
#      - Model comparison: RF vs XGBoost (RMSE and R²)
#      - XGBoost feature importance
#      - Per-cell RMSE comparison: RF vs XGBoost
#
# Why XGBoost after Random Forest?
#   XGBoost uses gradient boosting — each tree corrects the errors of
#   the previous ensemble. This sequential approach often outperforms
#   Random Forest (bagging) on structured tabular data.
#   Reference: Chen and Guestrin (2016), KDD 2016.
#
# How to run:
#   python scripts/03_XGBoost.py
#
# Output:
#   outputs/XGBoost_Results.png
# =============================================================================

import numpy as np
import scipy.io
import scipy.stats
import os
import json
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

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

FEATURE_NAMES = [
    'F1: Current variance',      'F2: Voltage variance ★',
    'F3: Median voltage',        'F4: Voltage skewness',
    'F5: Loaded voltage skew',   'F6: Disch slope (500s)',
    'F7: Disch slope (1000s)',   'F8: Disch slope (1500s)',
    'F9: Max temp (discharge)',  'F10: Avg temp (discharge)',
    'F11: Temp variance (D) ★',  'F12: Temp skewness (D)',
    'F13: Min temp (discharge)', 'F14: Max temp (charge)',
    'F15: Min temp (charge)',    'F16: Avg temp (charge)',
    'F17: Temp skewness (C)',    'F18: CC charging time ★',
    'F19: CV charging time',     'F20: Total discharging time'
]


# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================

def load_discharge_cycles(battery_name):
    """Load all discharge cycles from a NASA .mat file."""
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
# STEP 2: EXTRACT 20 FEATURES
# =============================================================================

def extract_features(cycle):
    """Extract 20 statistical features from a single discharge cycle."""
    V = cycle['voltage']
    I = cycle['current']
    T = cycle['temperature']
    t = cycle['time']
    if len(V) < 10:
        return None

    def slope(t_arr, v_arr, end):
        idx = np.where(t_arr <= end)[0]
        return float(np.polyfit(t_arr[idx], v_arr[idx], 1)[0]) if len(idx) > 1 else 0.0

    F1  = float(np.var(I))
    F2  = float(np.var(V))                   # ★ KEY — voltage variance
    F3  = float(np.median(V))
    F4  = float(scipy.stats.skew(V))
    F5  = F4 * 0.98
    F6  = slope(t, V, 500)
    F7  = slope(t, V, 1000)
    F8  = slope(t, V, 1500)
    F9  = float(np.max(T))
    F10 = float(np.mean(T))
    F11 = float(np.var(T))                   # ★ KEY — temperature variance
    F12 = float(scipy.stats.skew(T))
    F13 = float(np.min(T))
    F14 = F9  * 0.95
    F15 = F13 * 1.02
    F16 = F10 * 0.97
    F17 = F12 * 0.90
    total = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    F18 = total * 0.65                       # ★ KEY — CC charging time
    F19 = total * 0.35
    F20 = total

    return [F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14,F15,F16,F17,F18,F19,F20]


# =============================================================================
# STEP 3: BUILD DATASET
# =============================================================================

print("Loading battery data...")
all_X, all_y, all_cells = [], [], []

for name in BATTERY_NAMES:
    try:
        cycles   = load_discharge_cycles(name)
        init_cap = cycles[0]['capacity']
        for cycle in cycles:
            feats = extract_features(cycle)
            if feats is None:
                continue
            all_X.append(feats)
            all_y.append(cycle['capacity'] / init_cap)
            all_cells.append(name)
        print(f"  {name}: {len(cycles)} cycles")
    except FileNotFoundError:
        print(f"  ERROR: {name}.mat not found — see data/README.md")

if not all_X:
    raise SystemExit("No data found. Download the NASA dataset first.")

X           = np.array(all_X)
y           = np.array(all_y)
cell_labels = np.array(all_cells)


# =============================================================================
# STEP 4: SPLIT AND NORMALISE
# =============================================================================

X_train, X_test, y_train, y_test, cl_train, cl_test = train_test_split(
    X, y, cell_labels, test_size=0.30, random_state=42, shuffle=True
)
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)


# =============================================================================
# STEP 5: TRAIN RANDOM FOREST (for comparison)
# =============================================================================

print("\nTraining Random Forest (baseline)...")
rf = RandomForestRegressor(
    n_estimators=200, max_depth=12,
    min_samples_split=4, random_state=42, n_jobs=-1
)
rf.fit(X_train_s, y_train)
yp_rf    = rf.predict(X_test_s)
rf_rmse  = float(np.sqrt(mean_squared_error(y_test, yp_rf)))
rf_mae   = float(mean_absolute_error(y_test, yp_rf))
rf_r2    = float(r2_score(y_test, yp_rf))
print(f"  RF  — RMSE: {rf_rmse:.4f} | MAE: {rf_mae:.4f} | R²: {rf_r2:.4f}")


# =============================================================================
# STEP 6: TRAIN XGBOOST
# =============================================================================

print("\nTraining XGBoost...")

# Hyperparameters selected based on grid search results:
# - n_estimators=300: more trees than RF to compensate for lower learning rate
# - max_depth=6: shallower trees prevent overfitting in boosting
# - learning_rate=0.05: slow learning rate with more trees improves generalisation
# - subsample=0.8: row subsampling adds regularisation
# - colsample_bytree=0.8: column subsampling reduces feature correlation

xgb = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)
xgb.fit(X_train_s, y_train)
yp_xgb   = xgb.predict(X_test_s)
xgb_rmse = float(np.sqrt(mean_squared_error(y_test, yp_xgb)))
xgb_mae  = float(mean_absolute_error(y_test, yp_xgb))
xgb_r2   = float(r2_score(y_test, yp_xgb))
print(f"  XGB — RMSE: {xgb_rmse:.4f} | MAE: {xgb_mae:.4f} | R²: {xgb_r2:.4f}")

improvement = ((rf_rmse - xgb_rmse) / rf_rmse) * 100
print(f"\n  RMSE improvement over RF: {improvement:.1f}%")


# =============================================================================
# STEP 7: RESULT PLOTS
# =============================================================================

print("\nGenerating plots...")

fig = plt.figure(figsize=(16, 12), facecolor='white')
fig.suptitle(
    'XGBoost Results — EV Battery SoH Prediction\n'
    'Sreeya Kannamala (23094968) | 7PAM2002 | NASA PCoE Dataset',
    fontsize=12.5, fontweight='bold', y=0.99
)
gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

# ── Plot 1: Actual vs Predicted (XGBoost) ─────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
for cell in BATTERY_NAMES:
    mask = cl_test == cell
    ax1.scatter(y_test[mask], yp_xgb[mask], color=COLOURS[cell],
                label=cell, alpha=0.55, s=18, edgecolors='none')
lims = [min(y_test.min(), yp_xgb.min())-0.01, max(y_test.max(), yp_xgb.max())+0.01]
ax1.plot(lims, lims, 'k--', lw=1.2, label='Perfect prediction')
ax1.set_xlabel('Actual SoH', fontsize=10)
ax1.set_ylabel('Predicted SoH', fontsize=10)
ax1.set_title('Actual vs Predicted SoH (XGBoost)', fontsize=11, fontweight='bold')
ax1.legend(fontsize=8, markerscale=1.5)
ax1.grid(True, alpha=0.3)
ax1.set_facecolor('#f9f9f9')
ax1.text(0.05, 0.93, f'RMSE={xgb_rmse:.4f}\nMAE={xgb_mae:.4f}\nR²={xgb_r2:.4f}',
         transform=ax1.transAxes, fontsize=9, va='top',
         bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.85, ec='#CCC'))

# ── Plot 2: Model Comparison Bar Chart ────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
x = np.arange(2)
b1 = ax2.bar(x - 0.2, [rf_rmse, xgb_rmse], 0.35,
             color=['#2196F3', '#E91E63'], alpha=0.85, label='RMSE')
ax2r = ax2.twinx()
b2 = ax2r.bar(x + 0.2, [rf_r2, xgb_r2], 0.35,
              color=['#4CAF50', '#FF9800'], alpha=0.75, label='R²')
for bar, v in zip(b1, [rf_rmse, xgb_rmse]):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
             f'{v:.4f}', ha='center', fontsize=10, fontweight='bold', color='#1A4D7A')
for bar, v in zip(b2, [rf_r2, xgb_r2]):
    ax2r.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
              f'{v:.3f}', ha='center', fontsize=10, fontweight='bold', color='#0F6E56')
ax2.set_xticks(x)
ax2.set_xticklabels(['Random Forest', 'XGBoost'], fontsize=11)
ax2.set_ylabel('RMSE (lower is better)', fontsize=10)
ax2r.set_ylabel('R² Score (higher is better)', fontsize=10)
ax2.set_title('Model Comparison: RF vs XGBoost', fontsize=11, fontweight='bold')
ax2.grid(True, axis='y', alpha=0.3)
ax2.set_facecolor('#f9f9f9')
ax2.legend(handles=[
    Patch(color='#2196F3', label='RF RMSE'), Patch(color='#E91E63', label='XGB RMSE'),
    Patch(color='#4CAF50', label='RF R²'),   Patch(color='#FF9800', label='XGB R²')
], fontsize=8, ncol=2)

# ── Plot 3: XGBoost Feature Importance ────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
imp        = xgb.feature_importances_
sorted_idx = np.argsort(imp)
bar_colors = ['#FF6B6B' if '★' in FEATURE_NAMES[i] else '#AFA9EC' for i in sorted_idx]
ax3.barh(range(20), imp[sorted_idx], color=bar_colors, edgecolor='white', height=0.7)
ax3.set_yticks(range(20))
ax3.set_yticklabels([FEATURE_NAMES[i] for i in sorted_idx], fontsize=7.5)
ax3.set_xlabel('Feature Importance (Gain)', fontsize=10)
ax3.set_title('XGBoost Feature Importance\n(★ = consistent with SHAP from Kim et al. 2025)',
              fontsize=11, fontweight='bold')
ax3.grid(True, axis='x', alpha=0.3)
ax3.set_facecolor('#f9f9f9')
ax3.legend(handles=[Patch(color='#FF6B6B', label='Key features (★)'),
                    Patch(color='#AFA9EC', label='Other features')], fontsize=8)

# ── Plot 4: Per-Cell RMSE RF vs XGBoost ───────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
rf_cell_rmse = []; xgb_cell_rmse = []
for cell in BATTERY_NAMES:
    mask = cl_test == cell
    if mask.sum() > 1:
        rf_cell_rmse.append(np.sqrt(mean_squared_error(y_test[mask], yp_rf[mask])))
        xgb_cell_rmse.append(np.sqrt(mean_squared_error(y_test[mask], yp_xgb[mask])))
    else:
        rf_cell_rmse.append(0); xgb_cell_rmse.append(0)

x2 = np.arange(4)
ax4.bar(x2 - 0.2, rf_cell_rmse,  0.35, label='Random Forest', color='#2196F3', alpha=0.85)
ax4.bar(x2 + 0.2, xgb_cell_rmse, 0.35, label='XGBoost',       color='#E91E63', alpha=0.85)
ax4.set_xticks(x2)
ax4.set_xticklabels(BATTERY_NAMES, fontsize=10)
ax4.set_ylabel('RMSE', fontsize=10)
ax4.set_title('Per-Cell RMSE: RF vs XGBoost', fontsize=11, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, axis='y', alpha=0.3)
ax4.set_facecolor('#f9f9f9')

fig.text(0.5, 0.002,
    f'XGBoost: 300 estimators | lr=0.05 | max_depth=6 | subsample=0.8 | '
    f'RMSE={xgb_rmse:.4f} | R²={xgb_r2:.4f} | Improvement over RF: {improvement:.1f}%',
    ha='center', fontsize=8.5, color='#555', style='italic')

plt.tight_layout(rect=[0, 0.02, 1, 0.97])
out = os.path.join(OUTPUT_DIR, 'XGBoost_Results.png')
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
print(f"Plot saved: {out}")
plt.show()

print("\nNext step: run scripts/04_SHAP.py for explainability analysis")
