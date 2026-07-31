# Predicting EV Battery State of Health Using Machine Learning
### An Explainable AI Approach

**Student:** Sreeya Kannamala | **ID:** 23094968 | **Email:** sk24aee@herts.ac.uk  
**Module:** 7PAM2002 — Data Science Project | **University:** University of Hertfordshire  
**Supervisor:** Zena Poudel

---

## Project Overview

Electric vehicle (EV) batteries degrade over time, losing capacity with every charge and discharge cycle. Accurately predicting a battery's **State of Health (SoH)** is critical for preventing unexpected failures and reducing the £89 billion annual cost of premature battery replacements.

Most existing ML models are **black boxes** — they predict but cannot explain *why*. This project builds ML models that are both accurate and **explainable**, using cross-model feature importance analysis validated against SHAP findings from the literature.

---

## Research Question

> Can machine learning models accurately predict the State of Health (SoH) of lithium-ion EV batteries from charge/discharge cycle data, and can feature importance analysis identify the key degradation features to make predictions interpretable for real-world battery management systems?

---

## Dataset

**NASA Prognostics Center of Excellence (PCoE) — Battery Aging Dataset**

| Field | Details |
|---|---|
| Source | NASA Ames Research Center, California, USA |
| Year | 2007 |
| Licence | Public domain — free, no registration required |
| Download | https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip |
| Batteries | B0005, B0006, B0007, B0018 (4 Li-Ion 18650 cells) |
| Cycles | 132–168 per battery until end of life (30% capacity fade) |

---

## Project Structure

```
ev-battery-soh-prediction/
│
├── README.md                    ← This file
├── requirements.txt             ← Python dependencies
│
├── data/
│   └── README.md                ← Dataset download instructions
│
├── scripts/
│   ├── 01_EDA.py                ← Exploratory Data Analysis
│   ├── 02_RandomForest.py       ← Random Forest baseline model
│   ├── 03_XGBoost.py            ← XGBoost model and comparison
│   └── 04_SHAP.py               ← Feature importance and SHAP analysis (novel contribution)
│
└── outputs/                     ← Generated plots
    ├── EDA_Battery_SoH.png
    ├── RandomForest_Results.png
    ├── XGBoost_Results.png
    ├── SHAP_Feature_Importance.png
    └── Cross_Model_Agreement.png
```

---

## Methodology

### Features Extracted — 20 Per Cycle

| Group | Features | Key Features |
|---|---|---|
| Voltage (discharge) | F1–F5 | **F2: Voltage variance ★** |
| Voltage slope | F6–F8 | — |
| Temperature (discharge) | F9–F13 | **F11: Temp variance ★** |
| Temperature (charge) | F14–F17 | — |
| Charging time | F18–F19 | **F18: CC charging time ★** |
| Discharging time | F20 | — |

★ = Top features consistently identified by RF (Gini), XGBoost (Gain), and SHAP (Kim et al., 2025)

### Models

| Model | Status | RMSE | MAE | R² |
|---|---|---|---|---|
| Random Forest | ✅ Complete | 0.0436 | 0.0352 | 0.636 |
| XGBoost | ✅ Complete | 0.0427 | 0.0335 | 0.652 |
| DLinear (Kim et al., 2025) | 📄 Benchmark | ~0.005 | — | — |

---

## Key Results

- **XGBoost improves over Random Forest:** RMSE reduced from 0.0436 → 0.0427
- **Novel finding:** F2 (voltage variance), F11 (temperature variance), and F18 (CC charging time) are ranked as the top 3 most important features by **all three independent methods** — Random Forest Gini impurity, XGBoost gain, and SHAP from Kim et al. (2025)
- **Physical interpretation:** These features directly reflect the electrochemical mechanisms of Li-ion degradation (SEI growth, increased internal resistance, reduced charge acceptance)
- **Real-world implication:** Explainable models can report which physical measurements drove each SoH prediction — essential for regulated deployment in EV battery management systems

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Sreeya200/ev-battery-soh-prediction.git
cd ev-battery-soh-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NASA dataset (see data/README.md)

# 4. Run in order:
python scripts/01_EDA.py
python scripts/02_RandomForest.py
python scripts/03_XGBoost.py
python scripts/04_SHAP.py
```

---

## References

1. Kim, J. et al. (2025). *State-of-Health Prediction for EV Lithium-Ion Batteries via DLinear and Robust Explainable Feature Selection.* arXiv:2501.11542
2. Saha, B. and Goebel, K. (2007). *Battery Data Set.* NASA PCoE, NASA Ames Research Center.
3. Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), pp. 5–32.
4. Chen, T. and Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD 2016.
5. Lundberg, S.M. and Lee, S.I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS 30.
