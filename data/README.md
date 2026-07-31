# Dataset Download Instructions

## NASA PCoE Li-Ion Battery Aging Dataset

This project uses the NASA Prognostics Center of Excellence Battery Aging Dataset.
It is publicly available and free — no registration or login required.

### Step 1 — Download

**Direct download link:**
```
https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip
```

**Official NASA page:**
```
https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
```

### Step 2 — Extract

Extract the zip file into this `data/` folder. You should see:

```
data/
├── B0005.mat
├── B0006.mat
├── B0007.mat
└── B0018.mat
```

### About the Data

- **Format:** MATLAB .mat files (loaded using scipy.io in Python)
- **Batteries:** 4 Li-Ion 18650 cells (B0005, B0006, B0007, B0018)
- **Cycles:** 132–168 discharge cycles per battery
- **End of life:** 30% capacity fade (2 Ah → 1.4 Ah)
- **Measurements per cycle:** Voltage (V), Current (A), Temperature (°C), Capacity (Ah)

### Licence

NASA is a US federal government agency. All data published through the NASA PCoE
repository is in the **public domain** under the US Government Open Data Policy.
No copyright restrictions. Free for academic and research use.

### Citation

> B. Saha and K. Goebel (2007). "Battery Data Set", NASA Prognostics Data Repository,
> NASA Ames Research Center, Moffett Field, CA.
> https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
