
# 🌍 Global Education & Development Analysis

This project analyzes global education indicators alongside socio‑economic development metrics using **World Bank Open Data**. It performs EDA, feature engineering, and machine‑learning–based prediction of education outcomes, all inside a **single Google Colab notebook**.

> **Main artifact**: `Global_Edu_Dev_Colab.ipynb`  
> Running the notebook reproduces the entire pipeline end‑to‑end.

---

## ✅ What I did (project summary)
- **Data acquisition**: Pulled multiple World Bank indicators (2000–2024) via `wbgapi`.
- **Robust shaping**: Handled API variations (wide tables with years as columns) by reshaping to a tidy panel with an explicit `year` column.
- **Feature engineering**:
  - Added **1‑year lags** for all numeric indicators (country‑wise).
  - Applied **log transform** to GDP per capita for scale robustness.
- **Exploratory analysis (EDA)**:
  - Trends of the chosen education target **by income group** across years.
  - **Missingness** profiling across indicators.
  - **Spearman correlation** heatmap on the latest year per country.
- **Modeling**:
  - Predictive target (default): **Secondary enrollment (gross)** `SE.SEC.ENRR`.
  - Preprocessing pipeline (impute → scale → one‑hot) + **XGBoost regressor**.
  - **GroupKFold (k=5) by country** to prevent leakage across years.
  - Metrics: **R²** and **MAE**, printed per fold and averaged.
- **Interpretability & diagnostics**:
  - **Permutation importance** (top features contributing to R²).
  - **Country‑level outlier analysis** via mean absolute error over time.
  - Example **Actual vs Predicted** plot for one country.
- **Artifacts**:
  - Saved trained pipeline to `models/model_<TARGET>.joblib`.
  - Exported latest‑year **predictions CSV** to `data/raw/latest_<TARGET>_predictions.csv`.

---

## 📂 Notebook outputs & folder layout (Colab)
When you run the notebook, it creates this structure under `/content/global-edu-dev/`:

```
global-edu-dev/
├─ configs/
│  └─ indicators.yaml        # list of target + feature indicator codes
├─ data/
│  ├─ raw/
│  │  ├─ wb_indicators.csv   # raw pull from World Bank
│  │  └─ latest_<TARGET>_predictions.csv
│  └─ processed/
│     └─ wb_panel.parquet    # engineered panel with lags/logs
├─ models/
│  └─ model_<TARGET>.joblib  # trained pipeline (preproc + model)
└─ Global_Edu_Dev_Colab.ipynb
```

---

## 🧮 Indicators used

**Education outcomes (candidate targets)**
- `SE.PRM.CMPT.ZS` – Primary completion rate
- `SE.SEC.ENRR` – Secondary enrollment (gross) ✅ *(default TARGET)*
- `SE.ADT.LITR.ZS` – Adult literacy rate
- `SE.XPD.TOTL.GD.ZS` – Government education expenditure (% of GDP)

**Socio‑economic predictors**
- `NY.GDP.PCAP.CD` – GDP per capita (current US$)
- `SI.POV.DDAY` – Poverty headcount ratio ($2.15/day, 2017 PPP)
- `SL.UEM.TOTL.ZS` – Unemployment (%)
- `IT.NET.USER.ZS` – Internet users (%)
- `EG.ELC.ACCS.ZS` – Electricity access (%)
- `SP.URB.TOTL.IN.ZS` – Urban population (% of total)
- `SP.DYN.LE00.IN` – Life expectancy at birth
- `SH.DYN.MORT` – Under‑5 mortality
- `SH.XPD.CHEX.PC.CD` – Health exp. per capita (current US$)
- `SP.POP.TOTL` – Total population
- `EN.ATM.CO2E.PC` – CO₂ emissions (metric tons per capita)

You can edit `configs/indicators.yaml` or directly tweak lists inside the notebook.

---

## ▶️ How to reproduce in **Google Colab**

1. Open **Google Colab** → upload `Global_Edu_Dev_Colab.ipynb`.
2. Run the first cell to install dependencies.
3. **Optional:** In the *Configuration* cell, change:
   - `TARGET` (e.g., to `SE.ADT.LITR.ZS`),
   - `START_YEAR`, `END_YEAR` (default: 2000–2024).
4. Run all cells. Artifacts will be saved under `/content/global-edu-dev/` (see layout above).

**Notes on robustness**  
World Bank responses can differ by `wbgapi` version. The notebook includes a **safe reshape** (wide→long→wide) so a proper `year` column always exists. It also handles the `indicator` vs `series` key mismatch you might encounter.

---

## 🧠 Methodology details

### Data design
- **Panel data:** country × year (2000–2024).
- Join with country metadata: **region**, **income group**, **lending type**.

### Preprocessing
- Numeric features: **median imputation**, **standard scaling**.
- Categorical: **most‑frequent imputation**, **one‑hot encoding** for region/income/lending.
- **Lag features:** 1‑year lag for all numeric indicators per country.
- **Transforms:** `log1p` on GDP per capita to stabilize variance.

### Modeling & validation
- Estimator: **XGBRegressor** (reasonable defaults).
- Validation: **GroupKFold (k=5) by country** to avoid country‑level leakage.
- Metrics: **R²** (fit quality) and **MAE** (error in target units).

### Interpretability & diagnostics
- **Permutation importance** (ΔR²) to rank influential variables.
- **Country MAE rankings** to surface positive/negative outliers.
- Actual vs Predicted plots for qualitative inspection.

---

## 📈 How to change the target & re‑run
In the *Configuration* cell of the notebook, set:
```python
TARGET = 'SE.SEC.ENRR'  # e.g., 'SE.ADT.LITR.ZS'
START_YEAR, END_YEAR = 2000, 2024
```
Run the rest of the cells. A new model and predictions CSV will be written with the updated target in the filename.

---

## 🔬 Limitations & caveats
- **Missing data**: Some indicators are sparse by country/year; imputation is used.
- **Correlation ≠ causation**: The model reveals associations, not causal effects.
- **Reporting inconsistencies**: Cross‑country differences in measurement can bias trends.
- **Temporal dynamics**: Lags are simple; more sophisticated time‑series models could help.
- **Forecast horizon**: This setup is near‑term (next‑year signal). For longer horizons, consider temporal splits and walk‑forward validation.

---

## 🚀 Ideas for next steps
- Add education quality metrics (test scores) if available.
- **Temporal CV** (train ≤ t, test at t+1) to evaluate true forecasting.
- Try **LightGBM**, **CatBoost**, or **Ridge/Lasso** for baselines and ensembles.
- Use **panel fixed‑effects** models for interpretable coefficients.
- Add **SHAP** for richer local/global explainability.
- Publish a simple dashboard (Streamlit) for exploration and “what‑if” sliders.

---

## 📜 Data source & attribution
- Indicators from **World Bank Open Data** via the [`wbgapi`](https://pypi.org/project/wbgapi/) Python package.  
  Please review indicator metadata on the World Bank website for definitions and caveats.

---

## 🗂 License & citation
- Add a license file (e.g., **MIT**) if you plan to share the repo.
- Cite the World Bank appropriately per their terms if you publish results.

---

## 🙋 Support
If you run into errors (e.g., missing `year`, `indicator` vs `series`), they’re usually due to `wbgapi` version/format differences. The notebook contains **fix cells** to standardize. Share any tracebacks and I’ll patch with a minimal drop‑in.
