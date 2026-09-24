# TCO Material Recommendation System for Fluorescent Lamp Coatings

## Project Overview

This project implements a machine-learning system that recommends the most suitable Transparent Conducting Oxide (TCO) material for coating fluorescent-lamp glass or electrode surfaces. The user specifies desired levels for eight key properties; a Random Forest Regressor then supports ranking of candidate materials and generation of suitability-based recommendations.

**Important scientific note:** The numerical values currently stored in `data/tco_materials.csv` are **synthetic demonstration data**. They were created only to exercise the complete ML pipeline and to illustrate relative trends that appear in the open literature. They are **not** experimentally measured values and must not be treated as such. The CSV is deliberately structured so that it can later be replaced by literature-derived or laboratory-measured data without any change to the rest of the codebase.

## Problem Statement

Fluorescent lamps contain mercury vapour. Protective or functional coatings on the inner glass surface or near the electrodes can improve electrical behaviour, reduce mercury loss/corrosion, maintain optical transmission and increase long-term durability. Choosing the best TCO for a given set of operating requirements is therefore a multi-criteria materials-selection problem that benefits from a data-driven ranking approach.

## Motivation

- TCOs combine optical transparency with electrical conductivity – properties that are useful for lamp electrodes, charge dissipation layers and protective coatings.
- Different TCOs (ITO, FTO, AZO, ATO, GZO, IZO, ZnO, SnO₂, …) trade off conductivity, stability, mercury compatibility and cost.
- A reproducible ML pipeline makes it easy to update rankings when new experimental data become available.

## Objectives

1. Build a clean, modular Random-Forest-based recommendation pipeline.
2. Provide an interactive Streamlit interface for specifying property requirements and inspecting results.
3. Clearly separate demonstration/synthetic data from the claim of experimental validity.
4. Keep the code simple enough for a student to understand and extend.

## TCO Materials Considered

| Material | Full name / notes                  |
|----------|------------------------------------|
| ITO      | Indium Tin Oxide                   |
| FTO      | Fluorine-doped Tin Oxide           |
| AZO      | Aluminium-doped Zinc Oxide         |
| ATO      | Antimony-doped Tin Oxide           |
| GZO      | Gallium-doped Zinc Oxide           |
| IZO      | Indium Zinc Oxide                  |
| ZnO      | Zinc Oxide (undoped / lightly doped) |
| SnO₂     | Tin dioxide                        |

## The Eight Selected Properties

1. **Electrical Conductivity** – ability to conduct current (higher better).
2. **Optical Transparency** – visible-light transmission (higher better).
3. **Thermal Stability** – resistance to high-temperature processing / operation.
4. **Chemical Stability** – resistance to acids, bases and lamp chemistry.
5. **Mercury Corrosion Resistance** – ability to limit mercury-related degradation.
6. **Adhesion Strength** – mechanical bonding to glass / electrode substrates.
7. **Durability** – long-term mechanical and environmental robustness.
8. **Cost** – relative material + processing cost (lower better).

All demonstration values are expressed on a relative 1–10 scale.

## Dataset Description

File: `data/tco_materials.csv`

Columns:

- `Material`
- `Electrical_Conductivity`
- `Optical_Transparency`
- `Thermal_Stability`
- `Chemical_Stability`
- `Mercury_Corrosion_Resistance`
- `Adhesion_Strength`
- `Durability`
- `Cost`
- `Suitability_Score` (synthetic composite target used only for training the regressor)

**Status:** Demonstration / synthetic. Relative ordering roughly follows published trends (ITO high conductivity, FTO/SnO₂ high thermal & chemical stability, ZnO-family low cost, etc.). Replace this file with real data when available.

## Data Preprocessing

- Load CSV with pandas.
- Check and (if necessary) fill missing numerical values with the median.
- Separate the eight property columns as features and `Suitability_Score` as target.
- Standard-scale the features (`StandardScaler`).
- Train/test split with a fixed `random_state=42` for reproducibility.

## Why Random Forest?

- Handles non-linear relationships and feature interactions without extensive hyper-parameter tuning.
- Provides built-in feature-importance estimates that are later used as weights when ranking materials against user requirements.
- Robust on small tabular datasets (the current demonstration set contains only eight materials).
- Only algorithm used in this project, as required.

## Model Architecture / Workflow

```
CSV Dataset
    ↓
Data Cleaning (missing-value check)
    ↓
Feature Selection (8 properties)
    ↓
Standard Scaling
    ↓
Train / Test Split (random_state=42)
    ↓
RandomForestRegressor (n_estimators=100, max_depth=5, …)
    ↓
Evaluation (R², MAE, RMSE)
    ↓
Feature Importance extraction
    ↓
Persist model + scaler (joblib)
    ↓
User Requirements (qualitative → numeric)
    ↓
Weighted match scoring (RF importances as weights)
    ↓
Rank materials & recommend best
```

## Evaluation Metrics

Computed both on a held-out test split and on the full demonstration set:

- R² score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

Because the demonstration set is tiny, absolute metric values are mainly a sanity check that the pipeline runs.

## Feature Importance

The Random Forest reports the relative contribution of each property to the predicted suitability score. These importances are also used as weights when computing how well each material matches a user’s stated requirements.

## How User Requirements Are Converted into Model Inputs

Qualitative levels are mapped to the same 1–10 numeric scale used in the dataset:

| Level      | Numeric (positive properties) | Numeric (Cost – inverted) |
|------------|-------------------------------|---------------------------|
| Very Low   | 2.0                           | 9.5                       |
| Low        | 4.0                           | 8.0                       |
| Medium     | 6.0                           | 6.0                       |
| High       | 8.0                           | 4.0                       |
| Very High  | 9.5                           | 2.0                       |

Cost is inverted so that a user preference for “Very Low cost” becomes a high numerical “goodness” value.

## How Material Ranking Works

1. The trained Random Forest predicts a suitability score for every material (using the scaled property vectors).
2. Independently, a weighted match score is calculated between the user’s numeric preference vector and each material’s property vector. Feature importances from the Random Forest serve as the weights; the Cost column of each material is inverted so that lower real cost is treated as better.
3. Materials are ranked by the match score. Labels (“Highly Recommended”, “Recommended”, …) are assigned from the match-score magnitude.
4. A short natural-language explanation is generated from the actual property values of the top material and the user’s requested levels.

No material is hard-coded as the winner; the ranking is produced at run time from the model and the user inputs.

## Project Structure

```
TCO-Material-Recommender/
├── data/
│   └── tco_materials.csv          # demonstration dataset
├── model/
│   ├── train_model.py             # training & evaluation script
│   ├── random_forest.pkl          # persisted model
│   ├── scaler.pkl                 # persisted scaler
│   ├── feature_importance.png
│   └── actual_vs_predicted.png
├── app/
│   └── app.py                     # Streamlit application
├── notebooks/
│   └── analysis.ipynb             # exploratory notebook
├── requirements.txt
└── README.md
```

## Installation

```bash
# clone or copy the project folder
cd TCO-Material-Recommender

# (recommended) create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

## Training the Model

```bash
python model/train_model.py
```

This regenerates `random_forest.pkl`, `scaler.pkl` and the two diagnostic plots.

## Running the Streamlit Application

```bash
streamlit run app/app.py
```

Then open the URL shown in the terminal (usually http://localhost:8501).

## Deploying to the Cloud

The application is configured for **Streamlit Community Cloud** with Supabase
Storage used for the dataset and trained model artifacts.

### 1. Prepare Supabase Storage

Create a Storage bucket named `tco-files` and upload these files using the
same paths:

```
data/tco_materials.csv
model/random_forest.pkl
model/scaler.pkl
```

Use a read-only/public bucket policy appropriate for your Supabase project, or
use a server-side key with storage access. Do not commit the key to Git.

### 2. Publish the repository

Push this repository to GitHub. In Streamlit Community Cloud, choose **New
app**, select the repository and branch, and set the main file to:

```
app/app.py
```

### 3. Add Streamlit Cloud secrets

In the app's **Settings > Secrets** panel, add:

```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-supabase-key"
```

The local equivalent is shown in `.streamlit/secrets.toml.example`. The real
`.streamlit/secrets.toml` file is ignored by Git.

After saving the secrets, restart the app. Streamlit Cloud installs
`requirements.txt` automatically and serves the deployed URL it assigns.

## Limitations

- The current dataset contains only eight synthetic samples; metrics and importances are illustrative.
- No experimental validation of the recommendations has been performed inside real fluorescent lamps.
- Property scales are relative; absolute engineering decisions require measured values, thickness, deposition method, substrate compatibility, etc.
- Mercury-corrosion behaviour is highly sensitive to lamp chemistry, temperature and coating microstructure.

## Future Improvements

- Replace the synthetic CSV with carefully curated literature or laboratory data.
- Enlarge the material set and add process-related features (deposition temperature, thickness, substrate type).
- Add uncertainty quantification (e.g. quantile regression forests).
- Perform experimental validation of the top-ranked coatings.
- Deploy the Streamlit app behind authentication or as a lightweight web service.

## Experimental Validation Requirements

Before any recommendation from this system is used for real lamp design:

1. Obtain measured property values under conditions representative of fluorescent-lamp manufacturing and operation.
2. Confirm mercury-consumption and lumen-maintenance data for candidate coatings.
3. Verify adhesion, thermal cycling and chemical compatibility with the specific glass and electrode materials.
4. Re-train the Random Forest on the validated data and re-evaluate rankings.

---

*This project is intended for educational demonstration of a complete, Random-Forest-only materials-recommendation pipeline.*
