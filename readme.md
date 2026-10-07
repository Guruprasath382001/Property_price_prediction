# Property Price Prediction

A machine learning project that predicts house prices using property characteristics such as living area, bedrooms, bathrooms, location, condition, waterfront status, and view.

The project follows a structured machine learning workflow:

```text
Raw Dataset
    ↓
Price Outlier Analysis
    ↓
Data Cleaning
    ↓
Model Tuning
    ↓
Final Model Training
    ↓
Model Evaluation / Error Analysis
```

---


# 2. Dataset

The original dataset is:

```text
data/data.csv
```

The dataset contains property information such as:

- Date
- Price
- Bedrooms
- Bathrooms
- Living area
- Lot area
- Floors
- Waterfront
- View
- Condition
- Above-ground area
- Basement area
- Year built
- Renovation year
- Street
- City
- State ZIP
- Country

The target variable is:

```text
price
```

---

# 3. Machine Learning Pipeline

The scripts should be executed in the following order.

```text
1. price_outlier_analysis.py
            ↓
2. data_cleaning.py
            ↓
3. tune_models.py
            ↓
4. train.py
            ↓
5. error_analysis.py
```

Each step has a different responsibility.

---

# 4. Step 1 — Price Outlier Analysis

File:

experiments/price_outlier_analysis.py

### Purpose

This script investigates the original dataset and identifies unusual property prices.

It does not directly create the final cleaned dataset.

Instead, it produces an analysis file:

```text
experiments_results/price_outlier_analysis.csv
```

The analysis classifies records based on the price investigation.

For example:

```text
NORMAL
INVALID_PRICE
EXTREME_PRICE_ANOMALY
SUSPICIOUS_LOW_PRICE
SUSPICIOUS_HIGH_PRICE
```

The output of this step is used by the cleaning process.

---

# 5. Step 2 — Clean the Dataset

File:

```text
src/data_cleaning.py
```

### Purpose

This script creates the final dataset used by the machine learning models.

It reads:

```text
data/data.csv
```

and:

1. Reads the results from:

```text
experiments_results/price_outlier_analysis.csv
```

2. Keeps records classified as:

```text
NORMAL
```

3. Removes unusable columns such as:

```text
street
country
```

4. Handles invalid renovation years.

5. Saves the final dataset as:

```text
data/cleaned_data.csv
```

# 6. Step 3 — Model Tuning

File:

```text
experiments/tune_models.py
```


### Purpose

This script compares multiple machine learning algorithms and searches for their best hyperparameters.

The candidate models include:

- Linear Regression
- Ridge Regression
- Gradient Boosting
- Random Forest

The model is evaluated using cross-validation on the development dataset.

The test dataset is not used for model selection.

The results are saved to:

```text
experiments_results/model_tuning_results.json
```

The model with the best cross-validation RMSE is selected.

---

# 7. Step 4 — Train the Final Model

File:

```text
src/train.py
```


### Purpose

This script trains the final selected model using the selected hyperparameters.

The current selected model is:

```text
Ridge Regression
```

with:

```text
alpha = 1
```

The final model is evaluated once on the held-out test dataset.

The trained pipeline is saved to:

```text
models/house_price_pipeline.pkl
```

Model metadata is saved to:

```text
models/metadata.json
```

---

# 8. Step 5 — Error Analysis

File:

```text
experiments/error_analysis.py
```

### Purpose

This script investigates where the final model makes prediction errors.

It reports:

- MAE
- RMSE
- R²
- Largest prediction errors
- Underprediction vs overprediction
- Error by actual price range
- Error by city
- Error distribution

The detailed results are saved to:

```text
experiments_results/error_analysis.csv
```

The summary is saved to:

```text
experiments_results/error_analysis_summary.json
```

This step is diagnostic.

Large prediction errors should not automatically be treated as bad data.

---

Therefore the complete pipeline is:

```text
data/data.csv
     │
     ▼
price_outlier_analysis.py
     │
     │
     ▼
experiments_results/price_outlier_analysis.csv
     │
     ▼
data_cleaning.py
     │
     │
     ▼
data/cleaned_data.csv
     │
     ▼
tune_models.py
     │
     │
     ▼
experiments_results/model_tuning_results.json
     │
     ▼
train.py
     │
     ├──► models/house_price_pipeline.pkl
     │
     └──► models/metadata.json
     │
     ▼
error_analysis.py
     │
     ├──► error_analysis.csv
     │
     └──► error_analysis_summary.json
```
---


# 11. Log-Target Experiment

An additional experiment was performed using a logarithmic transformation of the target variable.

File:

```text
experiments/tune_log_target.py
```


The experiment compares:

```text
Raw Target
vs
Log Target
```

For this dataset, the raw-target Ridge model performed better.

Therefore the current production model uses the raw target:

```text
price
```

rather than:

```text
log1p(price)
```

The log-target experiment is kept separately because it is an experiment, not part of the current production pipeline.

---


# 12. Important Machine Learning Principles Used

### Train/Test Separation

The test dataset is kept separate from model selection.

Model tuning uses the development data with cross-validation.

The test dataset is used for final evaluation.

```text
Development Data
       │
       ├── Cross Validation
       │
       └── Model Selection
       
Test Data
       │
       └── Final Evaluation
```

### No Blind Outlier Removal

The project does not remove every statistically unusual property.

An expensive house is not automatically bad data.

Outlier investigation is performed first, and only records classified as non-normal by the project's analysis are removed during cleaning.

### Error Analysis Is Diagnostic

A large prediction error does not automatically mean the corresponding row is incorrect.

It may indicate:

- Missing features
- Location effects
- Luxury properties
- Non-linear relationships
- Limitations of the model

Therefore error analysis should be used to understand the model before changing the dataset.

---

# 13. Environment

Recommended Python version:

```text
Python 3.11.9
```

Create virtual environment:

```bash
python -m venv property_venv
```

Activate on Windows:

```bash
property_venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 14. Running the API

After the model has been trained:

```bash
uvicorn main:app --reload
```

The FastAPI application will run at:

```text
http://127.0.0.1:8000
```

The API uses the trained model saved in:

```text
models/house_price_pipeline.pkl
```

---

# 15. Re-running the Project

Whenever the raw dataset changes, run the complete pipeline again:


Do not run `train.py` before `data_cleaning.py`, because the training pipeline expects:

```text
data/cleaned_data.csv
```

Similarly, do not run `data_cleaning.py` before `price_outlier_analysis.py`, because the cleaning process depends on:

```text
experiments_results/price_outlier_analysis.csv
```