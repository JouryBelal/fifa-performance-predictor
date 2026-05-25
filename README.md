# FIFA Player Performance Predictor

A machine learning system that predicts FIFA player **market value** (regression) and **performance tier** (classification) using ensemble models built with Python and Scikit-learn.

---

## Results

### Regression — Predicting Player Market Value (Value Per M$)

| Model | R² Score | RMSE |
|---|---|---|
| Baseline (Ridge Regression) | 0.535 | 5.371 |
| **Ensemble (KNN + SVR + Random Forest)** | **0.964** | **2.641** |

> Ensemble model improved R² by **80% over baseline**
> Cross-validation stability: R² = 0.9635 ± 0.0094 across 5 folds

### Classification — Predicting Performance Tier (Weak / Moderate / Good / Very Good)

| Model | Accuracy | F1 Score |
|---|---|---|
| Baseline (Logistic Regression) | 80.4% | 0.804 |
| **Ensemble (KNN + SVC + Random Forest)** | **85.3%** | **0.853** |

> Cross-validation stability: 85.1% ± 0.41% across 3 folds

---

## Models & Techniques

### Baseline (Assignment 1 & 2)
- Linear Regression, Ridge Regression
- Logistic Regression
- Full preprocessing pipeline: MinMax scaling, log transforms, one-hot encoding, median imputation

### Ensemble (Assignment 3)
- **Voting Regressor** — tuned KNN + SVR + Random Forest
- **Soft Voting Classifier** — tuned KNN + SVC + Random Forest
- Hyperparameter tuning via **GridSearchCV**
- Custom **outlier handler** transformer (IQR-based clipping)
- Bias-variance diagnosis on all models

### Best Hyperparameters Found
| Model | Key Parameters |
|---|---|
| Random Forest (Reg) | n_estimators=200, max_depth=None |
| SVR | C=10, kernel=rbf, epsilon=0.05 |
| KNN (Reg) | n_neighbors=9, weights=uniform |
| Random Forest (Clf) | n_estimators=200, max_depth=None |
| SVC | C=1, kernel=rbf |
| KNN (Clf) | n_neighbors=9, weights=uniform |

---

## Tools & Libraries

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## How to Run

1. Clone the repository
```bash
git clone https://github.com/JouryBelal/fifa-performance-predictor.git
```

2. Install dependencies
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

3. Add the dataset files:
   - `Fifa.csv` — for the FIFA prediction notebooks
   - `car_price.csv` — for the baseline regression notebook

4. Run the notebooks in order:
   - `assignment1_final.ipynb` — EDA, preprocessing, and baseline models
   - `assignment3_completed_from_assignment2.ipynb` — ensemble models, tuning, and evaluation

---

## Project Structure

```
fifa-performance-predictor/
├── assignment1_final.ipynb                        # EDA + baseline models
├── assignment3_completed_from_assignment2.ipynb   # Ensemble models + tuning
└── README.md
