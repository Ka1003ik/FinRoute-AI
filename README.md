# FinRoute AI

FinRoute AI is a script-based NLP project that classifies public financial complaint narratives by product to support complaint routing.

## Current status

- Python 3.14.10 virtual environment is set up.
- CFPB historical exports from January through July 2026 were profiled.
- CFPB notes that its published complaint database is not a representative sample of all consumers' experiences; treat model results as a routing prototype, not population estimates.
- The modeling table contains 121,053 rows across 11 product categories after excluding cross-label feature conflicts and rows with no baseline tokens.
- Same-label repeated complaints are retained. Rows whose cleaned baseline TF-IDF feature signature maps to multiple product labels are excluded from the single-label modeling table and counted in the data-preparation report.
- The shared baseline cleaner normalizes Unicode, unescapes HTML entities, removes standalone repeated-x redaction markers, collapses whitespace, and lowercases text. It preserves numbers, punctuation, negation, and financial terms.
- Train, validation, and test partitions have been created with an 80/10/10 target ratio. Cleaned baseline TF-IDF feature signatures do not overlap across partitions.
- The initial word TF-IDF benchmark uses `min_df=3`, up to 300,000 features, and unigrams plus bigrams. It is fitted on training text only.
- Seven candidates have validation results: Multinomial Naive Bayes, Complement Naive Bayes, one-vs-rest Logistic Regression, LinearSVC, SGDClassifier, Random Forest, and XGBoost. LinearSVC currently leads on validation macro F1 (0.7738). SGDClassifier has the highest macro recall so far (0.7867); Random Forest scores macro F1 0.6666 and XGBoost 0.6870.
- The current selected artifact is `models/baseline_best.joblib`. Model selection is provisional: the remaining candidates, tuning, error analysis, calibration, and untouched test evaluation are still pending.
- Shared evaluation metrics live in `src/evaluation/metrics.py`; reusable SGD configuration lives in `src/models/sgd.py`.
- The Random Forest baseline uses a training-fitted chi-square selector to reduce TF-IDF to 10,000 features before fitting the forest; its settings are recorded in the validation report.
- XGBoost 3.4.1 is pinned in the direct and locked dependency files. Its baseline also uses 10,000 training-selected chi-square features, sparse CPU histogram trees, and balanced sample weights computed from training labels.
- The test partition has not been used. LinearSVC top-2 ranking uses decision scores, which are not calibrated probabilities and must not drive confidence-based routing.

## Product architecture and roadmap

The target product is an intelligent complaint classification and routing system. Its planned stack is Python, pandas/NumPy, scikit-learn, NLTK/spaCy experiments, XGBoost, PyTorch/Transformers with DistilBERT, FastAPI/Uvicorn, Streamlit, PostgreSQL, MLflow, and Docker Compose. Dependencies will be added only when their project phase begins; the current environment is Python 3.14.10.

Development proceeds in these phases:

1. Environment and project setup.
2. CFPB historical data acquisition and validation.
3. Exploratory analysis and reports.
4. Reusable, leakage-aware text preprocessing.
5. Split strategy comparison and leakage checks.
6. Word, character, and combined TF-IDF; separate DistilBERT tokenization.
7. Compare MultinomialNB, ComplementNB, Logistic Regression, LinearSVC, SGDClassifier, Random Forest, XGBoost, and DistilBERT.
8. Tune with training-only cross-validation and validation selection.
9. Evaluate finalists consistently on one held-out test set.
10. Error analysis and explainability.
11. Validation-based confidence calibration and human-aware routing.
12. FastAPI backend.
13. Streamlit frontend connected to the API.
14. PostgreSQL application records and monitoring.
15. Docker packaging, documentation, and deployment.

The project already completed environment setup, data validation, EDA, preprocessing, and an initial grouped 80/10/10 split. The attached master prompt recommends 70/15/15 and asks us to consider chronological evaluation; those will be measured as separate split-strategy experiments rather than silently replacing the current leakage-checked baseline. Likewise, its suggested TF-IDF settings (`min_df=2`, `max_df=0.95`, `sublinear_tf=True`, `max_features=100000`) will be evaluated as a separate configuration against the current 300,000-feature baseline.

Complaint-to-department routing will use configurable mappings and human review for uncertain cases. Raw LinearSVC decision scores will not be presented as probabilities. The application will avoid retaining unnecessary complaint text or personal information.

## Project principles

- Use Python scripts, not notebooks.
- Keep data validation, preprocessing, training, evaluation, and inference modular.
- Keep source data, derived datasets, and model artifacts out of version control.
- Split data before fitting learned text features.
- Keep repeated feature-equivalent narratives in one partition to prevent leakage.
- Compare models using macro F1 and per-class metrics, not accuracy alone.

## Reproducibility

- `requirements.txt` pins the direct project dependencies.
- `requirements-lock.txt` records the full installed Python environment.
- The split seed is `42`; split details and the input file hash are recorded in `outputs/reports/split_summary.json`.
- Aggregate reports and figures are kept in version control; complaint records and prediction outputs are ignored.

## Environment

The current project environment uses Python 3.14.10 in `.venv`. Activate it in PowerShell with:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the exact environment from the lock file with:

```powershell
python -m pip install -r requirements-lock.txt
```

