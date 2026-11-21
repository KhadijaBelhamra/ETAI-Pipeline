# Baseline Predictive Pipeline -- ETAI
20230007 – Khadija Belhamra

This is the **starting point** for your semester project: a small but *complete* predictive pipeline -- every piece a real project needs (entry point, config, data loading, preprocessing, model, evaluation), just kept as simple as possible for now.

The task: predict two-year recidivism using ProPublica's COMPAS
dataset -- the data behind a real 2016 investigation into a risk-
assessment algorithm actually used by US courts to help inform bail and sentencing decisions. See `data/README.md` for the full problem description and a complete data dictionary before you start.

It has some **deliberately weak spots**. Part of your work this
semester is finding them and making them better -- see the pipeline progress table below, which tracks what changes and why as the weeks
go on.

## Project structure

```
.
├── main.py                # entry point: train, evaluate and save the model
├── predict.py             # entry point: predict a new CSV with the saved model (week 10)
├── run_experiments.py      # runs a week's set of experiments and prints the scoreboard
├── config.yaml             # all tunable settings live here (COMPAS)
├── config_adult.yaml       # same, for the Adult income data (week 9)
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── data_diagnostics.py # exploration helpers (not called by main.py)
│   ├── preprocessing.py    # cleaning, engineered features, preprocessing pipeline, dev/test split
│   ├── model.py            # model + pipeline construction (forests, boosting incl. XGBoost, VoteSelector feature selection)
│   ├── tuning.py           # Optuna tuning and nested cross-validation
│   ├── evaluate.py         # cross-validation, several metrics (incl. F-beta), fairness check
│   └── results.py          # saves each run's report and the scoreboard to disk
├── results/                # COMPAS runs, created automatically
├── results_adult/          # Adult runs, created automatically
├── models/                 # created automatically: model.joblib, the final fitted pipeline (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    ├── compas_new_people.csv    # 20 new people, no target, for predict.py
    ├── adult_train.csv, adult_new.csv
    └── README.md            # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Added to the pipeline |         Comments         |
|------|------------------------|---------------------------|
| 2 | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` | We have: <br> Logistic reg train accuracy: 0.680 <br> Logistic reg test accuracy:  0.677 <br> Gap (train - test): +0.002 <br><br> Tree train accuracy: 0.829 <br> Tree test accuracy:  0.626 <br> Gap (train - test): +0.203 <br> <br> The logistic regression has a better test score than the tree model. We should also note that the decision tree is clearly overfitting with a gap of over 0.2 units between the train and test accuracy. The decision tree needs to be parametrized to reduce overfitting and improve its performance on unseen data.|
| 3 | Added a configuration-driven `clean_dataset()` function to the pipeline, applied before the existing baseline preprocessing stage. The function standardizes categorical values, converts placeholder tokens and domain-rule violations into missing values, removes exact duplicate rows and repeated IDs, and drops redundant columns. The existing `preprocess()` function is retained for feature/target separation and the stratified train/test split. No new imputation, encoding, or other feature transformations are introduced this week. | Cleaning had different effects across models. Logistic regression showed a small decrease in test accuracy (0.677 → 0.669). The decision tree improved on the test set (0.626 → 0.650) while its train–test gap decreased substantially (0.203 → 0.145), indicating reduced overfitting after cleaning.|
| 4 | Reworked the evaluation pipeline around a configuration-driven scikit-learn Pipeline, with preprocessing fitted inside each cross-validation fold. Added median/mode imputation, robust scaling, configurable categorical encoding with target encoding, and missingness indicators. The dataset is now split into a development set and a locked test set, with 5-fold stratified cross-validation performed only on the development set. Added out-of-fold predictions for the classification report and fairness analysis, refit the final model on the full development set, and added configurable model selection with a Dummy classifier, logistic regression, and Random Forest. Results are automatically saved in separate model-specific folders under results/. | The Dummy classifier provides a baseline accuracy of 0.549, with no predictive ability for class 1. Logistic regression achieved a mean validation accuracy of 0.672 (std = 0.013), with a small mean train–validation gap of +0.003. Random Forest achieved a mean validation accuracy of 0.652 (std = 0.017), with a larger mean train–validation gap of +0.080, indicating more overfitting than logistic regression under the current configuration. The out-of-fold fairness results also show different FPR patterns across races for the three models. The test set remains locked and has not been evaluated.|
| 5 | **Feature engineering:** six row-wise features added in `src/preprocessing.py` (`add_engineered_features`) and switched on/off from `preprocessing.engineered_features` in the YAML: `log_priors`, `has_priors`, `priors_per_adult_year`, `has_juvenile_record`, `has_juvenile_felony`, `young_with_priors`. They learn nothing from the data, so they are leak-safe; missing inputs stay missing and are imputed inside each CV fold. <br><br> **Ensembles:** random forest, oblique random forest and AdaBoost. <br><br> **Tuning:** Optuna (TPE) with nested cross-validation for an honest estimate of the tuned model (`src/tuning.py`), search spaces per model in the YAML. New `run_experiments.py`, scoreboard in `results/runs.csv`. | **Nested-CV accuracy of the tuned models** (5 folds, 30 Optuna trials, 6 engineered features; mean ± std; untuned in brackets; no-model floor 0.549): <br>AdaBoost 0.677 ± 0.011 (0.672) <br> Oblique random forest 0.677 ± 0.014 (0.640) <br> Random forest 0.675 ± 0.014 (0.638) <br> Logistic regression 0.674 ± 0.014 (0.678) <br> Decision tree 0.664 ± 0.014 (0.575) <br> Naive Bayes 0.658 ± 0.014 (0.654) <br><br>**1. No clear winner.** Every model beats the floor by 0.11–0.13, but the top four are within 0.003 of each other, far less than the fold-to-fold std (0.011–0.014), so the ranking is not reliable. The ensembles do not clearly beat the simpler logistic regression. AdaBoost (best mean, lowest std) and logistic regression (simplest, already 0.678 untuned) are the sensible candidates. <br><br>**2. Tuning helps the high-variance models.** Decision tree +0.089, oblique random forest +0.037, random forest +0.037. Tuning chose shallow trees (depth 2–8) with large leaves, which cut the train–validation gap from +0.116 / +0.136 / +0.110 to +0.016 / +0.010 / +0.011, so the overfitting seen in earlier weeks is largely removed. Already-stable models gained almost nothing (AdaBoost +0.005, Naive Bayes +0.004), and logistic regression even dropped slightly (0.678 → 0.674), which is within noise. The optimism check is small (+0.001 to +0.009, largest for the decision tree), so the nested estimates can be trusted. <br><br>**3. A plateau around 0.68.** Very different model families end at the same accuracy, which suggests the limit comes from the information in the features, not the model. The six engineered features were used in every run, so their separate effect was not measured (a features-off comparison is still to do); logistic regression is at 0.678 against 0.672 in week 4, a difference smaller than one std. <br><br>**4. Fairness.** The false-positive rate for African-American vs Caucasian defendants is 0.28–0.35 vs 0.14–0.22 for our models (Naive Bayes 0.16 vs 0.07), against 0.45 vs 0.23 for COMPAS. Our rates are lower, partly because the models predict recidivism less often (recall of class 1: 0.38–0.61), but the gap between the two groups stays at roughly 1.6–2.3 times, as for COMPAS, even though race is not a model input (other features such as prior counts act as proxies). Asian and Native American groups are too small (n=21 and n=6) to interpret. <br><br>The locked test set has still not been evaluated. Next step: metrics beyond accuracy (precision, recall, AUC) to compare models that accuracy cannot separate. |
| 7 | **Boosting:** `gradient_boosting` (scikit-learn `GradientBoostingClassifier`) and `hist_gradient_boosting` (`HistGradientBoostingClassifier`) registered in `src/model.py`, each with an Optuna search space in the YAML (number of rounds, learning rate, tree size, and subsampling / L2 regularisation), next to AdaBoost from week 5. **New metrics:** every run now reports accuracy, balanced accuracy, precision, recall, F1, F-beta, ROC-AUC and PR-AUC. Because a false positive (predicting that someone will reoffend when they will not) is much more costly than a false negative, the **selection metric is `fbeta_0.5`**: F-beta with beta < 1 weights precision more than recall (beta = 0.5: twice as much), so tuning and the scoreboard favour models that flag fewer people wrongly. F1 is reported as the balanced reference, and ROC-AUC / PR-AUC as threshold-free checks (they do not depend on the 0.5 cut-off). The six engineered features and the per-model `results/<model>/` folders are kept. | **Nested-CV F-beta 0.5 (tuned, 30 trials; floor 0.000):** gradient boosting 0.647, oblique forest 0.647, AdaBoost 0.646, logistic regression 0.642, HistGradientBoosting 0.640, random forest 0.639, decision tree 0.628, naive Bayes 0.611 (not tuned). <br><br>**Boosting:** gradient boosting ranks first (best ROC-AUC, 0.728) but is not significantly better than the week 5 models (paired test, p = 0.27-0.95); it only clearly beats the decision tree and the untuned naive Bayes. <br><br>**Tuning on F-beta 0.5 instead of accuracy:** accuracy unchanged (-0.002 to +0.001) and the same hyperparameters for 4 of 5 models; false positives fell slightly (0-8%), true positives by 0-4%. <br><br>**Fairness:** no improvement. The false-positive rate is still about 1.7-1.9 times higher for African-American than for Caucasian defendants (e.g. 0.28 vs 0.15 for gradient boosting; COMPAS 0.45 vs 0.23). <br><br>Caveat: HistGradientBoosting used `class_weight: balanced`; naive Bayes was not tuned. |
| 8 | **XGBoost + feature selection:** `xgboost` (`XGBClassifier`) registered in `src/model.py` with an Optuna search space in the YAML (no `scale_pos_weight`, it would add false positives). New `VoteSelector` step between preprocessing and model, switched with `feature_selection.enabled`: random forest, gradient boosting and an L1 logistic regression vote, features with at least `min_votes` votes are kept; fitted on training rows only, `min_votes` tuned with the model. | **XGBoost, F0.5 (accuracy), nested CV:** untuned 0.642 (0.675), tuned 0.645 (0.677), tuned + feature selection 0.647 (0.679, kept 5 of 16 features). Tuning and selection each add about +0.003, but paired tests give p = 0.69 and 0.71 and the gaps are smaller than the fold std (~0.015), so neither is a real gain. Selection does match the full model with a third of the features (age, priors_count, log_priors, priors_per_adult_year, age_cat). False positive rate African-American / Caucasian: 0.30 / 0.18 untuned, 0.30 / 0.15 tuned, 0.28 / 0.15 with selection, so fairness barely moves (COMPAS: 0.45 / 0.23). Other models, selection with min_votes=2 and no tuning, vs their untuned score: GB 0.639 (0.643), AdaBoost 0.637 (0.639), LR 0.637 (0.644), RF 0.595 (0.599), all slightly lower, within noise. |
| 9 | **Beyond COMPAS (option A, Adult income):** the pipeline now also runs on a second dataset, the Adult income data, with its own `config_adult.yaml` (`python main.py config_adult.yaml`). Plain `python main.py` still runs COMPAS and gives the same numbers as before (checked against the week 8 run). I fixed the parts of the code that only worked for COMPAS, and added `src/predict.py`, which predicts the 2,000 new people in `adult_new.csv`. | XGBoost with fixed parameters reaches all three targets on Adult: ROC-AUC 0.912, accuracy 0.863, F1 0.681 (gradient boosting with defaults is very close: 0.912 / 0.861 / 0.675). Tuning did not help either model, and tuned gradient boosting misses the accuracy target by 0.001. Feature selection keeps 8 of the 36 features and drops all 25 anonymous `v` columns. More in "Beyond COMPAS" below. |
| 10 | **Missing values:** the numeric imputer is now a setting, `preprocessing.imputation.numeric_strategy` in the YAML (`median`, `mean`, `knn`, `iterative` = MICE, `random_forest` = missForest). The numeric columns are scaled first and then imputed, so KNN compares columns on one scale. All of them learn from the features only, never from the target.<br><br>**Saving and predicting:** `main.py` saves the final fitted pipeline to `models/model.joblib`, and the new `predict.py` loads it and predicts a CSV from the terminal (one row per input row). | On COMPAS no imputer is significantly better than the median (best: iterative, F0.5 0.647 vs 0.642, p = 0.42), so I keep `median`. `predict.py` writes 20 predictions for `compas_new_people.csv`. Details in "Missing values" and "Predicting new data" below. |

## Environment setup

You only need to do this once per machine.

### macOS / Linux
```bash
python3 -m venv venv                 # creates an isolated Python environment in a folder called "venv"
source venv/bin/activate             # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**macOS / Linux**
```bash
source venv/bin/activate
python main.py
```

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, loads and preprocesses the data, trains the model, and prints:
- **train accuracy and test accuracy, side by side.** Comparing the two is how you catch overfitting: if the model looks much better on the data it was trained on than on data it's never seen, it has memorised rather than learned something that generalises. 
- a classification report on the test set
- a false-positive-rate-by-race comparison between our model and
  COMPAS's own score

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Beyond COMPAS

For week 9 I picked option A: run the pipeline on the Adult income data, where the question is whether someone earns more than 50,000 USD a year. There are 6,000 labelled people (24% earn more) and 2,000 new people to predict.

**How to run**
```
python main.py                     # COMPAS, as before
python main.py config_adult.yaml   # Adult; reports in results_adult/<model>/, predictions next to them
```
In `config_adult.yaml`, `tuning.enabled` and `feature_selection.enabled` switch tuning and feature selection on and off. I tested with tuning off and did the final runs with it on.

**Choices for Adult**
- The audited attribute is `sex`. `race` is not used as a feature either, like in the COMPAS runs (delete it from `data.drop_columns` to allow it).
- `fnlwgt` is dropped because it is a census sampling weight, not a trait of the person, and `education` is dropped because it says the same as `education_num`.
- The 25 anonymous columns `v01`-`v25` are all kept, and feature selection decides what to do with them.
- `workclass` and `occupation` are missing for the same people, who earn more than 50K only about half as often, so being missing carries information. That is why `workclass` gets a missingness flag.
- The metric used for tuning and ranking is F1.

**COMPAS still gives the same results.** After these changes I ran `python main.py` again on COMPAS with XGBoost (no tuning, no feature selection) and compared it with the same run from week 8: F0.5 0.642 ± 0.017, accuracy 0.675, recall 0.561 and ROC-AUC 0.723 are identical in both.

**What broke, and how I fixed it**

| What broke | Why | How it was fixed |
|------------|-----|------------------|
| `?` stayed as a category in `workclass`, `occupation` and `native_country` | The cleaning only turned placeholders into missing values in columns that had category rules (`canonical_categories`), and those were all COMPAS columns. | New config list `diagnostics.placeholder_columns`. Now `?` becomes a real missing value in all three (379 / 381 / 86 rows, as in the data card) and no `?` category is left. |
| The fairness report would have stopped with `KeyError: score_text` | It compared against COMPAS's own score and talked about people who "did not reoffend". | The wording and the benchmark now come from `data.fairness` in the config, and the benchmark is optional. |
| `split_features_target` also assumed `score_text` | The extra column kept for the fairness report was hard-coded. | It now uses `data.fairness.benchmark_column`. |
| No way to predict people without a target | The pipeline only evaluated models. | `src/predict.py` cleans the new file the same way, never drops a row (2,000 in, 2,000 out), and saves id, probability and prediction. |
| A column missing from the feature lists is silently ignored | `ColumnTransformer` drops whatever it is not told about. | Each run now prints the columns that are not used. |
| (avoided) The `age` rule `min: 18` would have erased the 74 people aged 17 | It is a COMPAS rule. | Adult has its own rules in its config. |

**Results** (5-fold CV on the 4,800 development rows, mean ± std; tuned rows are nested CV with 30 Optuna trials; feature selection is on in every run)

| Run | Tuned | ROC-AUC | Accuracy | F1 | Hyperparameters |
|-----|-------|---------|----------|----|-----------------|
| No model (always "<= 50K") | | 0.50 | 0.76 | 0.00 | |
| Gradient boosting | no | 0.912 ± 0.016 | 0.861 ± 0.011 | 0.675 ± 0.035 | scikit-learn defaults |
| Gradient boosting | yes | 0.907 ± 0.016 | 0.854 ± 0.011 | 0.661 ± 0.032 | tuned (`min_votes` 1) |
| **XGBoost** | **no** | **0.912 ± 0.017** | **0.863 ± 0.013** | **0.681 ± 0.036** | fixed: 200 trees, learning rate 0.1, depth 3 |
| XGBoost | yes | 0.908 ± 0.016 | 0.855 ± 0.010 | 0.675 ± 0.033 | tuned (`min_votes` 2) |

The "no model" row is the baseline from the assignment (a model that always answers "<= 50K": 76% of people earn 50K or less); my own runs confirm its F1 of 0.00 on the same folds.

The targets are ROC-AUC >= 0.90, accuracy >= 0.855 and F1 >= 0.66, all in the same run. XGBoost with fixed parameters reaches all three with the best margins, and so does gradient boosting with defaults. Tuned XGBoost is right at the limit on accuracy (0.855), and tuned gradient boosting misses accuracy by 0.001.

**Features kept.** Every run keeps the same 8 of the 36 features: `age`, `education_num`, `capital_gain`, `capital_loss`, `hours_per_week`, `marital_status`, `occupation` and `relationship`. None of the `v01`-`v25` columns survived, and neither did the `workclass` missingness flag.

**Fairness.** I looked at the false positive rate by `sex`, meaning people who earn 50K or less but are predicted to earn more. It is 0.02-0.03 for women and 0.08-0.10 for men in all four runs, so men are wrongly flagged as high earners about three times as often as women.

**Predictions.** `adult_predictions.csv` has id, probability and a 0/1 prediction for the 2,000 people in `adult_new.csv`. It comes from the XGBoost run with fixed parameters.

**What was the hardest assumption to find?** The `?` placeholders, because nothing crashed: the cleaning only handled placeholders in columns that had category rules, so `?` would have quietly stayed a category. The targets did not need tuning, since plain boosted trees with feature selection already reach them, and selection threw out all 25 anonymous columns. Tuning gave no gain: in corrected paired tests on the folds, tuned vs fixed gave p = 0.37 for gradient boosting and 0.51 for XGBoost on F1, and XGBoost vs gradient boosting gave p = 0.25, so the differences between the four runs are within the fold-to-fold noise. The tuned gradient boosting was also a bit optimistic (F1 0.678 inside the folds vs 0.661 outside).

## Missing values

The imputer is chosen in the YAML (`preprocessing.imputation.numeric_strategy`). I compared the four options on COMPAS with XGBoost (fixed parameters, no tuning, no feature selection) on the same folds, using the corrected paired test against the median.

| Imputer | F0.5 (mean ± std) | Difference vs median | p-value vs median |
|---|---|---|---|
| `median` (baseline) | 0.642 ± 0.017 | | |
| `knn` | 0.642 ± 0.017 | 0.000 | 1.00 |
| `iterative` (MICE) | 0.647 ± 0.016 | +0.005 | 0.42 |
| `random_forest` (missForest) | 0.639 ± 0.017 | -0.003 | 0.23 |

None of the imputers is significantly different from the median: the biggest gap (iterative, +0.005) is smaller than the fold-to-fold std (about 0.017) and its p-value is 0.42. So I keep `median`: it is the simplest, the fastest, and the fancier imputers give no measurable gain here, probably because the gaps are small (at most about 6% of rows, in `priors_count`) and the model barely depends on the exact filled value.

Changing the order to "scale, then impute" did not change the median baseline (same scores to three decimals), so the earlier runs stay comparable. In the scoreboard, runs from before this week show `median_old` in the `imputer` column.

## Predicting new data

```
python main.py                                                 # trains, evaluates, saves models/model.joblib
python predict.py data/compas_new_people.csv predictions.csv   # predicts, no training
```
For the Adult config: `python main.py config_adult.yaml`, then `python predict.py data/adult_new.csv adult_predictions.csv --config config_adult.yaml`.

`predictions.csv` has one row per person, with three columns: `id`, `probability` (of the positive class, here reoffending within two years) and `prediction` (1 if the probability is at least 0.5). For `compas_new_people.csv` it has 20 rows. The file is not clean on purpose, and nobody is dropped: person 900004 has `priors_count = "?"` and 900008 has no age (the cleaning makes them gaps and the saved imputer fills them), 900012 has a charge degree never seen in training (`"X"`, which the encoders accept), and 900015 and 900016 are the same person twice (both are kept, with the same prediction). The model file is not tracked in git, so `main.py` has to run first.

## Dataset

See `data/README.md`.