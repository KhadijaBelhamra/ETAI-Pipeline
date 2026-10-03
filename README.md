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
├── main.py                # entry point: run the whole pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── preprocessing.py    # cleaning + train/test split
│   ├── model.py             # model construction
│   ├── evaluate.py         # accuracy metrics + fairness check
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Added to the pipeline | Comments |
|------|------------------------|----------|
| 2 | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` | We have: <br> Logistic reg train accuracy: 0.680 <br> Logistic reg test accuracy:  0.677 <br> Gap (train - test): +0.002 <br><br> Tree train accuracy: 0.829 <br> Tree test accuracy:  0.626 <br> Gap (train - test): +0.203 <br> <br> The logistic regression has a better test score than the tree model. We should also note that the decision tree is clearly overfitting with a gap of over 0.2 units between the train and test accuracy. The decision tree needs to be parametrized to reduce overfitting and improve its performance on unseen data.|
| 3 | Added a configuration-driven `clean_dataset()` function to the pipeline, applied before the existing baseline preprocessing stage. The function standardizes categorical values, converts placeholder tokens and domain-rule violations into missing values, removes exact duplicate rows and repeated IDs, and drops redundant columns. The existing `preprocess()` function is retained for feature/target separation and the stratified train/test split. No new imputation, encoding, or other feature transformations are introduced this week. | Cleaning had different effects across models. Logistic regression showed a small decrease in test accuracy (0.677 → 0.669). The decision tree improved on the test set (0.626 → 0.650) while its train–test gap decreased substantially (0.203 → 0.145), indicating reduced overfitting after cleaning.|
| 4 | Reworked the evaluation pipeline around a configuration-driven scikit-learn Pipeline, with preprocessing fitted inside each cross-validation fold. Added median/mode imputation, robust scaling, configurable categorical encoding with target encoding, and missingness indicators. The dataset is now split into a development set and a locked test set, with 5-fold stratified cross-validation performed only on the development set. Added out-of-fold predictions for the classification report and fairness analysis, refit the final model on the full development set, and added configurable model selection with a Dummy classifier, logistic regression, and Random Forest. Results are automatically saved in separate model-specific folders under results/. | The Dummy classifier provides a baseline accuracy of 0.549, with no predictive ability for class 1. Logistic regression achieved a mean validation accuracy of 0.672 (std = 0.013), with a small mean train–validation gap of +0.003. Random Forest achieved a mean validation accuracy of 0.652 (std = 0.017), with a larger mean train–validation gap of +0.080, indicating more overfitting than logistic regression under the current configuration. The out-of-fold fairness results also show different FPR patterns across races for the three models. The test set remains locked and has not been evaluated.|

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

## Dataset

See `data/README.md`.
