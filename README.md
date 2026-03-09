# 🩺 Sepsis RL Policies, Hybrid Evaluation, and OPE

This repository contains:

* preprocessing utilities for MIMIC-III sepsis trajectories
* expert policy training and evaluation scripts for physician, CQL, DDDQN, model-based, MoE, and hybrid policies
* off-policy evaluation using PHWIS, PHWDR, and AM
* result-generation utilities such as action heatmaps and reporting scripts

---

## 📌 Overview

The repository is designed to support an end-to-end offline RL workflow for sepsis treatment, including:

* cohort extraction and preprocessing
* state and action construction
* training of multiple expert policies
* hybrid policy training and evaluation
* off-policy evaluation and downstream analysis

The underlying **MIMIC-III dataset is not included** in this repository. Access must be requested individually by each user. Once access has been obtained, the preprocessing workflow provided here can be applied to the raw data.

The preprocessing pipeline was taken from and slightly adapted based on the preprocessing approach of **Matthieu Komorowski et al.**

---

## 📁 Repository layout

* `preprocessing/` — cohort extraction and dataset construction

  * `preprocess.py` — SQL extraction and export helpers for MIMIC-III
  * `sepsis_cohort.py` — trajectory construction, feature engineering, normalization, and final exports
  * `cohort_flow_report.py` — staged cohort statistics reporting

* `expert/` — policy training and evaluation

  * `*_train.py` — training scripts
  * `*_eval.py` — evaluation scripts
  * `hybrid_train.py` — hybrid model-based + CQL training
  * `hybrid_eval.py` — standalone hybrid evaluation and OPE
  * `policy_runner.py` — config-based policy output generation

* `ope/` — OPE estimators and runner scripts

  * `phwis.py`
  * `phwdr.py`
  * `am.py`
  * `run_ope.py`

* `results/` — plotting and reporting scripts

  * e.g. action heatmaps and result summaries

* `data/` — expected CSV inputs

  * train, validation, and test splits
  * feature-list files

* `final_config.yaml` — main configuration file for config-based runs

---

## ⚙️ Environment

Use **Python 3.10+**. Python 3.11 and 3.12 may also work if all dependencies are compatible.

Typical dependencies used across the scripts include:

* `numpy`
* `pandas`
* `scipy`
* `pyyaml`
* `tensorflow`
* `fancyimpute`
* `pyprind`

If you are running on Colab or a GPU machine, ensure that:

* TensorFlow is installed in a compatible version
* GPU support is configured correctly
* all data paths are set consistently

---

## 🗂️ Data and configuration

Primary paths are defined in `final_config.yaml`:

* `paths.train_csv`
* `paths.val_csv`
* `paths.test_csv`
* `paths.state_features`
* `paths.results_dir`

Default example paths point to:

* `data/rl_train_data_final_cont.csv`
* `data/rl_val_data_final_cont.csv`
* `data/rl_test_data_final_cont.csv`

---

## 🧪 Preprocessing workflow

After obtaining access to MIMIC-III separately, the preprocessing code in this repository can be used to construct the final RL-ready datasets.

Typical preprocessing flow:

1. extract the relevant MIMIC-III tables and intermediate files
2. run the SQL/data export helpers
3. construct the sepsis cohort and trajectories
4. engineer and normalize features
5. export the final train, validation, and test CSV files used by the policy scripts

Example entry points:

```bash
python preprocessing/preprocess.py
python preprocessing/sepsis_cohort.py
```

The exact arguments depend on your local MIMIC-III setup and data paths.

---

## 🚀 Core workflows

## A) Train base policies

### CQL

```bash
python expert/cql_train.py \
  --data-dir data \
  --train-csv rl_train_data_final_cont.csv \
  --val-csv rl_val_data_final_cont.csv \
  --output-dir outputs/cql
```

### DDDQN

```bash
python expert/dddqn_train.py \
  --data-dir data \
  --train-csv rl_train_data_final_cont.csv \
  --val-csv rl_val_data_final_cont.csv \
  --output-dir outputs/dddqn
```

### Model-based

```bash
python expert/mb_train.py \
  --data-dir data \
  --train-csv rl_train_data_final_cont.csv \
  --val-csv rl_val_data_final_cont.csv \
  --output-dir outputs/mb
```

### Mixture-of-Experts (MoE)

```bash
python expert/moe_train.py \
  --data-dir data \
  --train-csv rl_train_data_final_cont.csv \
  --val-csv rl_val_data_final_cont.csv \
  --output-dir outputs/moe
```

### Physician policy

The physician policy is typically derived from clinician behavior in the data and is usually evaluated rather than trained as a learned model.

```bash
python expert/physician_eval.py \
  --data-dir data \
  --test-csv rl_test_data_final_cont.csv
```

---

## B) Evaluate base policies

### CQL

```bash
python expert/cql_eval.py \
  --data-dir data \
  --test-csv rl_test_data_final_cont.csv \
  --weights outputs/cql/model.weights.h5
```

### DDDQN

```bash
python expert/dddqn_eval.py \
  --data-dir data \
  --test-csv rl_test_data_final_cont.csv \
  --weights outputs/dddqn/model.weights.h5
```

### Model-based

```bash
python expert/mb_eval.py \
  --data-dir data \
  --test-csv rl_test_data_final_cont.csv \
  --model-dir outputs/mb
```

### Mixture-of-Experts (MoE)

```bash
python expert/moe_eval.py \
  --data-dir data \
  --test-csv rl_test_data_final_cont.csv \
  --model-dir outputs/moe
```

---

## C) Train the hybrid policy

Run `hybrid_train.py` **without** `--config` for the actual hybrid training path:

```bash
python expert/hybrid_train.py \
  --data-dir data \
  --cql-weights outputs/cql/model.weights.h5 \
  --mb-dir outputs/mb \
  --output-dir outputs/hybrid \
  --use-pretrained-mb \
  --ppo-steps 4000
```

This performs:

* PPO updates
* gate fitting
* export of hybrid artifacts such as dynamics models, PPO weights, and gating JSON files

---

## D) Evaluate the hybrid policy directly

```bash
python expert/hybrid_eval.py \
  --data-dir data \
  --train-csv rl_train_data_final_cont.csv \
  --test-csv rl_test_data_final_cont.csv \
  --cql-weights outputs/cql/model.weights.h5 \
  --hybrid-dir outputs/hybrid
```

---

## E) Generate policy outputs in config mode

```bash
python expert/hybrid_train.py --config final_config.yaml --train_or_load --eval_split test
```

This route enters `policy_runner.py` and writes policy-output `.npz` files for downstream OPE.

---

## 🔀 Important note on hybrid mode

`hybrid_train.py` supports two execution modes:

* if `--config` is provided, the script delegates to `policy_runner.py` and exits
* if `--config` is **not** provided, it runs the full hybrid training loop

Current repository behavior:

* config-based `hybrid` output generation uses real hybrid inference with CQL, model-based value estimates, and gating
* it does **not** use a placeholder hybrid policy

---

## 📊 OPE usage and interpretation

Main OPE estimators:

* **PHWIS**
* **PHWDR**
* **AM** (Direct Method)

Run all configured OPE methods with:

```bash
python ope/run_ope.py --config final_config.yaml --split test
```

### Stability notes

* PHWIS and PHWDR can disagree with AM when support mismatch is severe
* the current behavior policy in PHWIS and PHWDR uses global action frequencies, which can be brittle under covariate shift
* numerical stability for importance-weight normalization has been improved through log-space normalization

If OPE appears unstable, consider:

* increasing `--ope-epsilon` in hybrid evaluation
* enabling `--mask-actions`
* reducing gate sharpness
* reducing model-based usage rate

---

## 🛠️ Practical troubleshooting

* **TensorFlow startup warnings** about cuDNN or cuBLAS registration are common in some notebook and runtime setups; check whether metrics and output files are still produced
* if hybrid evaluation returns contradictory AM versus PHWIS or PHWDR, inspect support overlap before drawing ranking conclusions
* for reproducibility, pin package versions and store all model artifacts in `outputs/*` together with the exact config used
* if paths fail, first verify `final_config.yaml` and the CSV filenames in `data/`

---

## ✅ Suggested run order

1. obtain access to MIMIC-III independently
2. run preprocessing and construct the final RL datasets
3. verify paths in `final_config.yaml`
4. train or load the base experts such as CQL and model-based policies
5. train the hybrid policy in non-config mode
6. generate policy outputs in config mode
7. run OPE and safety checks
8. generate result plots such as action heatmaps

---

## ℹ️ Notes

* this repository does **not** distribute MIMIC-III data
* users are responsible for obtaining dataset access themselves
* preprocessing code was adapted from prior sepsis preprocessing work and adjusted for this repository's workflow
* exact script arguments may vary slightly depending on your local file structure and configuration
