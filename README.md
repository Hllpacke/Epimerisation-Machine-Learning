# PMX RDKit/Morgan regression scripts

Cleaned, GitHub-ready Python scripts extracted from the PMX potency and LogD notebook workflows. The original notebooks contained local paths and project-specific CSV names; these scripts remove those hard-coded paths and expect data to be supplied at runtime.

## Files

- `pmx_regression_utils.py` — shared RDKit feature generation, scaffold splitting, model training, validation, plotting, y-randomisation, applicability-domain and prediction utilities.
- `train_potency_regressor.py` — trains a corrected pIC50 model.
- `train_logd_regressor.py` — trains a LogD model.
- `requirements.txt` — Python dependencies.

## Expected input format

Provide a CSV with at least:

- one SMILES column, either supplied with `--smiles-col` or auto-detected from common names such as `SMILES`, `smiles`, `CanonicalSMILES`.
- one target column, supplied with `--target-col`.
- optionally one compound ID column, supplied with `--id-col` or auto-detected from common names such as `ID`, `Compound_ID`, `Name`.

Data files are intentionally not included.

## Example: potency model

```bash
python train_potency_regressor.py \
  --data-csv path/to/training_data.csv \
  --target-col "Corrected pIC50(D10)" \
  --smiles-col SMILES \
  --id-col ID \
  --output-dir results/potency
```

To score a new molecule CSV after training:

```bash
python train_potency_regressor.py \
  --data-csv path/to/training_data.csv \
  --target-col "Corrected pIC50(D10)" \
  --predict-csv path/to/new_molecules.csv \
  --predict-output results/potency/new_molecule_predictions.csv \
  --output-dir results/potency
```

## Example: LogD model

```bash
python train_logd_regressor.py \
  --data-csv path/to/training_data.csv \
  --target-col "AZ_LogD74" \
  --smiles-col SMILES \
  --id-col ID \
  --output-dir results/logd
```

## Main outputs

Each workflow writes outputs to `--output-dir`, including:

- trained model: `potency_model.joblib` or `logd_model.joblib`
- metadata JSON with sanitised runtime metadata
- `metrics.csv`
- `baseline_metrics.csv`
- `external_validation_predictions.csv`
- `permutation_importance.csv`
- applicability-domain results
- y-randomisation results, unless `--skip-y-randomisation` is used
- plots under `plots/`

## Notes

The scaffold split uses Murcko scaffolds and `GroupShuffleSplit`, with internal `GroupKFold` for hyperparameter tuning. Features are RDKit descriptors plus Morgan fingerprints. SHAP is optional and only runs when `--run-shap` is passed and `shap` is installed.
