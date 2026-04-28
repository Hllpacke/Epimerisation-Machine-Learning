"""
Reusable utilities for PMX molecular regression workflows.

This module contains the shared chemistry feature generation, scaffold splitting,
model evaluation, applicability-domain and plotting utilities extracted from the
original PMX notebook workflows. It intentionally contains no project-specific
file paths or private data.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    RandomizedSearchCV,
    learning_curve,
)
from sklearn.pipeline import Pipeline

try:
    import shap  # type: ignore

    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback for minimal environments
    tqdm = None


DEFAULT_SMILES_COLUMNS = [
    "SMILES",
    "smiles",
    "Structure",
    "structure",
    "CanonicalSMILES",
    "canonical_smiles",
]

DEFAULT_ID_COLUMNS = [
    "ID",
    "Compound_ID",
    "Compound",
    "Name",
    "Molecule",
    "MolID",
]

DEFAULT_MOTIF_SMARTS = {
    "has_quinoline": "c1ccc2ncccc2c1",
    "has_isoquinoline": "c1ccc2ccncc2c1",
    "has_quinazoline": "c1nc2ccccc2nc1",
}

DESC_FUNCS = {
    "MolWt": Descriptors.MolWt,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumHDonors": Descriptors.NumHDonors,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "RingCount": Descriptors.RingCount,
    "NumAromaticRings": Descriptors.NumAromaticRings,
    "TPSA": Descriptors.TPSA,
    "FractionCSP3": Descriptors.FractionCSP3,
    "MolLogP_Crippen": Crippen.MolLogP,
    "MolMR_Crippen": Crippen.MolMR,
    "NHOHCount": Descriptors.NHOHCount,
    "NOCount": Descriptors.NOCount,
    "LabuteASA": Descriptors.LabuteASA,
}


@dataclass
class RegressionConfig:
    data_csv: str
    target_col: str
    output_dir: str = "results"
    smiles_col: str | None = None
    id_col: str | None = None
    test_size: float = 0.20
    random_seed: int = 42
    morgan_radius: int = 2
    morgan_bits: int = 2048
    cv_folds: int = 5
    n_iter: int = 40
    n_jobs: int = -1
    run_y_randomisation: bool = True
    n_y_randomisation: int = 30
    run_shap: bool = False
    shap_top_k: int = 60
    predict_csv: str | None = None
    predict_output: str | None = None
    add_motif_flags: bool = False


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    return output_dir


def detect_column(
    df: pd.DataFrame,
    explicit_col: str | None,
    candidates: Sequence[str],
    kind: str,
    required: bool = True,
) -> str | None:
    """Return an explicit or auto-detected column name."""
    if explicit_col:
        if explicit_col not in df.columns:
            raise ValueError(
                f"Requested {kind} column '{explicit_col}' was not found. "
                f"Available columns include: {list(df.columns)[:50]}"
            )
        return explicit_col

    detected = next((col for col in candidates if col in df.columns), None)
    if required and detected is None:
        raise ValueError(
            f"Could not find a {kind} column. Looked for: {list(candidates)}. "
            f"Available columns include: {list(df.columns)[:50]}"
        )
    return detected


def mol_from_smiles(smiles: object) -> Chem.Mol | None:
    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None


def prepare_training_dataframe(
    data_csv: str | Path,
    target_col: str,
    smiles_col: str | None = None,
    id_col: str | None = None,
) -> tuple[pd.DataFrame, str, str | None]:
    """Load, clean and RDKit-parse a training CSV."""
    df = pd.read_csv(data_csv)
    detected_smiles = detect_column(
        df, smiles_col, DEFAULT_SMILES_COLUMNS, kind="SMILES", required=True
    )
    detected_id = detect_column(df, id_col, DEFAULT_ID_COLUMNS, kind="ID", required=False)

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' was not found. "
            f"Available columns include: {list(df.columns)[:50]}"
        )

    df = df.copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df[detected_smiles] = df[detected_smiles].astype(str)

    df = df[df[target_col].notna()].copy()
    df = df[df[detected_smiles].notna() & (df[detected_smiles].str.len() > 0)].copy()
    df["Mol"] = df[detected_smiles].apply(mol_from_smiles)
    df = df[df["Mol"].notna()].copy()
    df.reset_index(drop=True, inplace=True)

    if df.empty:
        raise ValueError("No valid rows remained after target, SMILES and RDKit cleaning.")

    return df, detected_smiles, detected_id


def compute_desc(mol: Chem.Mol) -> dict[str, float]:
    return {name: float(func(mol)) for name, func in DESC_FUNCS.items()}


def compute_morgan_bits(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def featurize_mols(
    mols: Sequence[Chem.Mol],
    radius: int = 2,
    n_bits: int = 2048,
    show_progress: bool = True,
) -> pd.DataFrame:
    iterator: Iterable[Chem.Mol]
    if show_progress and tqdm is not None:
        iterator = tqdm(mols, desc="Computing descriptors/fingerprints")
    else:
        iterator = mols

    desc_rows: list[dict[str, float]] = []
    fp_rows: list[np.ndarray] = []
    for mol in iterator:
        desc_rows.append(compute_desc(mol))
        fp_rows.append(compute_morgan_bits(mol, radius=radius, n_bits=n_bits))

    x_desc = pd.DataFrame(desc_rows)
    x_fp = pd.DataFrame(fp_rows, columns=[f"morgan_{i}" for i in range(n_bits)])
    return pd.concat([x_desc, x_fp], axis=1)


def featurize_smiles(
    smiles_list: Sequence[object],
    radius: int = 2,
    n_bits: int = 2048,
) -> tuple[pd.DataFrame, list[Chem.Mol], list[int]]:
    mols: list[Chem.Mol] = []
    valid_idx: list[int] = []
    for i, smiles in enumerate(smiles_list):
        mol = mol_from_smiles(smiles)
        if mol is None:
            continue
        mols.append(mol)
        valid_idx.append(i)

    x_new = featurize_mols(mols, radius=radius, n_bits=n_bits, show_progress=False)
    return x_new, mols, valid_idx


def murcko_smiles(mol: Chem.Mol) -> str:
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, isomericSmiles=False)
    except Exception:
        return "NA"


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def spearman_r(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))


def regression_report(y_true: Sequence[float], y_pred: Sequence[float], label: str) -> dict[str, float | int | str]:
    return {
        "Split": label,
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Spearman": spearman_r(y_true, y_pred),
        "n": int(len(y_true)),
    }


def build_rf_pipeline(random_seed: int = 42, n_jobs: int = -1, **rf_params) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("var", VarianceThreshold(0.0)),
            (
                "rf",
                RandomForestRegressor(
                    random_state=random_seed,
                    n_jobs=n_jobs,
                    **rf_params,
                ),
            ),
        ]
    )


def default_rf_param_dist() -> dict[str, list[object]]:
    return {
        "rf__n_estimators": [600, 900, 1200, 1600],
        "rf__max_depth": [None, 10, 14, 18, 26],
        "rf__min_samples_split": [2, 5, 10],
        "rf__min_samples_leaf": [1, 2, 4, 8],
        "rf__max_features": ["sqrt", 0.2, 0.35, 0.5, 0.8],
        "rf__bootstrap": [True],
    }


def scaffold_split(
    x: pd.DataFrame,
    y: np.ndarray,
    scaffolds: np.ndarray,
    test_size: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    train_idx, test_idx = next(splitter.split(idx, groups=scaffolds))
    return (
        x.iloc[train_idx].copy(),
        x.iloc[test_idx].copy(),
        y[train_idx],
        y[test_idx],
        scaffolds[train_idx],
        scaffolds[test_idx],
        train_idx,
        test_idx,
    )


def make_group_cv(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    train_groups: np.ndarray,
    requested_folds: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    n_groups = len(set(train_groups))
    n_splits = min(requested_folds, n_groups)
    if n_splits < 2:
        raise ValueError(
            "At least two unique training scaffolds are required for GroupKFold. "
            "Use a larger dataset or reduce the scaffold holdout constraint."
        )
    group_kfold = GroupKFold(n_splits=n_splits)
    return list(group_kfold.split(x_train, y_train, groups=train_groups))


def train_baselines(
    x_desc: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    random_seed: int = 42,
) -> pd.DataFrame:
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(x_desc.iloc[train_idx], y_train)

    ridge = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("var", VarianceThreshold(0.0)),
            ("model", Ridge(alpha=1.0, random_state=random_seed)),
        ]
    )
    ridge.fit(x_desc.iloc[train_idx], y_train)

    return pd.DataFrame(
        [
            regression_report(y_train, dummy.predict(x_desc.iloc[train_idx]), "Train (Dummy)"),
            regression_report(y_test, dummy.predict(x_desc.iloc[test_idx]), "External (Dummy)"),
            regression_report(y_train, ridge.predict(x_desc.iloc[train_idx]), "Train (Ridge, descriptors)"),
            regression_report(y_test, ridge.predict(x_desc.iloc[test_idx]), "External (Ridge, descriptors)"),
        ]
    )


def tune_random_forest(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    cv_iter: list[tuple[np.ndarray, np.ndarray]],
    random_seed: int = 42,
    n_iter: int = 40,
    n_jobs: int = -1,
    verbose: int = 1,
) -> RandomizedSearchCV:
    rf_pipe = build_rf_pipeline(random_seed=random_seed, n_jobs=n_jobs)
    search = RandomizedSearchCV(
        rf_pipe,
        param_distributions=default_rf_param_dist(),
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv_iter,
        random_state=random_seed,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    try:
        search.fit(x_train, y_train)
    except Exception:
        search = RandomizedSearchCV(
            rf_pipe,
            param_distributions=default_rf_param_dist(),
            n_iter=n_iter,
            scoring="neg_mean_squared_error",
            cv=cv_iter,
            random_state=random_seed,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        search.fit(x_train, y_train)
    return search


def save_target_distribution(y: np.ndarray, target_col: str, output_dir: Path) -> None:
    plt.figure()
    plt.hist(y, bins=30)
    plt.xlabel(target_col)
    plt.ylabel("Count")
    plt.title("Target distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "target_distribution.png", dpi=300)
    plt.close()


def save_parity_plot(
    y_train: np.ndarray,
    pred_train: np.ndarray,
    y_test: np.ndarray,
    pred_test: np.ndarray,
    target_col: str,
    output_dir: Path,
) -> None:
    plt.figure()
    plt.scatter(y_train, pred_train, alpha=0.6, label="Train")
    plt.scatter(y_test, pred_test, alpha=0.8, label="External test")
    mn = min(np.min(y_train), np.min(y_test), np.min(pred_train), np.min(pred_test))
    mx = max(np.max(y_train), np.max(y_test), np.max(pred_train), np.max(pred_test))
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel(f"Experimental {target_col}")
    plt.ylabel(f"Predicted {target_col}")
    plt.title("Parity plot")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "parity_plot.png", dpi=300)
    plt.close()


def save_residual_plots(
    y_train: np.ndarray,
    pred_train: np.ndarray,
    y_test: np.ndarray,
    pred_test: np.ndarray,
    target_col: str,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    res_train = pred_train - y_train
    res_test = pred_test - y_test

    plt.figure()
    plt.scatter(pred_train, res_train, alpha=0.6, label="Train")
    plt.scatter(pred_test, res_test, alpha=0.8, label="External test")
    plt.axhline(0, linestyle="--")
    plt.xlabel(f"Predicted {target_col}")
    plt.ylabel("Residual (predicted - experimental)")
    plt.title("Residuals vs predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "residuals_vs_predicted.png", dpi=300)
    plt.close()

    plt.figure()
    plt.hist(res_train, bins=25, alpha=0.6, label="Train")
    plt.hist(res_test, bins=25, alpha=0.8, label="External test")
    plt.axvline(0, linestyle="--")
    plt.xlabel("Residual (predicted - experimental)")
    plt.ylabel("Count")
    plt.title("Residual distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "residual_distribution.png", dpi=300)
    plt.close()

    plt.figure()
    plt.scatter(y_train, np.abs(res_train), alpha=0.6, label="Train")
    plt.scatter(y_test, np.abs(res_test), alpha=0.8, label="External test")
    plt.xlabel(f"Experimental {target_col}")
    plt.ylabel("Absolute residual")
    plt.title("Absolute error vs experimental")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "absolute_error_vs_experimental.png", dpi=300)
    plt.close()

    return res_train, res_test


def save_learning_curve(
    model: Pipeline,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    cv_iter: list[tuple[np.ndarray, np.ndarray]],
    target_col: str,
    output_dir: Path,
    n_jobs: int = -1,
) -> None:
    train_sizes, train_scores, test_scores = learning_curve(
        model,
        x_train,
        y_train,
        cv=cv_iter,
        scoring="neg_mean_squared_error",
        train_sizes=np.linspace(0.2, 1.0, 6),
        n_jobs=n_jobs,
    )

    train_rmse = np.sqrt(-train_scores.mean(axis=1))
    test_rmse = np.sqrt(-test_scores.mean(axis=1))
    curve_df = pd.DataFrame(
        {
            "train_size": train_sizes,
            "cv_train_rmse": train_rmse,
            "cv_validation_rmse": test_rmse,
        }
    )
    curve_df.to_csv(output_dir / "learning_curve.csv", index=False)

    plt.figure()
    plt.plot(train_sizes, train_rmse, marker="o", label="CV train RMSE")
    plt.plot(train_sizes, test_rmse, marker="o", label="CV validation RMSE")
    plt.xlabel("Training set size")
    plt.ylabel(f"RMSE ({target_col})")
    plt.title("Learning curve (scaffold GroupKFold)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "learning_curve.png", dpi=300)
    plt.close()


def run_permutation_importance(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    output_dir: Path,
    random_seed: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
    perm = permutation_importance(
        model,
        x_test,
        y_test,
        scoring="neg_mean_squared_error",
        n_repeats=10,
        random_state=random_seed,
        n_jobs=n_jobs,
    )
    perm_df = pd.DataFrame(
        {
            "feature": x_test.columns,
            "perm_mean": perm.importances_mean,
            "perm_std": perm.importances_std,
        }
    ).sort_values("perm_mean", ascending=False)
    perm_df.to_csv(output_dir / "permutation_importance.csv", index=False)

    top = perm_df.head(25).iloc[::-1]
    plt.figure(figsize=(8, 7))
    plt.barh(top["feature"], top["perm_mean"])
    plt.xlabel("Permutation importance (ΔMSE)")
    plt.title("Top 25 permutation importances (external test)")
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "permutation_importance.png", dpi=300)
    plt.close()
    return perm_df


def morgan_fp_from_smiles(smiles: object, radius: int = 2, n_bits: int = 2048):
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def sim5_to_training(
    query_smiles: Sequence[object],
    train_smiles: Sequence[object],
    radius: int = 2,
    n_bits: int = 2048,
    k: int = 5,
) -> np.ndarray:
    fps_train = [morgan_fp_from_smiles(s, radius, n_bits) for s in train_smiles]
    fps_train = [fp for fp in fps_train if fp is not None]
    out: list[float] = []

    for smiles in query_smiles:
        fp_query = morgan_fp_from_smiles(smiles, radius, n_bits)
        if fp_query is None or len(fps_train) == 0:
            out.append(np.nan)
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp_query, fps_train)
        sims_sorted = np.sort(np.array(sims))[::-1]
        out.append(float(np.mean(sims_sorted[:k])) if len(sims_sorted) >= k else float(np.mean(sims_sorted)))
    return np.array(out, dtype=float)


def save_applicability_domain_plot(
    train_sim5: np.ndarray,
    test_sim5: np.ndarray,
    output_dir: Path,
) -> None:
    pd.DataFrame({"SIM5_train": pd.Series(train_sim5), "SIM5_external_test": pd.Series(test_sim5)}).to_csv(
        output_dir / "applicability_domain_sim5.csv", index=False
    )
    plt.figure()
    plt.hist(train_sim5[~np.isnan(train_sim5)], bins=25, alpha=0.6, label="Train vs train")
    plt.hist(test_sim5[~np.isnan(test_sim5)], bins=25, alpha=0.8, label="External test vs train")
    plt.xlabel("SIM_5 (mean Tanimoto to five nearest training neighbours)")
    plt.ylabel("Count")
    plt.title("Applicability domain check")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "applicability_domain_sim5.png", dpi=300)
    plt.close()


def run_y_randomisation(
    best_params: dict[str, object],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    pred_test: np.ndarray,
    output_dir: Path,
    n_perm: int = 30,
    random_seed: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    rf_params = {key.replace("rf__", ""): value for key, value in best_params.items()}
    r2_values: list[float] = []

    for _ in range(n_perm):
        y_shuffled = rng.permutation(y_train)
        model = build_rf_pipeline(random_seed=random_seed, n_jobs=n_jobs, **rf_params)
        model.fit(x_train, y_shuffled)
        r2_values.append(float(r2_score(y_test, model.predict(x_test))))

    r2_rand = np.array(r2_values, dtype=float)
    r2_real = float(r2_score(y_test, pred_test))
    rand_mean = float(np.mean(r2_rand))
    rand_std = float(np.std(r2_rand, ddof=1))
    z_score = (r2_real - rand_mean) / (rand_std + 1e-12)
    empirical_p = float(np.mean(r2_rand >= r2_real))

    summary = pd.DataFrame(
        [
            {
                "R2_real": r2_real,
                "R2_randomised_mean": rand_mean,
                "R2_randomised_std": rand_std,
                "z_score": float(z_score),
                "empirical_p": empirical_p,
                "n_permutations": int(n_perm),
            }
        ]
    )
    pd.DataFrame({"R2_randomised": r2_rand}).to_csv(output_dir / "y_randomisation_values.csv", index=False)
    summary.to_csv(output_dir / "y_randomisation_summary.csv", index=False)

    plt.figure()
    plt.hist(r2_rand, bins=15, alpha=0.85)
    plt.axvline(r2_real, linestyle="--")
    plt.xlabel("R2 under y-randomisation")
    plt.ylabel("Count")
    plt.title("Y-randomisation distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "y_randomisation.png", dpi=300)
    plt.close()
    return summary


def run_shap_summary(
    best_params: dict[str, object],
    permutation_df: pd.DataFrame,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    output_dir: Path,
    random_seed: int = 42,
    n_jobs: int = -1,
    top_k: int = 60,
) -> None:
    if not SHAP_AVAILABLE:
        print("SHAP is not installed; skipping SHAP summary.")
        return

    top_features = permutation_df["feature"].head(top_k).tolist()
    x_top = x_train[top_features].copy()
    rf_params = {key.replace("rf__", ""): value for key, value in best_params.items()}
    explain_model = build_rf_pipeline(random_seed=random_seed, n_jobs=n_jobs, **rf_params)
    explain_model.fit(x_top, y_train)

    x_transformed = explain_model.named_steps["imputer"].transform(x_top)
    x_transformed = explain_model.named_steps["var"].transform(x_transformed)
    rf_model = explain_model.named_steps["rf"]
    explainer = shap.TreeExplainer(rf_model)
    n_shap = min(250, x_transformed.shape[0])
    shap_values = explainer.shap_values(x_transformed[:n_shap])

    plt.figure()
    shap.summary_plot(shap_values, x_transformed[:n_shap], plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "shap_summary_bar.png", dpi=300, bbox_inches="tight")
    plt.close()


def has_substruct(smiles: object, smarts: str) -> bool:
    mol = mol_from_smiles(smiles)
    query = Chem.MolFromSmarts(smarts)
    if mol is None or query is None:
        return False
    return bool(mol.HasSubstructMatch(query))


def add_motif_flags(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    df = df.copy()
    for flag_name, smarts in DEFAULT_MOTIF_SMARTS.items():
        df[flag_name] = df[smiles_col].apply(lambda smiles: has_substruct(smiles, smarts))
    return df


def score_new_molecules(
    model: Pipeline,
    predict_csv: str | Path,
    output_csv: str | Path,
    smiles_col: str | None = None,
    id_col: str | None = None,
    prediction_col: str = "prediction",
    radius: int = 2,
    n_bits: int = 2048,
    add_flags: bool = False,
) -> pd.DataFrame:
    new_df = pd.read_csv(predict_csv)
    detected_smiles = detect_column(
        new_df, smiles_col, DEFAULT_SMILES_COLUMNS, kind="SMILES", required=True
    )
    detected_id = detect_column(new_df, id_col, DEFAULT_ID_COLUMNS, kind="ID", required=False)

    x_new, _mols, valid_idx = featurize_smiles(
        new_df[detected_smiles].astype(str).tolist(), radius=radius, n_bits=n_bits
    )
    scored = new_df.iloc[valid_idx].copy()
    scored[prediction_col] = model.predict(x_new)
    if add_flags:
        scored = add_motif_flags(scored, detected_smiles)

    preferred_cols = [col for col in [detected_id, detected_smiles] if col is not None]
    other_cols = [col for col in scored.columns if col not in preferred_cols]
    scored = scored[preferred_cols + other_cols]
    scored.to_csv(output_csv, index=False)
    return scored


def save_model_and_metadata(
    model: Pipeline,
    output_dir: Path,
    config: RegressionConfig,
    smiles_col: str,
    id_col: str | None,
    best_params: dict[str, object],
    metrics_df: pd.DataFrame,
    model_name: str = "model.joblib",
    metadata_name: str = "metadata.json",
) -> None:
    model_path = output_dir / model_name
    metadata_path = output_dir / metadata_name
    joblib.dump(model, model_path)

    metadata = {
        "created": datetime.now().isoformat(),
        "workflow": "PMX RDKit descriptor + Morgan fingerprint random forest regressor",
        "target_col": config.target_col,
        "smiles_col": smiles_col,
        "id_col": id_col,
        "morgan_radius": config.morgan_radius,
        "morgan_bits": config.morgan_bits,
        "descriptor_names": list(DESC_FUNCS.keys()),
        "best_params": best_params,
        "config": asdict(config),
        "metrics": metrics_df.to_dict(orient="records"),
        "note": "Training data are not included in this repository. Supply a CSV via the CLI.",
    }
    # Avoid storing private absolute paths in metadata.
    metadata["config"]["data_csv"] = "<provided at runtime>"
    if metadata["config"].get("predict_csv"):
        metadata["config"]["predict_csv"] = "<provided at runtime>"

    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
