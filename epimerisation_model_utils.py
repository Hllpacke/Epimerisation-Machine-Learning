"""
Shared utilities for epimerisation-risk classification workflows.

The functions here were refactored from exploratory notebooks into reusable,
repository-friendly code. They intentionally contain no local file paths or
project-specific CSV filenames. Provide all datasets through script arguments.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, FindMolChiralCenters

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from scipy.special import ndtr
from scipy.stats import chi2


DEFAULT_METADATA_COLUMNS = [
    "conditions",
    "reaction_class",
    "substrate_class",
    "derivative_notes",
    "specific_reagents",
    "solvent_1",
    "solvent_2",
]
DEFAULT_NUMERIC_METADATA_COLUMNS = ["temperature_C", "reaction_time_h"]


@dataclass
class ModelConfig:
    """Metadata required to reproduce training-time featurisation."""

    model_name: str
    smiles_col: str
    label_col: str
    n_bits: int
    radius: int
    include_metadata: bool
    add_epimerisable_flag: bool
    metadata_columns: list[str]
    numeric_metadata_columns: list[str]
    feature_columns: list[str]
    seed: int
    threshold: float
    corr_threshold: float


class CorrFilter(BaseEstimator, TransformerMixin):
    """Remove highly correlated columns after scaling/imputation.

    Parameters
    ----------
    thresh:
        Absolute correlation threshold above which a later feature is removed.
    feature_names:
        Names corresponding to input columns. These are stored so downstream
        feature importances can be mapped back to chemistry descriptors.
    """

    def __init__(self, thresh: float = 0.90, feature_names: Sequence[str] | None = None):
        self.thresh = thresh
        self.feature_names = None if feature_names is None else list(feature_names)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        if self.feature_names is None:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
        else:
            self.feature_names_ = list(self.feature_names)
        corr = pd.DataFrame(X, columns=self.feature_names_).corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > self.thresh)]
        self.keep_ = [i for i, col in enumerate(self.feature_names_) if col not in to_drop]
        self.dropped_features_ = to_drop
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.keep_]

    @property
    def keep(self) -> list[int]:
        """Backwards-compatible alias used in the original notebooks."""
        return self.keep_


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_csv_with_required_columns(
    csv_path: str | Path,
    required_columns: Sequence[str],
    rename: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Read a CSV and validate that required columns are present."""
    df = pd.read_csv(csv_path)
    if rename:
        df = df.rename(columns=rename)
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {csv_path}: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    return df


def normalise_smiles_column(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    """Return a copy with the requested SMILES column renamed to 'smi_start'."""
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column '{smiles_col}' not found. Columns: {list(df.columns)}")
    out = df.copy()
    if smiles_col != "smi_start":
        out = out.rename(columns={smiles_col: "smi_start"})
    return out


def compute_rdkit_descriptors(smiles: Iterable[str], prefix: str = "rdkit") -> pd.DataFrame:
    """Calculate the full RDKit descriptor list for each SMILES string."""
    names = [f"{prefix}_{name}" for name, _ in Descriptors._descList]
    rows: list[list[float]] = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        if mol is None:
            rows.append([np.nan] * len(names))
            continue
        values = []
        for _, func in Descriptors._descList:
            try:
                values.append(float(func(mol)))
            except Exception:
                values.append(np.nan)
        rows.append(values)
    return pd.DataFrame(rows, columns=names).replace([np.inf, -np.inf], np.nan)


def compute_morgan_fingerprints(
    smiles: Iterable[str],
    radius: int = 2,
    n_bits: int = 1024,
    prefix: str = "morgan",
    include_density: bool = True,
) -> pd.DataFrame:
    """Calculate Morgan bit-vector fingerprints and optional bit density."""
    fps: list[np.ndarray] = []
    densities: list[float] = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        arr = np.zeros((n_bits,), dtype=int)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            DataStructs.ConvertToNumpyArray(fp, arr)
            densities.append(float(arr.sum()) / float(n_bits))
        else:
            arr = np.full((n_bits,), np.nan)
            densities.append(np.nan)
        fps.append(arr)
    columns = [f"{prefix}_bit_{i}" for i in range(n_bits)]
    df_fp = pd.DataFrame(fps, columns=columns)
    if include_density:
        df_fp[f"{prefix}_density"] = densities
    return df_fp


def has_epimerisable_center(smiles: str) -> int:
    """Flag molecules with at least one assigned/unassigned chiral atom bearing H.

    This is a pragmatic screen used in the original novel-compound notebook to
    mask probabilities for molecules without an obvious epimerisable centre.
    It is not a replacement for expert mechanistic annotation.
    """
    mol = Chem.MolFromSmiles(str(smiles)) if pd.notna(smiles) else None
    if mol is None:
        return 0
    for atom_idx, _assignment in FindMolChiralCenters(mol, includeUnassigned=True):
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetTotalNumHs() > 0:
            return 1
    return 0


def build_metadata_features(
    df: pd.DataFrame,
    metadata_columns: Sequence[str] = DEFAULT_METADATA_COLUMNS,
    numeric_metadata_columns: Sequence[str] = DEFAULT_NUMERIC_METADATA_COLUMNS,
    reference_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """One-hot encode categorical reaction metadata and append numeric metadata.

    For prediction, pass `reference_columns` from the training set to align new
    data to exactly the same metadata columns. Missing metadata is filled with
    `Missing` for categorical features and 0 for numeric features.
    """
    tmp = df.copy()
    for col in metadata_columns:
        if col not in tmp.columns:
            tmp[col] = "Missing"
    for col in numeric_metadata_columns:
        if col not in tmp.columns:
            tmp[col] = 0.0

    cat = tmp[list(metadata_columns)].fillna("Missing").astype(str)
    cat_df = pd.get_dummies(cat, drop_first=False)
    num_df = tmp[list(numeric_metadata_columns)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    meta = pd.concat([cat_df, num_df], axis=1)

    if reference_columns is not None:
        return align_columns(meta, list(reference_columns), fill_value=0.0)
    return meta


def make_feature_matrix(
    df: pd.DataFrame,
    n_bits: int = 1024,
    radius: int = 2,
    include_metadata: bool = False,
    add_epimerisable_flag: bool = False,
    metadata_columns: Sequence[str] = DEFAULT_METADATA_COLUMNS,
    numeric_metadata_columns: Sequence[str] = DEFAULT_NUMERIC_METADATA_COLUMNS,
    reference_feature_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build the model feature matrix from a dataframe containing 'smi_start'."""
    if "smi_start" not in df.columns:
        raise ValueError("Feature generation requires a column named 'smi_start'.")

    parts = [
        compute_rdkit_descriptors(df["smi_start"], prefix="rdkit"),
        compute_morgan_fingerprints(df["smi_start"], radius=radius, n_bits=n_bits, prefix="morgan"),
    ]
    if include_metadata:
        parts.append(build_metadata_features(df, metadata_columns, numeric_metadata_columns))
    if add_epimerisable_flag:
        parts.append(df["smi_start"].map(has_epimerisable_center).rename("has_epimerisable_center").to_frame())

    features = pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)
    if reference_feature_columns is not None:
        features = align_columns(features, list(reference_feature_columns), fill_value=0.0)
    return features


def align_columns(df: pd.DataFrame, reference_columns: Sequence[str], fill_value: float = 0.0) -> pd.DataFrame:
    """Add missing columns, drop extras, and order columns to match training."""
    out = df.copy()
    for col in reference_columns:
        if col not in out.columns:
            out[col] = fill_value
    return out[list(reference_columns)]


def clean_feature_matrix(X: pd.DataFrame, y: Sequence[int] | None = None) -> tuple[pd.DataFrame, np.ndarray | None, np.ndarray]:
    """Drop rows with missing descriptors before model fitting."""
    mask = X.notna().all(axis=1).to_numpy()
    X_clean = X.loc[mask].reset_index(drop=True)
    if y is None:
        return X_clean, None, mask
    y_arr = np.asarray(y)[mask]
    return X_clean, y_arr, mask


def make_smote_step(y_train: Sequence[int], seed: int) -> SMOTE | str:
    """Create a SMOTE step that is safe for small minority classes."""
    counts = pd.Series(y_train).value_counts()
    min_count = int(counts.min()) if len(counts) > 1 else 0
    if min_count < 2:
        return "passthrough"
    k_neighbors = min(5, min_count - 1)
    return SMOTE(random_state=seed, k_neighbors=k_neighbors)


def build_classifier_pipeline(
    feature_names: Sequence[str],
    y_train: Sequence[int],
    seed: int = 42,
    corr_threshold: float = 0.90,
) -> ImbPipeline:
    """Build the imputation → scaling → correlation-filter → SMOTE → RF pipeline."""
    return ImbPipeline(
        steps=[
            ("imp", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler()),
            ("corr", CorrFilter(thresh=corr_threshold, feature_names=list(feature_names))),
            ("smote", make_smote_step(y_train, seed=seed)),
            ("clf", RandomForestClassifier(class_weight="balanced", random_state=seed)),
        ]
    )


def default_param_distribution() -> dict[str, list[Any]]:
    return {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [2, 5, 10],
    }


def tune_classifier(
    X_train: pd.DataFrame,
    y_train: Sequence[int],
    seed: int = 42,
    n_iter: int = 20,
    n_splits: int = 5,
    n_jobs: int = -1,
    scoring: str = "precision",
    corr_threshold: float = 0.90,
) -> RandomizedSearchCV:
    """Hyperparameter tune the random-forest classifier pipeline."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pipe = build_classifier_pipeline(X_train.columns, y_train, seed=seed, corr_threshold=corr_threshold)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=default_param_distribution(),
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=seed,
        n_jobs=n_jobs,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    return search


def evaluate_classifier(model: Any, X: pd.DataFrame, y: Sequence[int], threshold: float = 0.50) -> tuple[dict[str, float], pd.DataFrame]:
    """Return core classification metrics and per-row predictions."""
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    metrics = {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "ROC AUC": roc_auc_score(y, proba),
        "PR AUC": average_precision_score(y, proba),
        "MCC": matthews_corrcoef(y, pred),
    }
    predictions = pd.DataFrame({"y_true": y, "y_pred": pred, "y_proba": proba})
    return metrics, predictions


def cross_validated_fold_metrics(
    tuned_model: Any,
    X_train: pd.DataFrame,
    y_train: Sequence[int],
    seed: int = 42,
    n_splits: int = 5,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """Compute per-fold MCC, ROC AUC, and PR AUC using the tuned pipeline."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    records = []
    X_arr = X_train.to_numpy()
    y_arr = np.asarray(y_train)
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_arr, y_arr)):
        fold_model = clone(tuned_model)
        fold_model.fit(X_arr[train_idx], y_arr[train_idx])
        proba = fold_model.predict_proba(X_arr[val_idx])[:, 1]
        pred = (proba >= threshold).astype(int)
        records.append(
            {
                "fold": fold_idx,
                "MCC": matthews_corrcoef(y_arr[val_idx], pred),
                "ROC AUC": roc_auc_score(y_arr[val_idx], proba),
                "PR AUC": average_precision_score(y_arr[val_idx], proba),
            }
        )
    return pd.DataFrame(records)


def y_randomization_cv(
    tuned_model: Any,
    X_train: pd.DataFrame,
    y_train: Sequence[int],
    observed_metrics: Mapping[str, float],
    seed: int = 42,
    n_splits: int = 5,
    n_shuffles: int = 50,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Cross-validated Y-randomisation against shuffled training labels."""
    rng = np.random.default_rng(seed)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scorers = {
        "Accuracy": "accuracy",
        "Precision": "precision",
        "Recall": "recall",
        "F1": "f1",
        "ROC AUC": "roc_auc",
        "MCC": "matthews_corrcoef",
    }
    # sklearn does not expose MCC as a string scorer on older versions.
    from sklearn.metrics import make_scorer

    scorers["MCC"] = make_scorer(matthews_corrcoef)

    random_scores = {metric: [] for metric in scorers}
    y_train = np.asarray(y_train)
    for _ in range(n_shuffles):
        shuffled = rng.permutation(y_train)
        cv_results = cross_validate(
            tuned_model,
            X_train,
            shuffled,
            cv=cv,
            scoring=scorers,
            return_train_score=False,
            n_jobs=n_jobs,
        )
        for metric in random_scores:
            random_scores[metric].append(float(np.mean(cv_results[f"test_{metric}"])))

    rows = []
    for metric, values in random_scores.items():
        arr = np.asarray(values, dtype=float)
        mean = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1))
        observed = float(observed_metrics.get(metric, np.nan))
        z_score = (observed - mean) / sd if sd > 0 else np.nan
        p_value = 1 - ndtr(z_score) if not np.isnan(z_score) else np.nan
        rows.append(
            {
                "Metric": metric,
                "Observed": observed,
                "Random_Mean": mean,
                "Random_SD": sd,
                "Z_score": z_score,
                "p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def get_rf_feature_importance(model: Any) -> pd.DataFrame:
    """Extract random-forest importances after the correlation filter."""
    corr = model.named_steps["corr"]
    final_features = np.asarray(corr.feature_names_)[corr.keep_]
    importances = model.named_steps["clf"].feature_importances_
    return (
        pd.DataFrame({"feature": final_features, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def fit_applicability_domain(
    X_train: pd.DataFrame,
    n_components: int = 10,
    alpha: float = 0.975,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit PCA-Mahalanobis applicability-domain model on training features."""
    scaler = StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)
    n_components = min(n_components, X_scaled.shape[0] - 1, X_scaled.shape[1])
    if n_components < 1:
        raise ValueError("Not enough samples/features to fit applicability-domain PCA.")
    pca = PCA(n_components=n_components, random_state=seed).fit(X_scaled)
    X_pca = pca.transform(X_scaled)
    cov = np.cov(X_pca, rowvar=False)
    inv_cov = np.linalg.pinv(np.atleast_2d(cov))
    mean = X_pca.mean(axis=0)
    threshold = float(chi2.ppf(alpha, df=n_components))
    return {
        "scaler": scaler,
        "pca": pca,
        "inv_cov": inv_cov,
        "mean": mean,
        "threshold": threshold,
        "alpha": alpha,
        "n_components": n_components,
    }


def score_applicability_domain(ad_model: Mapping[str, Any], X: pd.DataFrame) -> pd.DataFrame:
    """Calculate Mahalanobis distances and in-domain flags."""
    X_scaled = ad_model["scaler"].transform(X)
    X_pca = ad_model["pca"].transform(X_scaled)
    mean = ad_model["mean"]
    inv_cov = ad_model["inv_cov"]
    distances = []
    for row in X_pca:
        diff = row - mean
        distances.append(float(diff @ inv_cov @ diff.T))
    distances = np.asarray(distances)
    return pd.DataFrame(
        {
            "mahalanobis_distance": distances,
            "ad_threshold": float(ad_model["threshold"]),
            "in_applicability_domain": distances <= float(ad_model["threshold"]),
        }
    )


def save_model_bundle(model: Any, config: ModelConfig, output_path: str | Path) -> Path:
    """Save a trained sklearn/imblearn pipeline with its featurisation config."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"model": model, "config": asdict(config)}
    joblib.dump(bundle, output_path)
    return output_path


def load_model_bundle(path: str | Path) -> tuple[Any, ModelConfig]:
    bundle = joblib.load(path)
    return bundle["model"], ModelConfig(**bundle["config"])


def write_json(data: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)
    return path


def write_text_report(metrics: Mapping[str, float], y_true: Sequence[int], predictions: pd.DataFrame, output_path: str | Path) -> Path:
    """Save a compact text report with metrics, classification report and CM."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = classification_report(y_true, predictions["y_pred"], zero_division=0)
    cm = confusion_matrix(y_true, predictions["y_pred"])
    lines = ["Metrics", "======="]
    lines += [f"{key}: {value:.4f}" for key, value in metrics.items()]
    lines += ["", "Classification report", "=====================", report]
    lines += ["", "Confusion matrix", "================", str(cm)]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
