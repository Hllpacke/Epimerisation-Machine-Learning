"""
Train a PMX LogD regressor from a CSV file.

This script is a GitHub-ready version of the original notebook workflow. It has
no hard-coded local paths or private data. Provide your own training CSV using
--data-csv and optionally score a second CSV of proposed molecules using
--predict-csv.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from pmx_regression_utils import (
    RegressionConfig,
    ensure_output_dir,
    featurize_mols,
    make_group_cv,
    murcko_smiles,
    prepare_training_dataframe,
    regression_report,
    run_permutation_importance,
    run_shap_summary,
    run_y_randomisation,
    save_applicability_domain_plot,
    save_learning_curve,
    save_model_and_metadata,
    save_parity_plot,
    save_residual_plots,
    save_target_distribution,
    scaffold_split,
    score_new_molecules,
    sim5_to_training,
    train_baselines,
    tune_random_forest,
)


def parse_args() -> RegressionConfig:
    parser = argparse.ArgumentParser(
        description="Train a PMX LogD regressor using RDKit descriptors and Morgan fingerprints."
    )
    parser.add_argument("--data-csv", required=True, help="Path to the training CSV placeholder, e.g. data/training.csv")
    parser.add_argument("--target-col", default="AZ_LogD74", help="Target column in the training CSV, e.g. AZ_LogD74")
    parser.add_argument("--smiles-col", default=None, help="Optional SMILES column name; auto-detected if omitted")
    parser.add_argument("--id-col", default=None, help="Optional compound ID column name; auto-detected if omitted")
    parser.add_argument("--output-dir", default="results/logd", help="Directory for model, tables and plots")
    parser.add_argument("--test-size", type=float, default=0.20, help="External scaffold holdout fraction")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--morgan-radius", type=int, default=2)
    parser.add_argument("--morgan-bits", type=int, default=2048)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=40, help="RandomizedSearchCV iterations")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--skip-y-randomisation", action="store_true")
    parser.add_argument("--n-y-randomisation", type=int, default=30)
    parser.add_argument("--run-shap", action="store_true", help="Run optional SHAP summary if shap is installed")
    parser.add_argument("--shap-top-k", type=int, default=60)
    parser.add_argument("--predict-csv", default=None, help="Optional CSV of new molecules to score")
    parser.add_argument("--predict-output", default=None, help="Output CSV for predictions; defaults inside output-dir")
    parser.add_argument("--add-motif-flags", action="store_true", help="Add simple quinoline/isoquinoline/quinazoline SMARTS flags")
    args = parser.parse_args()

    return RegressionConfig(
        data_csv=args.data_csv,
        target_col=args.target_col,
        output_dir=args.output_dir,
        smiles_col=args.smiles_col,
        id_col=args.id_col,
        test_size=args.test_size,
        random_seed=args.random_seed,
        morgan_radius=args.morgan_radius,
        morgan_bits=args.morgan_bits,
        cv_folds=args.cv_folds,
        n_iter=args.n_iter,
        n_jobs=args.n_jobs,
        run_y_randomisation=not args.skip_y_randomisation,
        n_y_randomisation=args.n_y_randomisation,
        run_shap=args.run_shap,
        shap_top_k=args.shap_top_k,
        predict_csv=args.predict_csv,
        predict_output=args.predict_output,
        add_motif_flags=args.add_motif_flags,
    )


def main() -> None:
    config = parse_args()
    output_dir = ensure_output_dir(config.output_dir)

    df, smiles_col, id_col = prepare_training_dataframe(
        data_csv=config.data_csv,
        target_col=config.target_col,
        smiles_col=config.smiles_col,
        id_col=config.id_col,
    )
    print(f"Training rows after cleaning: {len(df)}")
    print(f"SMILES column: {smiles_col}")
    print(f"ID column: {id_col}")

    y = df[config.target_col].astype(float).values
    save_target_distribution(y, config.target_col, output_dir)

    x = featurize_mols(
        df["Mol"].tolist(),
        radius=config.morgan_radius,
        n_bits=config.morgan_bits,
    )
    x_desc = x[[col for col in x.columns if not col.startswith("morgan_")]].copy()

    scaffolds = df["Mol"].apply(murcko_smiles).values
    df["Scaffold"] = scaffolds

    (
        x_train,
        x_test,
        y_train,
        y_test,
        scaf_train,
        _scaf_test,
        train_idx,
        test_idx,
    ) = scaffold_split(
        x,
        y,
        scaffolds,
        test_size=config.test_size,
        random_seed=config.random_seed,
    )

    print(f"Train: {x_train.shape}; external scaffold holdout: {x_test.shape}")
    cv_iter = make_group_cv(x_train, y_train, scaf_train, requested_folds=config.cv_folds)

    baseline_metrics = train_baselines(
        x_desc=x_desc,
        train_idx=train_idx,
        test_idx=test_idx,
        y_train=y_train,
        y_test=y_test,
        random_seed=config.random_seed,
    )
    baseline_metrics.to_csv(output_dir / "baseline_metrics.csv", index=False)

    search = tune_random_forest(
        x_train=x_train,
        y_train=y_train,
        cv_iter=cv_iter,
        random_seed=config.random_seed,
        n_iter=config.n_iter,
        n_jobs=config.n_jobs,
    )
    best_model = search.best_estimator_
    print("Best parameters:", search.best_params_)

    pred_train = best_model.predict(x_train)
    pred_test = best_model.predict(x_test)
    rf_metrics = pd.DataFrame(
        [
            regression_report(y_train, pred_train, "Train (RF tuned)"),
            regression_report(y_test, pred_test, "External (RF tuned, scaffold holdout)"),
        ]
    )
    metrics_df = pd.concat([baseline_metrics, rf_metrics], ignore_index=True)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    print(metrics_df)

    save_parity_plot(y_train, pred_train, y_test, pred_test, config.target_col, output_dir)
    _res_train, res_test = save_residual_plots(y_train, pred_train, y_test, pred_test, config.target_col, output_dir)
    save_learning_curve(best_model, x_train, y_train, cv_iter, config.target_col, output_dir, n_jobs=config.n_jobs)

    train_smiles = df.iloc[train_idx][smiles_col].astype(str).values
    test_smiles = df.iloc[test_idx][smiles_col].astype(str).values
    sim5_train = sim5_to_training(train_smiles, train_smiles, config.morgan_radius, config.morgan_bits, k=5)
    sim5_test = sim5_to_training(test_smiles, train_smiles, config.morgan_radius, config.morgan_bits, k=5)
    save_applicability_domain_plot(sim5_train, sim5_test, output_dir)

    external = df.iloc[test_idx].copy()
    external["experimental"] = y_test
    external["predicted"] = pred_test
    external["error"] = res_test
    external["abs_error"] = np.abs(res_test)
    external["SIM_5"] = sim5_test
    external = external.drop(columns=["Mol"], errors="ignore")
    external.sort_values("abs_error", ascending=False).to_csv(output_dir / "external_validation_predictions.csv", index=False)

    permutation_df = run_permutation_importance(
        best_model,
        x_test,
        y_test,
        output_dir,
        random_seed=config.random_seed,
        n_jobs=config.n_jobs,
    )

    if config.run_shap:
        run_shap_summary(
            search.best_params_,
            permutation_df,
            x_train,
            y_train,
            output_dir,
            random_seed=config.random_seed,
            n_jobs=config.n_jobs,
            top_k=config.shap_top_k,
        )

    if config.run_y_randomisation:
        run_y_randomisation(
            search.best_params_,
            x_train,
            y_train,
            x_test,
            y_test,
            pred_test,
            output_dir,
            n_perm=config.n_y_randomisation,
            random_seed=config.random_seed,
            n_jobs=config.n_jobs,
        )

    save_model_and_metadata(
        best_model,
        output_dir,
        config,
        smiles_col=smiles_col,
        id_col=id_col,
        best_params=search.best_params_,
        metrics_df=metrics_df,
        model_name="logd_model.joblib",
        metadata_name="logd_metadata.json",
    )

    if config.predict_csv:
        predict_output = config.predict_output or str(output_dir / "new_molecule_predictions.csv")
        scored = score_new_molecules(
            best_model,
            predict_csv=config.predict_csv,
            output_csv=predict_output,
            smiles_col=config.smiles_col,
            id_col=config.id_col,
            prediction_col="LogD_pred",
            radius=config.morgan_radius,
            n_bits=config.morgan_bits,
            add_flags=config.add_motif_flags,
        )
        print(f"Scored {len(scored)} molecules: {predict_output}")

    print(f"Completed. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
