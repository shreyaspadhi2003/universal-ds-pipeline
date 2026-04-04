"""
Universal Data Science Pipeline - Main Runner
===============================================
USAGE:
    python main.py

Features:
  - Never crashes: every phase wrapped in error handling
  - SHAP explainability for best model
  - Interactive inputs for all options
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils.logger import PipelineLogger
from core.data_ingestion import DataIngestor
from core.data_cleaning import DataCleaner
from core.eda import EDAAnalyzer
from core.feature_analysis import FeatureAnalyzer
from core.regression_models import RegressionModeler
from core.classification_models import ClassificationModeler
from core.clustering_models import ClusteringModeler
from core.association_rules import AssociationAnalyzer
from core.time_series_models import TimeSeriesModeler
from core.anomaly_detection import AnomalyDetector
from core.ensemble_models import EnsembleModeler
from core.final_comparison import FinalComparison
from core.shap_explainer import ShapExplainer


def safe_run(phase_name, func, logger):
    """
    Run a function safely. If it errors, log and continue.
    Never crashes the pipeline.
    """
    try:
        return func()
    except Exception as e:
        logger.error(f"[{phase_name}] Error intercepted: {str(e)}")
        print(f"\n  *** ERROR in {phase_name} ***")
        print(f"  {str(e)}")
        print(f"  Pipeline continues...\n")
        # Log full traceback to file only
        with open(config.LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n--- TRACEBACK for {phase_name} ---\n")
            traceback.print_exc(file=f)
            f.write("--- END TRACEBACK ---\n\n")
        return None


def get_user_input():
    """Gather all user inputs interactively."""
    print("\n" + "=" * 60)
    print("  UNIVERSAL DATA SCIENCE PIPELINE")
    print("=" * 60)

    # --- File paths ---
    print("\nEnter file path(s), comma-separated for multiple files:")
    print("Example: data.csv  OR  customers.csv, orders.csv")
    file_input = input("> ").strip()
    file_paths = [f.strip() for f in file_input.split(",")]

    valid_paths = []
    for fp in file_paths:
        if os.path.exists(fp):
            valid_paths.append(fp)
        else:
            print(f"  WARNING: File not found: {fp}")

    if not valid_paths:
        print("ERROR: No valid files found. Exiting.")
        sys.exit(1)

    # --- Deep learning ---
    print("\nEnable deep learning models? (requires TensorFlow) [y/n]")
    dl_input = input("> ").strip().lower()
    config.ENABLE_DEEP_LEARNING = dl_input in ["y", "yes"]

    # --- Run options ---
    print("\nWhich phases to run? (Enter numbers comma-separated, or 'all')")
    print("  1.  Data Preparation & Cleaning")
    print("  2.  EDA & Visualization")
    print("  3.  Feature Analysis")
    print("  4.  Regression Models")
    print("  5.  Classification Models")
    print("  6.  Clustering")
    print("  7.  Association Rules")
    print("  8.  Time Series")
    print("  9.  Anomaly Detection")
    print("  10. Ensemble Models")
    print("  11. SHAP Explainability")
    print("  12. Final Comparison")

    phase_input = input("> ").strip().lower()
    if phase_input == "all":
        phases = list(range(1, 13))
    else:
        try:
            phases = [int(x.strip()) for x in phase_input.split(",")]
        except ValueError:
            print("Invalid input. Running all phases.")
            phases = list(range(1, 13))

    return valid_paths, phases


def run_pipeline(file_paths, phases):
    """Run the full pipeline with error handling on every phase."""
    logger = PipelineLogger(config.LOG_FILE, verbose=config.VERBOSE)
    comparison = FinalComparison(logger, config)

    # ========================================
    # PHASE 1: DATA INGESTION & CLEANING
    # ========================================
    ingestor = DataIngestor(logger)
    ingestor.load_files(file_paths)

    # Handle multiple files
    if len(ingestor.dataframes) > 1:
        shared_keys = ingestor.detect_relational_keys()
        if shared_keys:
            print("\nRelational keys detected:")
            for (f1, f2), keys in shared_keys.items():
                print(f"  {f1} <-> {f2}: {keys}")

            print("\nMerge these tables? [y/n]")
            merge_input = input("> ").strip().lower()
            if merge_input in ["y", "yes"]:
                filenames = list(ingestor.dataframes.keys())
                first_pair = list(shared_keys.keys())[0]
                merge_keys = shared_keys[first_pair]

                print(f"\nMerge type? [inner/left/right/outer] (default: inner)")
                how = input("> ").strip().lower() or "inner"

                ingestor.merge_tables(first_pair[0], first_pair[1],
                                      on_columns=merge_keys, how=how)
                ingestor.set_working_dataframe(df=ingestor.merged_df)
            else:
                print("\nWhich file to use as primary?")
                for i, name in enumerate(ingestor.dataframes.keys(), 1):
                    print(f"  {i}. {name}")
                choice = int(input("> ").strip()) - 1
                fname = list(ingestor.dataframes.keys())[choice]
                ingestor.set_working_dataframe(filename=fname)
        else:
            ingestor.set_working_dataframe()
    else:
        ingestor.set_working_dataframe()

    ingestor.detect_column_types()
    ingestor.print_summary()

    # --- Target variable ---
    candidates = ingestor.get_target_candidates()
    print("\nTarget variable candidates:")
    print(f"  Regression     : {candidates['regression']}")
    print(f"  Classification : {candidates['classification']}")
    print(f"\nEnter target variable name (or 'none' for unsupervised):")
    target_input = input("> ").strip()

    problem_type = None
    target_col = None

    if target_input.lower() != "none" and target_input in ingestor.merged_df.columns:
        target_col = target_input
        problem_type = ingestor.set_target(target_col)
    else:
        print("  No target set. Will run unsupervised methods only.")

    comparison.set_problem_type(problem_type)

    original_df = ingestor.merged_df.copy()

    # --- Cleaning ---
    cleaned_df = ingestor.merged_df.copy()
    if 1 in phases:
        def do_cleaning():
            nonlocal cleaned_df
            cleaner = DataCleaner(logger, config)
            cleaner.set_data(ingestor.merged_df, ingestor.column_types, target_col=target_col)

            print("\nMissing value strategy? [auto/mean/median/mode/knn/drop_rows] (default: auto)")
            missing_strategy = input("> ").strip().lower() or "auto"

            cleaned_df = cleaner.run_full_cleaning(missing_strategy=missing_strategy)
            cleaner.save_cleaned_data(config.DATA_PREP_DIR)
            cleaner.print_cleaning_summary()
            ingestor.generate_metadata(output_dir=config.DATA_PREP_DIR)
            return cleaned_df

        result = safe_run("Data Cleaning", do_cleaning, logger)
        if result is not None:
            cleaned_df = result

    # ========================================
    # PHASE 2: EDA
    # ========================================
    if 2 in phases:
        def do_eda():
            eda = EDAAnalyzer(logger, config)
            eda.set_data(
                cleaned_df=cleaned_df,
                original_df=original_df,
                column_types=ingestor.column_types,
                target_col=target_col,
                problem_type=problem_type,
            )
            eda.run_full_eda(output_dir=config.EDA_DIR)
            eda.print_eda_summary()

        safe_run("EDA", do_eda, logger)

    # ========================================
    # PHASE 3: FEATURE ANALYSIS
    # ========================================
    if 3 in phases and target_col:
        def do_feature():
            analyzer = FeatureAnalyzer(logger, config)
            analyzer.set_data(cleaned_df, target_col=target_col, problem_type=problem_type)
            analyzer.run_full_analysis(output_dir=config.FEATURE_DIR)
            analyzer.print_feature_summary()

        safe_run("Feature Analysis", do_feature, logger)

    # ========================================
    # PHASE 4: REGRESSION
    # ========================================
    reg_results = None
    best_reg_model = None
    reg_X_train = None
    reg_X_test = None

    if 4 in phases and problem_type == "regression":
        def do_regression():
            nonlocal reg_results, best_reg_model, reg_X_train, reg_X_test
            reg_modeler = RegressionModeler(logger, config)
            reg_modeler.set_data(cleaned_df, target_col=target_col)
            reg_results = reg_modeler.train_all()
            reg_modeler.save_results(reg_results, config.REGRESSION_DIR)
            reg_modeler.print_results_summary(reg_results)
            comparison.add_results("regression", reg_results)

            # Store best model for SHAP
            successful = reg_results[reg_results["r2"].notna()]
            if len(successful) > 0:
                best_name = successful.iloc[0]["model"]
                if best_name in reg_modeler.trained_models:
                    best_reg_model = reg_modeler.trained_models[best_name]
                    reg_X_train = reg_modeler.X_train
                    reg_X_test = reg_modeler.X_test

        safe_run("Regression Models", do_regression, logger)

    # ========================================
    # PHASE 5: CLASSIFICATION
    # ========================================
    cls_results = None
    best_cls_model = None
    cls_X_train = None
    cls_X_test = None

    if 5 in phases and problem_type == "classification":
        def do_classification():
            nonlocal cls_results, best_cls_model, cls_X_train, cls_X_test
            cls_modeler = ClassificationModeler(logger, config)
            cls_modeler.set_data(cleaned_df, target_col=target_col)
            cls_results = cls_modeler.train_all()
            cls_modeler.save_results(cls_results, config.CLASSIFICATION_DIR)
            cls_modeler.print_results_summary(cls_results)
            comparison.add_results("classification", cls_results)

            successful = cls_results[cls_results["accuracy"].notna()]
            if len(successful) > 0:
                best_name = successful.iloc[0]["model"]
                if best_name in cls_modeler.trained_models:
                    best_cls_model = cls_modeler.trained_models[best_name]
                    cls_X_train = cls_modeler.X_train
                    cls_X_test = cls_modeler.X_test

        safe_run("Classification Models", do_classification, logger)

    # ========================================
    # PHASE 6: CLUSTERING
    # ========================================
    if 6 in phases:
        def do_clustering():
            clust_modeler = ClusteringModeler(logger, config)
            clust_modeler.set_data(cleaned_df, target_col=target_col)
            clust_modeler.find_optimal_k(
                output_dir=os.path.join(config.CLUSTERING_DIR, "traditional")
            )
            clust_results = clust_modeler.train_all()
            clust_modeler.save_results(clust_results, config.CLUSTERING_DIR)
            clust_modeler.print_results_summary(clust_results)

        safe_run("Clustering", do_clustering, logger)

    # ========================================
    # PHASE 7: ASSOCIATION RULES
    # ========================================
    if 7 in phases and config.ENABLE_ASSOCIATION:
        def do_association():
            assoc = AssociationAnalyzer(logger, config)
            assoc_results = assoc.run_all(original_df)
            if assoc_results.get("suitable"):
                assoc.save_results(config.ASSOCIATION_DIR)
                assoc.print_summary()

        safe_run("Association Rules", do_association, logger)

    # ========================================
    # PHASE 8: TIME SERIES
    # ========================================
    if 8 in phases and config.ENABLE_TIME_SERIES:
        def do_timeseries():
            ts_modeler = TimeSeriesModeler(logger, config)
            ts_detection = ts_modeler.detect_time_series(original_df, ingestor.column_types)

            if ts_detection["is_timeseries"]:
                dt_col = ts_detection["datetime_cols"][0]
                val_col = ts_detection["value_cols"][0] if target_col is None else target_col

                print(f"\nTime series detected. Datetime: {dt_col}, Value: {val_col}")
                print("Run time series models? [y/n]")
                ts_input = input("> ").strip().lower()

                if ts_input in ["y", "yes"]:
                    ts_modeler.prepare_series(original_df, datetime_col=dt_col, value_col=val_col)
                    ts_results = ts_modeler.train_all()
                    ts_modeler.save_results(ts_results, config.TIMESERIES_DIR)
                    ts_modeler.print_results_summary(ts_results)

        safe_run("Time Series", do_timeseries, logger)

    # ========================================
    # PHASE 9: ANOMALY DETECTION
    # ========================================
    if 9 in phases and config.ENABLE_ANOMALY:
        def do_anomaly():
            detector = AnomalyDetector(logger, config)
            detector.set_data(cleaned_df, target_col=target_col)
            detector.run_all()
            detector.save_results(config.ANOMALY_DIR)
            detector.print_results_summary()

        safe_run("Anomaly Detection", do_anomaly, logger)

    # ========================================
    # PHASE 10: ENSEMBLE MODELS
    # ========================================
    if 10 in phases and config.ENABLE_ENSEMBLE and problem_type:
        def do_ensemble():
            individual_results = reg_results if problem_type == "regression" else cls_results

            ens_modeler = EnsembleModeler(logger, config)
            ens_modeler.set_data(cleaned_df, target_col=target_col, problem_type=problem_type,
                                 individual_results=individual_results)
            ens_results = ens_modeler.train_all()

            successful_ind = None
            if individual_results is not None:
                if problem_type == "regression":
                    successful_ind = individual_results[individual_results["r2"].notna()]
                else:
                    successful_ind = individual_results[individual_results["accuracy"].notna()]

            combined = ens_modeler.compare_with_individuals(ens_results, successful_ind)
            ens_modeler.save_results(ens_results, config.ENSEMBLE_DIR, combined_df=combined)
            ens_modeler.print_results_summary(ens_results, combined)
            comparison.add_results("ensemble", ens_results)

        safe_run("Ensemble Models", do_ensemble, logger)

    # ========================================
    # PHASE 11: SHAP EXPLAINABILITY
    # ========================================
    if 11 in phases and problem_type:
        def do_shap():
            shap_dir = os.path.join(config.COMPARISON_DIR, "shap")
            shap_exp = ShapExplainer(logger, config)

            if problem_type == "regression" and best_reg_model is not None:
                best_name = reg_results[reg_results["r2"].notna()].iloc[0]["model"]
                shap_exp.compute_shap(
                    model=best_reg_model,
                    X_train=reg_X_train,
                    X_test=reg_X_test,
                    problem_type="regression",
                    model_name=best_name,
                )
                shap_exp.generate_plots(shap_dir)
                shap_exp.print_summary()

            elif problem_type == "classification" and best_cls_model is not None:
                best_name = cls_results[cls_results["accuracy"].notna()].iloc[0]["model"]
                shap_exp.compute_shap(
                    model=best_cls_model,
                    X_train=cls_X_train,
                    X_test=cls_X_test,
                    problem_type="classification",
                    model_name=best_name,
                )
                shap_exp.generate_plots(shap_dir)
                shap_exp.print_summary()
            else:
                logger.warning("No trained model available for SHAP. Run regression or classification first.")

        safe_run("SHAP Explainability", do_shap, logger)

    # ========================================
    # PHASE 12: FINAL COMPARISON
    # ========================================
    if 12 in phases and problem_type:
        def do_comparison():
            grand_ranking = comparison.build_grand_ranking()
            comparison.save_results(config.COMPARISON_DIR)
            comparison.print_grand_summary()

        safe_run("Final Comparison", do_comparison, logger)

    # ========================================
    # DONE
    # ========================================
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"\n  All outputs saved to: {config.OUTPUT_DIR}")
    print(f"  Log file: {config.LOG_FILE}")
    print(f"\n  Output directories:")
    print(f"    Data Preparation : {config.DATA_PREP_DIR}")
    print(f"    EDA              : {config.EDA_DIR}")
    print(f"    Feature Analysis : {config.FEATURE_DIR}")
    print(f"    Regression       : {config.REGRESSION_DIR}")
    print(f"    Classification   : {config.CLASSIFICATION_DIR}")
    print(f"    Clustering       : {config.CLUSTERING_DIR}")
    print(f"    Association Rules: {config.ASSOCIATION_DIR}")
    print(f"    Time Series      : {config.TIMESERIES_DIR}")
    print(f"    Anomaly Detection: {config.ANOMALY_DIR}")
    print(f"    Ensemble         : {config.ENSEMBLE_DIR}")
    print(f"    SHAP             : {os.path.join(config.COMPARISON_DIR, 'shap')}")
    print(f"    Comparison       : {config.COMPARISON_DIR}")
    print("=" * 60 + "\n")


def main():
    file_paths, phases = get_user_input()
    run_pipeline(file_paths, phases)


if __name__ == "__main__":
    main()