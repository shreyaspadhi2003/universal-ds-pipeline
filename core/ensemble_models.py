"""
Hybrid / Ensemble Models Module (Segment 11)
==============================================
Handles:
  1. Voting Ensemble (hard/soft for classification, average for regression)
  2. Stacking Ensemble (meta-learner on base models)
  3. Weighted Average Ensemble (weight by individual performance)
  4. Blending Ensemble (holdout-based)
  5. Compare ensembles vs individual models
"""

import os
import time
import pandas as pd
import numpy as np
import json
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import (
    VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor,
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    BaggingClassifier, BaggingRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")


class EnsembleModeler:
    """Builds and evaluates hybrid/ensemble models."""

    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.problem_type = None  # 'regression' or 'classification'
        self.n_classes = 0
        self.results = []
        self.trained_models = {}
        self.individual_results = None  # DataFrame from previous segments

    # ----------------------------------------------------------
    # SETUP
    # ----------------------------------------------------------
    def set_data(self, df, target_col, problem_type, feature_cols=None,
                 individual_results=None):
        """
        Prepare data for ensemble modeling.

        Args:
            df: cleaned DataFrame
            target_col: target variable
            problem_type: 'regression' or 'classification'
            feature_cols: list of features (None = all except target)
            individual_results: DataFrame from regression/classification segments
        """
        self.logger.section("HYBRID / ENSEMBLE MODELING")

        self.problem_type = problem_type
        self.individual_results = individual_results

        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != target_col]

        X = df[feature_cols].select_dtypes(include=[np.number])
        y = df[target_col]

        valid = X.notna().all(axis=1) & y.notna()
        X = X[valid]
        y = y[valid]

        if problem_type == "classification":
            self.n_classes = y.nunique()
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y,
                test_size=self.config.TEST_SIZE,
                random_state=self.config.RANDOM_STATE,
                stratify=y,
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y,
                test_size=self.config.TEST_SIZE,
                random_state=self.config.RANDOM_STATE,
            )

        self.logger.log(f"Problem type: {problem_type}")
        self.logger.log(f"Train: {self.X_train.shape}, Test: {self.X_test.shape}")

    # ----------------------------------------------------------
    # BASE ESTIMATORS
    # ----------------------------------------------------------
    def _get_base_estimators(self):
        """Get diverse base estimators for ensembles."""
        rs = self.config.RANDOM_STATE

        if self.problem_type == "classification":
            estimators = [
                ("rf", RandomForestClassifier(n_estimators=100, random_state=rs, n_jobs=-1)),
                ("gb", GradientBoostingClassifier(n_estimators=100, random_state=rs)),
                ("et", ExtraTreesClassifier(n_estimators=100, random_state=rs, n_jobs=-1)),
                ("knn", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
                ("dt", DecisionTreeClassifier(max_depth=10, random_state=rs)),
            ]

            # Try adding XGBoost
            try:
                from xgboost import XGBClassifier
                estimators.append(
                    ("xgb", XGBClassifier(n_estimators=100, random_state=rs, verbosity=0, n_jobs=-1, eval_metric="logloss"))
                )
            except ImportError:
                pass

            # Try adding LightGBM
            try:
                from lightgbm import LGBMClassifier
                estimators.append(
                    ("lgbm", LGBMClassifier(n_estimators=100, random_state=rs, verbose=-1, n_jobs=-1))
                )
            except ImportError:
                pass

        else:  # regression
            estimators = [
                ("rf", RandomForestRegressor(n_estimators=100, random_state=rs, n_jobs=-1)),
                ("gb", GradientBoostingRegressor(n_estimators=100, random_state=rs)),
                ("et", ExtraTreesRegressor(n_estimators=100, random_state=rs, n_jobs=-1)),
                ("knn", KNeighborsRegressor(n_neighbors=5, n_jobs=-1)),
                ("dt", DecisionTreeRegressor(max_depth=10, random_state=rs)),
            ]

            try:
                from xgboost import XGBRegressor
                estimators.append(
                    ("xgb", XGBRegressor(n_estimators=100, random_state=rs, verbosity=0, n_jobs=-1))
                )
            except ImportError:
                pass

            try:
                from lightgbm import LGBMRegressor
                estimators.append(
                    ("lgbm", LGBMRegressor(n_estimators=100, random_state=rs, verbose=-1, n_jobs=-1))
                )
            except ImportError:
                pass

        return estimators

    # ----------------------------------------------------------
    # EVALUATION
    # ----------------------------------------------------------
    def _evaluate_regression(self, name, y_true, y_pred, train_time):
        """Compute regression metrics."""
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        return {
            "model": name,
            "r2": round(r2, 6),
            "mae": round(mae, 6),
            "rmse": round(rmse, 6),
            "train_time_sec": round(train_time, 4),
        }

    def _evaluate_classification(self, name, y_true, y_pred, y_proba, train_time):
        """Compute classification metrics."""
        avg = "binary" if self.n_classes == 2 else "weighted"

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=avg, zero_division=0)
        recall = recall_score(y_true, y_pred, average=avg, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)

        auc = None
        if y_proba is not None:
            try:
                if self.n_classes == 2:
                    auc = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
            except Exception:
                pass

        return {
            "model": name,
            "accuracy": round(accuracy, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1_score": round(f1, 6),
            "auc_roc": round(auc, 6) if auc is not None else None,
            "train_time_sec": round(train_time, 4),
        }

    # ----------------------------------------------------------
    # 1. VOTING ENSEMBLE
    # ----------------------------------------------------------
    def train_voting(self):
        """Train voting ensemble."""
        estimators = self._get_base_estimators()

        if self.problem_type == "classification":
            # Hard voting
            self._train_single_voting(estimators, "hard")
            # Soft voting
            self._train_single_voting(estimators, "soft")
            # Top 3 voting
            self._train_single_voting(estimators[:3], "soft", suffix="Top3")
        else:
            self._train_single_voting_reg(estimators)
            self._train_single_voting_reg(estimators[:3], suffix="Top3")

    def _train_single_voting(self, estimators, voting_type, suffix=""):
        """Train a single voting classifier."""
        name = f"Voting ({voting_type}){' ' + suffix if suffix else ''}"
        self.logger.log(f"  Training {name}...")

        try:
            start = time.time()
            model = VotingClassifier(estimators=estimators, voting=voting_type, n_jobs=-1)
            model.fit(self.X_train, self.y_train)
            train_time = time.time() - start

            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test) if voting_type == "soft" else None

            metrics = self._evaluate_classification(name, self.y_test, y_pred, y_proba, train_time)
            self.results.append(metrics)
            self.trained_models[name] = model

            self.logger.log(f"    F1={metrics['f1_score']:.4f}, Acc={metrics['accuracy']:.4f}")
        except Exception as e:
            self.logger.warning(f"    {name} failed: {str(e)}")

    def _train_single_voting_reg(self, estimators, suffix=""):
        """Train a single voting regressor."""
        name = f"Voting Regressor{' ' + suffix if suffix else ''}"
        self.logger.log(f"  Training {name}...")

        try:
            start = time.time()
            model = VotingRegressor(estimators=estimators, n_jobs=-1)
            model.fit(self.X_train, self.y_train)
            train_time = time.time() - start

            y_pred = model.predict(self.X_test)
            metrics = self._evaluate_regression(name, self.y_test, y_pred, train_time)
            self.results.append(metrics)
            self.trained_models[name] = model

            self.logger.log(f"    R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")
        except Exception as e:
            self.logger.warning(f"    {name} failed: {str(e)}")

    # ----------------------------------------------------------
    # 2. STACKING ENSEMBLE
    # ----------------------------------------------------------
    def train_stacking(self):
        """Train stacking ensembles with different meta-learners."""
        estimators = self._get_base_estimators()

        if self.problem_type == "classification":
            meta_learners = [
                ("LR", LogisticRegression(max_iter=2000, random_state=self.config.RANDOM_STATE)),
                ("RF", RandomForestClassifier(n_estimators=50, random_state=self.config.RANDOM_STATE)),
            ]
            for meta_name, meta_model in meta_learners:
                self._train_single_stacking_cls(estimators, meta_model, meta_name)
        else:
            meta_learners = [
                ("Ridge", Ridge(alpha=1.0)),
                ("RF", RandomForestRegressor(n_estimators=50, random_state=self.config.RANDOM_STATE)),
            ]
            for meta_name, meta_model in meta_learners:
                self._train_single_stacking_reg(estimators, meta_model, meta_name)

    def _train_single_stacking_cls(self, estimators, meta_model, meta_name):
        """Train single stacking classifier."""
        name = f"Stacking (meta={meta_name})"
        self.logger.log(f"  Training {name}...")

        try:
            start = time.time()
            model = StackingClassifier(
                estimators=estimators,
                final_estimator=meta_model,
                cv=5,
                n_jobs=-1,
            )
            model.fit(self.X_train, self.y_train)
            train_time = time.time() - start

            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test) if hasattr(model, "predict_proba") else None

            metrics = self._evaluate_classification(name, self.y_test, y_pred, y_proba, train_time)
            self.results.append(metrics)
            self.trained_models[name] = model

            self.logger.log(f"    F1={metrics['f1_score']:.4f}, Acc={metrics['accuracy']:.4f}")
        except Exception as e:
            self.logger.warning(f"    {name} failed: {str(e)}")

    def _train_single_stacking_reg(self, estimators, meta_model, meta_name):
        """Train single stacking regressor."""
        name = f"Stacking (meta={meta_name})"
        self.logger.log(f"  Training {name}...")

        try:
            start = time.time()
            model = StackingRegressor(
                estimators=estimators,
                final_estimator=meta_model,
                cv=5,
                n_jobs=-1,
            )
            model.fit(self.X_train, self.y_train)
            train_time = time.time() - start

            y_pred = model.predict(self.X_test)
            metrics = self._evaluate_regression(name, self.y_test, y_pred, train_time)
            self.results.append(metrics)
            self.trained_models[name] = model

            self.logger.log(f"    R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")
        except Exception as e:
            self.logger.warning(f"    {name} failed: {str(e)}")

    # ----------------------------------------------------------
    # 3. WEIGHTED AVERAGE ENSEMBLE
    # ----------------------------------------------------------
    def train_weighted_average(self):
        """Weighted average based on individual model performance."""
        name = "Weighted Average"
        self.logger.log(f"  Training {name}...")

        estimators = self._get_base_estimators()

        try:
            start = time.time()
            predictions = []
            weights = []

            for est_name, est_model in estimators:
                est_model.fit(self.X_train, self.y_train)
                pred = est_model.predict(self.X_test)
                predictions.append(pred)

                # Weight by performance
                if self.problem_type == "regression":
                    score = max(0.01, r2_score(self.y_test, pred))
                else:
                    score = max(0.01, accuracy_score(self.y_test, pred))
                weights.append(score)

            # Normalize weights
            total_w = sum(weights)
            weights = [w / total_w for w in weights]

            if self.problem_type == "regression":
                weighted_pred = np.zeros(len(self.y_test))
                for pred, w in zip(predictions, weights):
                    weighted_pred += pred * w

                train_time = time.time() - start
                metrics = self._evaluate_regression(name, self.y_test, weighted_pred, train_time)
            else:
                # For classification, use weighted mode
                pred_matrix = np.array(predictions).T  # (n_samples, n_models)
                weighted_pred = []
                for i in range(len(self.y_test)):
                    classes = {}
                    for j, w in enumerate(weights):
                        c = pred_matrix[i, j]
                        classes[c] = classes.get(c, 0) + w
                    weighted_pred.append(max(classes, key=classes.get))
                weighted_pred = np.array(weighted_pred)

                train_time = time.time() - start
                metrics = self._evaluate_classification(name, self.y_test, weighted_pred, None, train_time)

            self.results.append(metrics)

            if self.problem_type == "regression":
                self.logger.log(f"    R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")
            else:
                self.logger.log(f"    F1={metrics['f1_score']:.4f}, Acc={metrics['accuracy']:.4f}")

        except Exception as e:
            self.logger.warning(f"    {name} failed: {str(e)}")

    # ----------------------------------------------------------
    # 4. BLENDING ENSEMBLE
    # ----------------------------------------------------------
    def train_blending(self):
        """Blending: train on split, blend predictions with meta-learner."""
        name = "Blending"
        self.logger.log(f"  Training {name}...")

        try:
            start = time.time()
            estimators = self._get_base_estimators()

            # Split training data into train and blend sets
            X_tr, X_blend, y_tr, y_blend = train_test_split(
                self.X_train, self.y_train,
                test_size=0.3,
                random_state=self.config.RANDOM_STATE,
            )

            # Train base models on X_tr, predict on X_blend
            blend_preds = []
            test_preds = []

            for est_name, est_model in estimators:
                est_model.fit(X_tr, y_tr)
                blend_preds.append(est_model.predict(X_blend))
                test_preds.append(est_model.predict(self.X_test))

            blend_features = np.column_stack(blend_preds)
            test_features = np.column_stack(test_preds)

            # Train meta-learner on blend predictions
            if self.problem_type == "classification":
                meta = LogisticRegression(max_iter=2000, random_state=self.config.RANDOM_STATE)
                meta.fit(blend_features, y_blend)
                final_pred = meta.predict(test_features)
                final_proba = meta.predict_proba(test_features) if hasattr(meta, "predict_proba") else None
                train_time = time.time() - start
                metrics = self._evaluate_classification(name, self.y_test, final_pred, final_proba, train_time)
            else:
                meta = Ridge(alpha=1.0)
                meta.fit(blend_features, y_blend)
                final_pred = meta.predict(test_features)
                train_time = time.time() - start
                metrics = self._evaluate_regression(name, self.y_test, final_pred, train_time)

            self.results.append(metrics)

            if self.problem_type == "regression":
                self.logger.log(f"    R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")
            else:
                self.logger.log(f"    F1={metrics['f1_score']:.4f}, Acc={metrics['accuracy']:.4f}")

        except Exception as e:
            self.logger.warning(f"    {name} failed: {str(e)}")

    # ----------------------------------------------------------
    # 5. RUN ALL ENSEMBLES
    # ----------------------------------------------------------
    def train_all(self):
        """
        Train all ensemble methods.

        Returns:
            DataFrame of ensemble results
        """
        self.logger.log(f"Training ensemble models ({self.problem_type})...\n")

        self.train_voting()
        self.train_stacking()
        self.train_weighted_average()
        self.train_blending()

        results_df = pd.DataFrame(self.results)

        if len(results_df) > 0:
            if self.problem_type == "regression":
                sort_col = "r2"
                ascending = False
            else:
                sort_col = "f1_score"
                ascending = False
            results_df = results_df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)

        successful = len(results_df)
        self.logger.success(f"\nEnsemble models complete: {successful} trained")

        if len(results_df) > 0:
            best = results_df.iloc[0]
            if self.problem_type == "regression":
                self.logger.success(f"Best ensemble: {best['model']} (R2={best['r2']:.4f})")
            else:
                self.logger.success(f"Best ensemble: {best['model']} (F1={best['f1_score']:.4f})")

        return results_df

    # ----------------------------------------------------------
    # 6. COMPARE WITH INDIVIDUAL MODELS
    # ----------------------------------------------------------
    def compare_with_individuals(self, ensemble_df, individual_df):
        """
        Create comparison between ensemble and individual models.

        Returns:
            combined DataFrame with source column
        """
        if individual_df is None or individual_df.empty:
            return ensemble_df

        ens = ensemble_df.copy()
        ens["source"] = "ensemble"

        ind = individual_df.copy()
        ind["source"] = "individual"

        # Align columns
        if self.problem_type == "regression":
            common_cols = ["model", "r2", "rmse", "mae", "train_time_sec", "source"]
        else:
            common_cols = ["model", "accuracy", "f1_score", "precision", "recall", "train_time_sec", "source"]

        ens_aligned = ens[[c for c in common_cols if c in ens.columns]]
        ind_aligned = ind[[c for c in common_cols if c in ind.columns]]

        combined = pd.concat([ens_aligned, ind_aligned], ignore_index=True)

        if self.problem_type == "regression":
            combined = combined.sort_values("r2", ascending=False).reset_index(drop=True)
        else:
            combined = combined.sort_values("f1_score", ascending=False).reset_index(drop=True)

        return combined

    # ----------------------------------------------------------
    # 7. SAVE RESULTS
    # ----------------------------------------------------------
    def save_results(self, ensemble_df, output_dir, combined_df=None):
        """Save ensemble results, comparison, and plots."""
        os.makedirs(output_dir, exist_ok=True)

        # Save ensemble results
        csv_path = os.path.join(output_dir, "ensemble_results.csv")
        ensemble_df.to_csv(csv_path, index=False)
        self.logger.success(f"Ensemble results saved to: {csv_path}")

        # Save combined comparison
        if combined_df is not None and len(combined_df) > 0:
            combined_path = os.path.join(output_dir, "ensemble_vs_individual.csv")
            combined_df.to_csv(combined_path, index=False)
            self.logger.success(f"Comparison saved to: {combined_path}")
            self._plot_comparison(combined_df, output_dir)

        # Plot ensemble results
        if len(ensemble_df) > 0:
            self._plot_ensemble_results(ensemble_df, output_dir)

        # Save summary
        if len(ensemble_df) > 0:
            best = ensemble_df.iloc[0]
            summary = {
                "problem_type": self.problem_type,
                "best_ensemble": best["model"],
                "total_ensembles": len(ensemble_df),
            }
            if self.problem_type == "regression":
                summary["best_r2"] = best.get("r2")
                summary["best_rmse"] = best.get("rmse")
            else:
                summary["best_f1"] = best.get("f1_score")
                summary["best_accuracy"] = best.get("accuracy")

            if combined_df is not None and len(combined_df) > 0:
                overall_best = combined_df.iloc[0]
                summary["overall_best_model"] = overall_best["model"]
                summary["overall_best_is_ensemble"] = overall_best["source"] == "ensemble"

            summary_path = os.path.join(output_dir, "ensemble_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, default=str)

        return csv_path

    def _plot_ensemble_results(self, df, output_dir):
        """Plot ensemble model comparison."""
        fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.5)))

        if self.problem_type == "regression":
            metric_col = "r2"
            title = "Ensemble Models - R2 Comparison"
            xlabel = "R2 Score"
        else:
            metric_col = "f1_score"
            title = "Ensemble Models - F1 Score Comparison"
            xlabel = "F1 Score"

        sorted_df = df.sort_values(metric_col, ascending=True)
        colors = ["green" if i == len(sorted_df) - 1 else "steelblue" for i in range(len(sorted_df))]

        ax.barh(range(len(sorted_df)), sorted_df[metric_col], color=colors, edgecolor="black")
        ax.set_yticks(range(len(sorted_df)))
        ax.set_yticklabels(sorted_df["model"], fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "ensemble_comparison.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)

    def _plot_comparison(self, combined_df, output_dir):
        """Plot ensemble vs individual comparison."""
        fig, ax = plt.subplots(figsize=(12, max(6, len(combined_df) * 0.3)))

        if self.problem_type == "regression":
            metric_col = "r2"
            xlabel = "R2 Score"
        else:
            metric_col = "f1_score"
            xlabel = "F1 Score"

        if metric_col not in combined_df.columns:
            return

        top_n = min(25, len(combined_df))
        top = combined_df.head(top_n).copy()

        colors = ["coral" if s == "ensemble" else "steelblue" for s in top["source"]]

        ax.barh(range(len(top)), top[metric_col], color=colors, edgecolor="black")
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(
            [f"{'[E] ' if s == 'ensemble' else ''}{m}" for m, s in zip(top["model"], top["source"])],
            fontsize=8,
        )
        ax.set_xlabel(xlabel)
        ax.set_title(f"Top {top_n} Models: Ensemble [E] vs Individual")
        ax.invert_yaxis()

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="coral", edgecolor="black", label="Ensemble"),
            Patch(facecolor="steelblue", edgecolor="black", label="Individual"),
        ]
        ax.legend(handles=legend_elements, loc="lower right")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "ensemble_vs_individual.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)

    # ----------------------------------------------------------
    # 8. PRINT SUMMARY
    # ----------------------------------------------------------
    def print_results_summary(self, ensemble_df, combined_df=None):
        """Print results table."""
        print("\n" + "=" * 85)
        print("  HYBRID / ENSEMBLE MODEL RESULTS")
        print("=" * 85)

        if len(ensemble_df) == 0:
            print("  No ensemble models completed.")
            print("=" * 85)
            return

        if self.problem_type == "regression":
            print(f"\n  {'Rank':<6}{'Model':<35}{'R2':<12}{'RMSE':<14}{'MAE':<14}{'Time(s)':<10}")
            print("  " + "-" * 83)
            for i, (_, row) in enumerate(ensemble_df.iterrows(), 1):
                print(
                    f"  {i:<6}{row['model']:<35}{row['r2']:<12.4f}"
                    f"{row['rmse']:<14.4f}{row['mae']:<14.4f}{row['train_time_sec']:<10.3f}"
                )
        else:
            print(f"\n  {'Rank':<6}{'Model':<35}{'F1':<10}{'Accuracy':<12}{'Precision':<12}{'Time(s)':<10}")
            print("  " + "-" * 83)
            for i, (_, row) in enumerate(ensemble_df.iterrows(), 1):
                print(
                    f"  {i:<6}{row['model']:<35}{row['f1_score']:<10.4f}"
                    f"{row['accuracy']:<12.4f}{row['precision']:<12.4f}{row['train_time_sec']:<10.3f}"
                )

        # Show if ensemble beat individuals
        if combined_df is not None and len(combined_df) > 0:
            best_overall = combined_df.iloc[0]
            print(f"\n  OVERALL BEST: {best_overall['model']} "
                  f"({'ENSEMBLE' if best_overall['source'] == 'ensemble' else 'INDIVIDUAL'})")

        best_ens = ensemble_df.iloc[0]
        print(f"\n  BEST ENSEMBLE: {best_ens['model']}")
        print("=" * 85 + "\n")