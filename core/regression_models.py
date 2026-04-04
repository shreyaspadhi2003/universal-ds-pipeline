"""
Regression Models Module (Segment 5)
======================================
Runs every suitable regression model:
  - Linear: Linear, Ridge, Lasso, ElasticNet, Polynomial
  - SVM: SVR (linear, rbf, poly)
  - Tree: Decision Tree, Random Forest, Extra Trees, Gradient Boosting, AdaBoost, Bagging
  - Boosting: XGBoost, LightGBM, CatBoost
  - Neighbors: KNN
  - Bayesian: Bayesian Ridge, ARD
  - Robust: Huber, RANSAC, Theil-Sen
  - Other: Passive Aggressive, Gaussian Process, MLP, Tweedie
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

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error,
)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# All regression models
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    BayesianRidge, ARDRegression,
    HuberRegressor, RANSACRegressor, TheilSenRegressor,
    PassiveAggressiveRegressor, SGDRegressor,
    TweedieRegressor,
)
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, AdaBoostRegressor,
    BaggingRegressor, StackingRegressor, VotingRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

warnings.filterwarnings("ignore")


class RegressionModeler:
    """Trains and evaluates all suitable regression models."""

    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results = []
        self.trained_models = {}

    # ----------------------------------------------------------
    # SETUP
    # ----------------------------------------------------------
    def set_data(self, df, target_col, feature_cols=None):
        """
        Prepare train/test split.

        Args:
            df: cleaned DataFrame
            target_col: target variable
            feature_cols: list of feature columns (None = all except target)
        """
        self.logger.section("REGRESSION MODELING")

        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != target_col]

        # Keep only numeric
        X = df[feature_cols].select_dtypes(include=[np.number])
        y = df[target_col]

        # Drop rows with NaN
        valid = X.notna().all(axis=1) & y.notna()
        X = X[valid]
        y = y[valid]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
        )

        self.logger.log(f"Train: {self.X_train.shape}, Test: {self.X_test.shape}")
        self.logger.log(f"Features: {list(X.columns)}")

    # ----------------------------------------------------------
    # MODEL DEFINITIONS
    # ----------------------------------------------------------
    def _get_all_models(self):
        """
        Return dict of {model_name: model_instance} for every
        regression model that suits the dataset.
        """
        n_samples = len(self.X_train)
        n_features = self.X_train.shape[1]

        models = {}

        # --- Linear Models ---
        models["Linear Regression"] = LinearRegression()
        models["Ridge"] = Ridge(alpha=1.0, random_state=self.config.RANDOM_STATE)
        models["Lasso"] = Lasso(alpha=0.1, random_state=self.config.RANDOM_STATE, max_iter=5000)
        models["ElasticNet"] = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.config.RANDOM_STATE, max_iter=5000)
        models["SGD Regressor"] = SGDRegressor(random_state=self.config.RANDOM_STATE, max_iter=1000)
        models["Bayesian Ridge"] = BayesianRidge()
        models["ARD Regression"] = ARDRegression()
        models["Huber Regressor"] = HuberRegressor(max_iter=500)
        models["RANSAC Regressor"] = RANSACRegressor(random_state=self.config.RANDOM_STATE)
        models["Passive Aggressive"] = PassiveAggressiveRegressor(random_state=self.config.RANDOM_STATE, max_iter=1000)
        models["Tweedie Regressor"] = TweedieRegressor(power=0, alpha=0.1, max_iter=1000)

        # Polynomial (degree 2) - only if not too many features
        if n_features <= 15:
            models["Polynomial (deg=2)"] = Pipeline([
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("lr", LinearRegression()),
            ])

        # TheilSen - slow on large datasets
        if n_samples <= 5000:
            models["Theil-Sen"] = TheilSenRegressor(random_state=self.config.RANDOM_STATE)

        # --- SVM ---
        if n_samples <= 10000:
            models["SVR (RBF)"] = SVR(kernel="rbf", C=1.0)
            models["SVR (Poly)"] = SVR(kernel="poly", degree=3, C=1.0)
        models["Linear SVR"] = LinearSVR(random_state=self.config.RANDOM_STATE, max_iter=5000)

        # --- Tree-Based ---
        models["Decision Tree"] = DecisionTreeRegressor(random_state=self.config.RANDOM_STATE, max_depth=10)
        models["Random Forest"] = RandomForestRegressor(
            n_estimators=100, random_state=self.config.RANDOM_STATE, n_jobs=-1
        )
        models["Extra Trees"] = ExtraTreesRegressor(
            n_estimators=100, random_state=self.config.RANDOM_STATE, n_jobs=-1
        )
        models["Gradient Boosting"] = GradientBoostingRegressor(
            n_estimators=100, random_state=self.config.RANDOM_STATE
        )
        models["AdaBoost"] = AdaBoostRegressor(
            n_estimators=100, random_state=self.config.RANDOM_STATE
        )
        models["Bagging"] = BaggingRegressor(
            n_estimators=100, random_state=self.config.RANDOM_STATE, n_jobs=-1
        )

        # --- KNN ---
        models["KNN (k=5)"] = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
        models["KNN (k=10)"] = KNeighborsRegressor(n_neighbors=10, n_jobs=-1)

        # --- Neural Network ---
        models["MLP Regressor"] = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=self.config.RANDOM_STATE,
            early_stopping=True,
        )

        # --- Gaussian Process (small datasets only) ---
        if n_samples <= self.config.MAX_ROWS_FOR_GP:
            models["Gaussian Process"] = GaussianProcessRegressor(
                random_state=self.config.RANDOM_STATE
            )

        # --- XGBoost ---
        try:
            from xgboost import XGBRegressor
            models["XGBoost"] = XGBRegressor(
                n_estimators=100,
                random_state=self.config.RANDOM_STATE,
                verbosity=0,
                n_jobs=-1,
            )
        except ImportError:
            self.logger.warning("XGBoost not installed. Skipping.")

        # --- LightGBM ---
        try:
            from lightgbm import LGBMRegressor
            models["LightGBM"] = LGBMRegressor(
                n_estimators=100,
                random_state=self.config.RANDOM_STATE,
                verbose=-1,
                n_jobs=-1,
            )
        except ImportError:
            self.logger.warning("LightGBM not installed. Skipping.")

        # --- CatBoost ---
        try:
            from catboost import CatBoostRegressor
            models["CatBoost"] = CatBoostRegressor(
                n_estimators=100,
                random_state=self.config.RANDOM_STATE,
                verbose=0,
            )
        except ImportError:
            self.logger.warning("CatBoost not installed. Skipping.")

        return models

    # ----------------------------------------------------------
    # EVALUATION METRICS
    # ----------------------------------------------------------
    def _evaluate(self, model_name, model, train_time):
        """Compute all regression metrics."""
        y_pred = model.predict(self.X_test)

        r2 = r2_score(self.y_test, y_pred)
        n = len(self.y_test)
        p = self.X_test.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / max(n - p - 1, 1)
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)

        try:
            mape = mean_absolute_percentage_error(self.y_test, y_pred)
        except Exception:
            mape = np.nan

        return {
            "model": model_name,
            "r2": round(r2, 6),
            "adjusted_r2": round(adj_r2, 6),
            "mae": round(mae, 6),
            "mse": round(mse, 6),
            "rmse": round(rmse, 6),
            "mape": round(mape, 6) if not np.isnan(mape) else None,
            "train_time_sec": round(train_time, 4),
        }

    # ----------------------------------------------------------
    # TRAIN ALL MODELS
    # ----------------------------------------------------------
    def train_all(self):
        """
        Train every suitable regression model, evaluate, and collect results.

        Returns:
            DataFrame of results sorted by R2
        """
        models = self._get_all_models()
        self.logger.log(f"Training {len(models)} regression models...\n")

        self.results = []
        total = len(models)

        for i, (name, model) in enumerate(models.items(), 1):
            try:
                self.logger.log(f"  [{i}/{total}] {name}...")
                start = time.time()
                model.fit(self.X_train, self.y_train)
                train_time = time.time() - start

                metrics = self._evaluate(name, model, train_time)
                self.results.append(metrics)
                self.trained_models[name] = model

                self.logger.log(
                    f"    R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}, "
                    f"Time={metrics['train_time_sec']:.2f}s"
                )

            except Exception as e:
                self.logger.warning(f"    FAILED: {str(e)}")
                self.results.append({
                    "model": name,
                    "r2": None,
                    "adjusted_r2": None,
                    "mae": None,
                    "mse": None,
                    "rmse": None,
                    "mape": None,
                    "train_time_sec": None,
                    "error": str(e),
                })

        results_df = pd.DataFrame(self.results)

        # Sort by R2 (descending), put failed models at bottom
        successful = results_df[results_df["r2"].notna()].sort_values("r2", ascending=False)
        failed = results_df[results_df["r2"].isna()]
        results_df = pd.concat([successful, failed]).reset_index(drop=True)

        self.logger.success(
            f"\nCompleted: {len(successful)} succeeded, {len(failed)} failed"
        )

        if len(successful) > 0:
            best = successful.iloc[0]
            self.logger.success(
                f"Best model: {best['model']} (R2={best['r2']:.4f}, RMSE={best['rmse']:.4f})"
            )

        return results_df

    # ----------------------------------------------------------
    # SAVE RESULTS
    # ----------------------------------------------------------
    def save_results(self, results_df, output_dir):
        """Save comparison table, plots, and best model info."""
        trad_dir = os.path.join(output_dir, "traditional")
        os.makedirs(trad_dir, exist_ok=True)

        # Save CSV
        csv_path = os.path.join(trad_dir, "regression_results.csv")
        results_df.to_csv(csv_path, index=False)
        self.logger.success(f"Results saved to: {csv_path}")

        # Plot: R2 comparison
        successful = results_df[results_df["r2"].notna()].copy()
        if len(successful) > 0:
            self._plot_r2_comparison(successful, trad_dir)
            self._plot_rmse_comparison(successful, trad_dir)
            self._plot_time_comparison(successful, trad_dir)
            self._plot_actual_vs_predicted(successful, trad_dir)

        # Save best model summary
        if len(successful) > 0:
            best = successful.iloc[0]
            summary = {
                "best_model": best["model"],
                "r2": best["r2"],
                "adjusted_r2": best["adjusted_r2"],
                "mae": best["mae"],
                "rmse": best["rmse"],
                "mape": best["mape"],
                "total_models_tested": len(results_df),
                "successful_models": len(successful),
                "failed_models": len(results_df) - len(successful),
            }
            summary_path = os.path.join(trad_dir, "best_model_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, default=str)
            self.logger.success(f"Best model summary saved to: {summary_path}")

        return csv_path

    def _plot_r2_comparison(self, df, output_dir):
        """Bar chart of R2 scores."""
        fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.35)))
        colors = ["green" if v > 0.8 else "steelblue" if v > 0.5 else "orange" if v > 0 else "red"
                   for v in df["r2"]]
        ax.barh(range(len(df)), df["r2"], color=colors, edgecolor="black")
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["model"], fontsize=9)
        ax.set_xlabel("R2 Score")
        ax.set_title("Regression Models - R2 Comparison")
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "r2_comparison.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)

    def _plot_rmse_comparison(self, df, output_dir):
        """Bar chart of RMSE scores."""
        fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.35)))
        sorted_df = df.sort_values("rmse", ascending=True)
        ax.barh(range(len(sorted_df)), sorted_df["rmse"], color="coral", edgecolor="black")
        ax.set_yticks(range(len(sorted_df)))
        ax.set_yticklabels(sorted_df["model"], fontsize=9)
        ax.set_xlabel("RMSE (lower is better)")
        ax.set_title("Regression Models - RMSE Comparison")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "rmse_comparison.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)

    def _plot_time_comparison(self, df, output_dir):
        """Bar chart of training times."""
        fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.35)))
        sorted_df = df.sort_values("train_time_sec", ascending=True)
        ax.barh(range(len(sorted_df)), sorted_df["train_time_sec"], color="seagreen", edgecolor="black")
        ax.set_yticks(range(len(sorted_df)))
        ax.set_yticklabels(sorted_df["model"], fontsize=9)
        ax.set_xlabel("Training Time (seconds)")
        ax.set_title("Regression Models - Training Time")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_time.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)

    def _plot_actual_vs_predicted(self, df, output_dir):
        """Scatter plot of actual vs predicted for top 3 models."""
        top_models = df.head(3)["model"].tolist()

        fig, axes = plt.subplots(1, min(3, len(top_models)), figsize=(5 * len(top_models), 5))
        if len(top_models) == 1:
            axes = [axes]

        for i, name in enumerate(top_models):
            if name in self.trained_models:
                model = self.trained_models[name]
                y_pred = model.predict(self.X_test)

                axes[i].scatter(self.y_test, y_pred, alpha=0.5, s=15, color="steelblue")
                min_val = min(self.y_test.min(), y_pred.min())
                max_val = max(self.y_test.max(), y_pred.max())
                axes[i].plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.7)
                axes[i].set_xlabel("Actual")
                axes[i].set_ylabel("Predicted")
                axes[i].set_title(f"{name}\nR2={df[df['model']==name]['r2'].values[0]:.4f}")

        plt.suptitle("Actual vs Predicted (Top 3 Models)", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "actual_vs_predicted.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)

    # ----------------------------------------------------------
    # PRINT SUMMARY
    # ----------------------------------------------------------
    def print_results_summary(self, results_df):
        """Print ranking table."""
        print("\n" + "=" * 90)
        print("  REGRESSION MODEL RESULTS")
        print("=" * 90)

        successful = results_df[results_df["r2"].notna()]

        if len(successful) == 0:
            print("  No models completed successfully.")
            print("=" * 90)
            return

        print(f"\n  {'Rank':<6}{'Model':<30}{'R2':<12}{'RMSE':<14}{'MAE':<14}{'Time(s)':<10}")
        print("  " + "-" * 84)

        for i, (_, row) in enumerate(successful.iterrows(), 1):
            print(
                f"  {i:<6}{row['model']:<30}{row['r2']:<12.4f}"
                f"{row['rmse']:<14.4f}{row['mae']:<14.4f}"
                f"{row['train_time_sec']:<10.3f}"
            )

        failed = results_df[results_df["r2"].isna()]
        if len(failed) > 0:
            print(f"\n  Failed Models ({len(failed)}):")
            for _, row in failed.iterrows():
                err = row.get("error", "Unknown error")
                print(f"    - {row['model']}: {err}")

        best = successful.iloc[0]
        print(f"\n  BEST MODEL: {best['model']}")
        print(f"  R2 = {best['r2']:.6f} | RMSE = {best['rmse']:.6f} | MAE = {best['mae']:.6f}")
        print("=" * 90 + "\n")