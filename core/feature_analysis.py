"""
Feature Analysis Module (Segment 4)
=====================================
Handles:
  1. 2X, 1Y analysis (pair of features vs target)
  2. Many X, 1Y analysis (all features vs target importance)
  3. Feature importance ranking (multiple methods)
  4. Feature selection (variance, SelectKBest, RFE)
  5. Save plots and reports
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json
from itertools import combinations

from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    RFE,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class FeatureAnalyzer:
    """Analyzes feature relationships and selects best features."""

    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        self.df = None
        self.target_col = None
        self.problem_type = None
        self.feature_cols = []
        self.feature_report = {
            "2x_1y_plots": 0,
            "importance_rankings": {},
            "selected_features": {},
        }

    # ----------------------------------------------------------
    # SETUP
    # ----------------------------------------------------------
    def set_data(self, df, target_col, problem_type):
        """
        Set data for feature analysis.

        Args:
            df: cleaned DataFrame (numeric, encoded, scaled)
            target_col: target variable name
            problem_type: 'regression' or 'classification'
        """
        self.df = df.copy()
        self.target_col = target_col
        self.problem_type = problem_type

        # Feature columns = everything except target
        self.feature_cols = [c for c in self.df.columns if c != self.target_col]

        # Keep only numeric features
        numeric_features = self.df[self.feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        self.feature_cols = numeric_features

        self.logger.section("FEATURE ANALYSIS")
        self.logger.log(f"Features: {len(self.feature_cols)}, Target: {self.target_col} ({self.problem_type})")

    # ----------------------------------------------------------
    # 1. 2X, 1Y ANALYSIS
    # ----------------------------------------------------------
    def analyze_2x_1y(self, output_dir=None, max_pairs=30):
        """
        For every pair of features, create a scatter plot colored by target.

        Args:
            output_dir: directory to save plots
            max_pairs: max number of pairs to plot (avoids explosion)
        """
        self.logger.log(f"Running 2X-1Y analysis (max {max_pairs} pairs)...")

        if len(self.feature_cols) < 2:
            self.logger.warning("Need at least 2 features for 2X-1Y analysis.")
            return

        if self.target_col not in self.df.columns:
            self.logger.warning("Target column not found.")
            return

        plot_dir = os.path.join(output_dir, "2x_1y_plots") if output_dir else None
        if plot_dir:
            os.makedirs(plot_dir, exist_ok=True)

        # Get top features by variance to limit combinations
        if len(self.feature_cols) > 10:
            variances = self.df[self.feature_cols].var().sort_values(ascending=False)
            top_features = variances.index[:10].tolist()
            self.logger.log(f"  Using top 10 features by variance for pair plots")
        else:
            top_features = self.feature_cols

        all_pairs = list(combinations(top_features, 2))
        pairs_to_plot = all_pairs[:max_pairs]

        self.logger.log(f"  Total possible pairs: {len(all_pairs)}, plotting: {len(pairs_to_plot)}")

        plot_count = 0
        for feat1, feat2 in pairs_to_plot:
            try:
                fig, ax = plt.subplots(figsize=(8, 6))

                if self.problem_type == "classification":
                    # Color by class
                    classes = self.df[self.target_col].unique()
                    colors = plt.cm.Set1(np.linspace(0, 1, len(classes)))

                    for cls, color in zip(classes, colors):
                        mask = self.df[self.target_col] == cls
                        ax.scatter(
                            self.df.loc[mask, feat1],
                            self.df.loc[mask, feat2],
                            c=[color],
                            label=f"Class {cls}",
                            alpha=0.5,
                            s=20,
                        )
                    ax.legend(title=self.target_col)

                elif self.problem_type == "regression":
                    # Color by target value
                    scatter = ax.scatter(
                        self.df[feat1],
                        self.df[feat2],
                        c=self.df[self.target_col],
                        cmap="viridis",
                        alpha=0.5,
                        s=20,
                    )
                    plt.colorbar(scatter, ax=ax, label=self.target_col)

                ax.set_xlabel(feat1)
                ax.set_ylabel(feat2)
                ax.set_title(f"{feat1} vs {feat2} (colored by {self.target_col})")
                plt.tight_layout()

                if plot_dir:
                    # Clean filenames
                    safe_f1 = feat1.replace("/", "_").replace(" ", "_")
                    safe_f2 = feat2.replace("/", "_").replace(" ", "_")
                    path = os.path.join(plot_dir, f"2x1y_{safe_f1}_vs_{safe_f2}.png")
                    plt.savefig(path, dpi=80, bbox_inches="tight")
                    plot_count += 1

                plt.close(fig)

            except Exception as e:
                self.logger.warning(f"  Failed plot {feat1} vs {feat2}: {str(e)}")
                plt.close("all")

        self.feature_report["2x_1y_plots"] = plot_count
        self.logger.success(f"2X-1Y plots saved: {plot_count}")

    # ----------------------------------------------------------
    # 2. MANY X, 1Y - FEATURE IMPORTANCE
    # ----------------------------------------------------------
    def compute_feature_importance(self, output_dir=None):
        """
        Compute feature importance using multiple methods:
        1. Correlation with target
        2. Mutual Information
        3. F-test (ANOVA for classification, F-regression for regression)
        4. Random Forest importance

        Returns:
            DataFrame with importance scores from all methods
        """
        self.logger.log("Computing feature importance (multiple methods)...")

        X = self.df[self.feature_cols].copy()
        y = self.df[self.target_col].copy()

        # Drop any rows with NaN
        valid_mask = X.notna().all(axis=1) & y.notna()
        X = X[valid_mask]
        y = y[valid_mask]

        if len(X) == 0:
            self.logger.error("No valid rows for feature importance.")
            return pd.DataFrame()

        importance_df = pd.DataFrame({"feature": self.feature_cols})

        # --- Method 1: Correlation with target ---
        try:
            correlations = X.corrwith(y).abs()
            importance_df["correlation"] = importance_df["feature"].map(correlations).fillna(0)
            self.logger.log("  Method 1: Correlation - done")
        except Exception as e:
            importance_df["correlation"] = 0
            self.logger.warning(f"  Correlation failed: {str(e)}")

        # --- Method 2: Mutual Information ---
        try:
            if self.problem_type == "classification":
                mi_scores = mutual_info_classif(X, y, random_state=self.config.RANDOM_STATE)
            else:
                mi_scores = mutual_info_regression(X, y, random_state=self.config.RANDOM_STATE)
            importance_df["mutual_info"] = mi_scores
            self.logger.log("  Method 2: Mutual Information - done")
        except Exception as e:
            importance_df["mutual_info"] = 0
            self.logger.warning(f"  Mutual Info failed: {str(e)}")

        # --- Method 3: F-test ---
        try:
            if self.problem_type == "classification":
                f_scores, p_values = f_classif(X, y)
            else:
                f_scores, p_values = f_regression(X, y)
            # Replace inf/nan
            f_scores = np.nan_to_num(f_scores, nan=0, posinf=0)
            importance_df["f_score"] = f_scores
            self.logger.log("  Method 3: F-test - done")
        except Exception as e:
            importance_df["f_score"] = 0
            self.logger.warning(f"  F-test failed: {str(e)}")

        # --- Method 4: Random Forest importance ---
        try:
            if self.problem_type == "classification":
                rf = RandomForestClassifier(
                    n_estimators=100, random_state=self.config.RANDOM_STATE, n_jobs=-1
                )
            else:
                rf = RandomForestRegressor(
                    n_estimators=100, random_state=self.config.RANDOM_STATE, n_jobs=-1
                )
            rf.fit(X, y)
            importance_df["rf_importance"] = rf.feature_importances_
            self.logger.log("  Method 4: Random Forest importance - done")
        except Exception as e:
            importance_df["rf_importance"] = 0
            self.logger.warning(f"  RF importance failed: {str(e)}")

        # --- Composite score (normalized average) ---
        score_cols = ["correlation", "mutual_info", "f_score", "rf_importance"]
        for col in score_cols:
            max_val = importance_df[col].max()
            if max_val > 0:
                importance_df[f"{col}_norm"] = importance_df[col] / max_val
            else:
                importance_df[f"{col}_norm"] = 0

        norm_cols = [f"{c}_norm" for c in score_cols]
        importance_df["composite_score"] = importance_df[norm_cols].mean(axis=1).round(4)

        # Sort by composite
        importance_df = importance_df.sort_values("composite_score", ascending=False).reset_index(drop=True)

        # Store in report
        self.feature_report["importance_rankings"] = {
            "top_10": importance_df.head(10)[["feature", "composite_score"]].to_dict("records"),
            "methods_used": score_cols,
        }

        # Save
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            # Save full ranking
            csv_path = os.path.join(output_dir, "feature_importance.csv")
            importance_df.to_csv(csv_path, index=False)
            self.logger.success(f"Feature importance saved to: {csv_path}")

            # Plot top features
            self._plot_importance(importance_df, output_dir)

        self.logger.success(f"Feature importance computed for {len(self.feature_cols)} features")
        return importance_df

    def _plot_importance(self, importance_df, output_dir):
        """Plot feature importance bar charts."""
        top_n = min(20, len(importance_df))
        top = importance_df.head(top_n)

        # Composite score plot
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
        bars = ax.barh(
            range(top_n),
            top["composite_score"],
            color="steelblue",
            edgecolor="black",
        )
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top["feature"])
        ax.set_xlabel("Composite Importance Score")
        ax.set_title(f"Top {top_n} Features by Composite Importance")
        ax.invert_yaxis()
        plt.tight_layout()

        path = os.path.join(output_dir, "feature_importance_composite.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        # Individual method comparison
        methods = ["correlation", "mutual_info", "f_score", "rf_importance"]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for i, method in enumerate(methods):
            sorted_df = importance_df.nlargest(top_n, method)
            axes[i].barh(
                range(len(sorted_df)),
                sorted_df[method],
                color=["steelblue", "coral", "seagreen", "orange"][i],
                edgecolor="black",
            )
            axes[i].set_yticks(range(len(sorted_df)))
            axes[i].set_yticklabels(sorted_df["feature"], fontsize=8)
            axes[i].set_title(f"{method.replace('_', ' ').title()}")
            axes[i].invert_yaxis()

        plt.suptitle("Feature Importance by Method", fontsize=14)
        plt.tight_layout()

        path = os.path.join(output_dir, "feature_importance_by_method.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        self.logger.log(f"  Importance plots saved")

    # ----------------------------------------------------------
    # 3. FEATURE SELECTION
    # ----------------------------------------------------------
    def select_features(self, importance_df=None, output_dir=None):
        """
        Apply multiple feature selection methods:
        1. Variance Threshold
        2. SelectKBest
        3. Recursive Feature Elimination (RFE)

        Returns:
            dict with selected features from each method + recommended set
        """
        self.logger.log("Running feature selection...")

        X = self.df[self.feature_cols].copy()
        y = self.df[self.target_col].copy()

        valid_mask = X.notna().all(axis=1) & y.notna()
        X = X[valid_mask]
        y = y[valid_mask]

        selection_results = {}

        # --- Method 1: Variance Threshold ---
        try:
            vt = VarianceThreshold(threshold=0.01)
            vt.fit(X)
            vt_features = X.columns[vt.get_support()].tolist()
            selection_results["variance_threshold"] = vt_features
            self.logger.log(f"  Variance Threshold: {len(vt_features)}/{len(self.feature_cols)} features kept")
        except Exception as e:
            selection_results["variance_threshold"] = self.feature_cols
            self.logger.warning(f"  Variance Threshold failed: {str(e)}")

        # --- Method 2: SelectKBest ---
        try:
            k = min(max(5, len(self.feature_cols) // 2), len(self.feature_cols))
            if self.problem_type == "classification":
                selector = SelectKBest(f_classif, k=k)
            else:
                selector = SelectKBest(f_regression, k=k)
            selector.fit(X, y)
            kbest_features = X.columns[selector.get_support()].tolist()
            selection_results["select_k_best"] = kbest_features
            self.logger.log(f"  SelectKBest (k={k}): {len(kbest_features)} features selected")
        except Exception as e:
            selection_results["select_k_best"] = self.feature_cols
            self.logger.warning(f"  SelectKBest failed: {str(e)}")

        # --- Method 3: RFE with Random Forest ---
        try:
            n_select = min(max(5, len(self.feature_cols) // 2), len(self.feature_cols))
            if self.problem_type == "classification":
                estimator = RandomForestClassifier(
                    n_estimators=50, random_state=self.config.RANDOM_STATE, n_jobs=-1
                )
            else:
                estimator = RandomForestRegressor(
                    n_estimators=50, random_state=self.config.RANDOM_STATE, n_jobs=-1
                )
            rfe = RFE(estimator, n_features_to_select=n_select, step=1)
            rfe.fit(X, y)
            rfe_features = X.columns[rfe.get_support()].tolist()
            selection_results["rfe"] = rfe_features
            self.logger.log(f"  RFE: {len(rfe_features)} features selected")
        except Exception as e:
            selection_results["rfe"] = self.feature_cols
            self.logger.warning(f"  RFE failed: {str(e)}")

        # --- Recommended: features selected by at least 2 methods ---
        all_selected = []
        for features in selection_results.values():
            all_selected.extend(features)

        from collections import Counter
        feature_votes = Counter(all_selected)
        recommended = [f for f, count in feature_votes.items() if count >= 2]

        # If too few, fall back to importance ranking
        if len(recommended) < 3 and importance_df is not None:
            recommended = importance_df.head(max(5, len(self.feature_cols) // 2))["feature"].tolist()
            self.logger.log("  Recommended set too small, using importance ranking fallback")

        selection_results["recommended"] = recommended

        self.feature_report["selected_features"] = {
            method: features for method, features in selection_results.items()
        }

        # Save
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            # Save selection results
            csv_path = os.path.join(output_dir, "selected_features.csv")
            rows = []
            for method, features in selection_results.items():
                for f in features:
                    rows.append({"method": method, "feature": f})
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            self.logger.success(f"Selection results saved to: {csv_path}")

            # Plot Venn-like comparison
            self._plot_selection_comparison(selection_results, output_dir)

        self.logger.success(f"Recommended features: {len(recommended)}")
        return selection_results

    def _plot_selection_comparison(self, selection_results, output_dir):
        """Plot which features each method selected."""
        methods = ["variance_threshold", "select_k_best", "rfe"]
        all_features = set()
        for m in methods:
            all_features.update(selection_results.get(m, []))

        all_features = sorted(all_features)
        if len(all_features) == 0:
            return

        # Create binary matrix
        matrix = []
        for f in all_features:
            row = [1 if f in selection_results.get(m, []) else 0 for m in methods]
            matrix.append(row)

        matrix_df = pd.DataFrame(matrix, index=all_features, columns=methods)

        fig, ax = plt.subplots(figsize=(8, max(6, len(all_features) * 0.3)))
        sns.heatmap(
            matrix_df,
            annot=True,
            cmap="YlGn",
            cbar=False,
            linewidths=0.5,
            ax=ax,
            fmt="d",
        )
        ax.set_title("Feature Selection Method Comparison")
        ax.set_xlabel("Method")
        plt.tight_layout()

        path = os.path.join(output_dir, "feature_selection_comparison.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    # ----------------------------------------------------------
    # 4. RUN FULL FEATURE ANALYSIS
    # ----------------------------------------------------------
    def run_full_analysis(self, output_dir=None):
        """
        Run all feature analysis steps.

        Returns:
            dict with importance_df and selection_results
        """
        self.logger.section("FULL FEATURE ANALYSIS")

        # 1. 2X-1Y plots
        self.analyze_2x_1y(output_dir)

        # 2. Feature importance
        importance_df = self.compute_feature_importance(output_dir)

        # 3. Feature selection
        selection_results = self.select_features(importance_df, output_dir)

        # Save report
        if output_dir:
            report_path = os.path.join(output_dir, "feature_report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(self.feature_report, f, indent=2, default=str)
            self.logger.success(f"Feature report saved to: {report_path}")

        self.logger.success("Full feature analysis complete")
        return {
            "importance": importance_df,
            "selection": selection_results,
        }

    # ----------------------------------------------------------
    # 5. PRINT SUMMARY
    # ----------------------------------------------------------
    def print_feature_summary(self):
        """Print feature analysis summary."""
        print("\n" + "=" * 60)
        print("  FEATURE ANALYSIS SUMMARY")
        print("=" * 60)

        print(f"\n  Total Features: {len(self.feature_cols)}")
        print(f"  2X-1Y Plots Generated: {self.feature_report['2x_1y_plots']}")

        if self.feature_report["importance_rankings"]:
            print(f"\n  Top 10 Features (Composite Score):")
            for item in self.feature_report["importance_rankings"].get("top_10", []):
                print(f"    {item['feature']:35s} : {item['composite_score']:.4f}")

        if self.feature_report["selected_features"]:
            print(f"\n  Feature Selection Results:")
            for method, features in self.feature_report["selected_features"].items():
                print(f"    {method:25s} : {len(features)} features")

            recommended = self.feature_report["selected_features"].get("recommended", [])
            if recommended:
                print(f"\n  Recommended Features ({len(recommended)}):")
                for f in recommended:
                    print(f"    - {f}")

        print("=" * 60 + "\n")