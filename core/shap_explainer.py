"""
SHAP Explainability Module
============================
Handles:
  1. Computing SHAP values for best model
  2. Summary plot (beeswarm)
  3. Bar plot (mean |SHAP|)
  4. Dependence plots (top features)
  5. Waterfall plot (single prediction)
  6. Works for both regression and classification
"""

import os
import numpy as np
import pandas as pd
import json
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class ShapExplainer:
    """Generates SHAP-based model explanations."""

    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        self.shap_values = None
        self.explainer = None
        self.X_explain = None
        self.feature_names = []
        self.report = {}

    # ----------------------------------------------------------
    # 1. COMPUTE SHAP VALUES
    # ----------------------------------------------------------
    def compute_shap(self, model, X_train, X_test, problem_type="regression",
                     model_name="Best Model", max_samples=500):
        """
        Compute SHAP values for the given model.

        Args:
            model: trained sklearn-compatible model
            X_train: training features (for background)
            X_test: test features (to explain)
            problem_type: 'regression' or 'classification'
            model_name: name for display
            max_samples: max samples to explain (SHAP can be slow)

        Returns:
            shap_values object
        """
        if not SHAP_AVAILABLE:
            self.logger.error("SHAP not installed. Run: pip install shap")
            return None

        self.logger.section("SHAP EXPLAINABILITY")
        self.logger.log(f"Computing SHAP values for: {model_name}")

        # Limit samples for speed
        if len(X_test) > max_samples:
            self.X_explain = X_test.sample(n=max_samples, random_state=self.config.RANDOM_STATE)
        else:
            self.X_explain = X_test.copy()

        if len(X_train) > max_samples:
            X_background = X_train.sample(n=max_samples, random_state=self.config.RANDOM_STATE)
        else:
            X_background = X_train.copy()

        self.feature_names = list(self.X_explain.columns)

        try:
            # Try TreeExplainer first (fast for tree-based models)
            if self._is_tree_model(model):
                self.logger.log("  Using TreeExplainer (fast)")
                self.explainer = shap.TreeExplainer(model)
                self.shap_values = self.explainer.shap_values(self.X_explain)
            else:
                # Fall back to KernelExplainer (works for any model, slower)
                self.logger.log("  Using KernelExplainer (slower, universal)")
                # Use smaller background for KernelExplainer
                bg_sample = shap.sample(X_background, min(100, len(X_background)))

                if problem_type == "regression":
                    self.explainer = shap.KernelExplainer(model.predict, bg_sample)
                else:
                    if hasattr(model, "predict_proba"):
                        self.explainer = shap.KernelExplainer(model.predict_proba, bg_sample)
                    else:
                        self.explainer = shap.KernelExplainer(model.predict, bg_sample)

                # Limit explanation samples for KernelExplainer (very slow)
                explain_sample = self.X_explain.head(min(100, len(self.X_explain)))
                self.shap_values = self.explainer.shap_values(explain_sample)
                self.X_explain = explain_sample

            # Handle classification multi-output
            if isinstance(self.shap_values, list):
                # For binary classification, take class 1
                if len(self.shap_values) == 2:
                    self.shap_values = self.shap_values[1]
                else:
                    # For multiclass, take mean absolute across classes
                    self.shap_values = np.mean(np.abs(np.array(self.shap_values)), axis=0)

            self.report["model_name"] = model_name
            self.report["problem_type"] = problem_type
            self.report["n_samples_explained"] = len(self.X_explain)
            self.report["n_features"] = len(self.feature_names)

            self.logger.success(f"SHAP values computed: {self.shap_values.shape}")
            return self.shap_values

        except Exception as e:
            self.logger.error(f"SHAP computation failed: {str(e)}")
            return None

    def _is_tree_model(self, model):
        """Check if model is tree-based (supports TreeExplainer)."""
        tree_types = [
            "RandomForest", "GradientBoosting", "ExtraTrees",
            "DecisionTree", "XGB", "LGBM", "CatBoost",
            "AdaBoost", "Bagging",
        ]
        model_type = type(model).__name__
        return any(t in model_type for t in tree_types)

    # ----------------------------------------------------------
    # 2. GENERATE ALL PLOTS
    # ----------------------------------------------------------
    def generate_plots(self, output_dir):
        """Generate all SHAP visualizations."""
        if self.shap_values is None:
            self.logger.warning("No SHAP values computed. Call compute_shap() first.")
            return

        os.makedirs(output_dir, exist_ok=True)

        self._plot_summary_bar(output_dir)
        self._plot_summary_beeswarm(output_dir)
        self._plot_dependence(output_dir)
        self._plot_waterfall(output_dir)
        self._save_importance_csv(output_dir)
        self._save_report(output_dir)

    def _plot_summary_bar(self, output_dir):
        """Bar plot of mean |SHAP| values."""
        try:
            self.logger.log("  Generating SHAP bar plot...")
            fig, ax = plt.subplots(figsize=(10, max(6, len(self.feature_names) * 0.3)))

            mean_shap = np.abs(self.shap_values).mean(axis=0)
            sorted_idx = np.argsort(mean_shap)
            sorted_features = [self.feature_names[i] for i in sorted_idx]
            sorted_values = mean_shap[sorted_idx]

            ax.barh(range(len(sorted_features)), sorted_values,
                     color="steelblue", edgecolor="black")
            ax.set_yticks(range(len(sorted_features)))
            ax.set_yticklabels(sorted_features, fontsize=9)
            ax.set_xlabel("Mean |SHAP Value|")
            ax.set_title(f"SHAP Feature Importance - {self.report.get('model_name', 'Model')}")
            plt.tight_layout()

            path = os.path.join(output_dir, "shap_bar.png")
            plt.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            self.logger.success(f"  Bar plot saved: {path}")

        except Exception as e:
            self.logger.warning(f"  Bar plot failed: {str(e)}")
            plt.close("all")

    def _plot_summary_beeswarm(self, output_dir):
        """Beeswarm summary plot."""
        try:
            self.logger.log("  Generating SHAP beeswarm plot...")
            fig = plt.figure(figsize=(10, max(6, len(self.feature_names) * 0.3)))

            shap.summary_plot(
                self.shap_values,
                self.X_explain,
                feature_names=self.feature_names,
                show=False,
                max_display=20,
            )

            path = os.path.join(output_dir, "shap_beeswarm.png")
            plt.savefig(path, dpi=100, bbox_inches="tight")
            plt.close("all")
            self.logger.success(f"  Beeswarm plot saved: {path}")

        except Exception as e:
            self.logger.warning(f"  Beeswarm plot failed: {str(e)}")
            plt.close("all")

    def _plot_dependence(self, output_dir, top_n=4):
        """Dependence plots for top features."""
        try:
            self.logger.log(f"  Generating SHAP dependence plots (top {top_n})...")

            mean_shap = np.abs(self.shap_values).mean(axis=0)
            top_indices = np.argsort(mean_shap)[-top_n:][::-1]

            fig, axes = plt.subplots(1, top_n, figsize=(5 * top_n, 4))
            if top_n == 1:
                axes = [axes]

            for i, idx in enumerate(top_indices):
                feature_name = self.feature_names[idx]
                axes[i].scatter(
                    self.X_explain.iloc[:, idx],
                    self.shap_values[:, idx],
                    c=self.shap_values[:, idx],
                    cmap="RdBu_r",
                    alpha=0.5,
                    s=15,
                )
                axes[i].set_xlabel(feature_name)
                axes[i].set_ylabel(f"SHAP value for {feature_name}")
                axes[i].set_title(feature_name)
                axes[i].axhline(y=0, color="gray", linestyle="--", alpha=0.5)

            plt.suptitle("SHAP Dependence Plots (Top Features)", fontsize=14)
            plt.tight_layout()

            path = os.path.join(output_dir, "shap_dependence.png")
            plt.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            self.logger.success(f"  Dependence plots saved: {path}")

        except Exception as e:
            self.logger.warning(f"  Dependence plots failed: {str(e)}")
            plt.close("all")

    def _plot_waterfall(self, output_dir, sample_idx=0):
        """Waterfall plot for a single prediction."""
        try:
            self.logger.log("  Generating SHAP waterfall plot...")

            fig = plt.figure(figsize=(10, 6))

            if hasattr(self.explainer, "expected_value"):
                base_value = self.explainer.expected_value
                if isinstance(base_value, (list, np.ndarray)):
                    base_value = base_value[0] if len(base_value) > 0 else 0
            else:
                base_value = 0

            explanation = shap.Explanation(
                values=self.shap_values[sample_idx],
                base_values=base_value,
                data=self.X_explain.iloc[sample_idx].values,
                feature_names=self.feature_names,
            )

            shap.plots.waterfall(explanation, show=False, max_display=15)

            path = os.path.join(output_dir, "shap_waterfall.png")
            plt.savefig(path, dpi=100, bbox_inches="tight")
            plt.close("all")
            self.logger.success(f"  Waterfall plot saved: {path}")

        except Exception as e:
            self.logger.warning(f"  Waterfall plot failed: {str(e)}")
            plt.close("all")

    # ----------------------------------------------------------
    # 3. SAVE IMPORTANCE CSV
    # ----------------------------------------------------------
    def _save_importance_csv(self, output_dir):
        """Save SHAP-based feature importance as CSV."""
        try:
            mean_shap = np.abs(self.shap_values).mean(axis=0)
            importance_df = pd.DataFrame({
                "feature": self.feature_names,
                "mean_abs_shap": mean_shap,
            }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

            # Normalize
            total = importance_df["mean_abs_shap"].sum()
            if total > 0:
                importance_df["importance_pct"] = (importance_df["mean_abs_shap"] / total * 100).round(2)

            csv_path = os.path.join(output_dir, "shap_importance.csv")
            importance_df.to_csv(csv_path, index=False)
            self.logger.success(f"  SHAP importance saved: {csv_path}")

            # Store top features in report
            self.report["top_features"] = importance_df.head(10).to_dict("records")

        except Exception as e:
            self.logger.warning(f"  Importance CSV failed: {str(e)}")

    def _save_report(self, output_dir):
        """Save SHAP report."""
        report_path = os.path.join(output_dir, "shap_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(self.report, f, indent=2, default=str)
        self.logger.success(f"  SHAP report saved: {report_path}")

    # ----------------------------------------------------------
    # 4. PRINT SUMMARY
    # ----------------------------------------------------------
    def print_summary(self):
        """Print SHAP analysis summary."""
        print("\n" + "=" * 60)
        print("  SHAP EXPLAINABILITY SUMMARY")
        print("=" * 60)

        if not self.report:
            print("  No SHAP analysis performed.")
            print("=" * 60)
            return

        print(f"\n  Model Explained   : {self.report.get('model_name', 'N/A')}")
        print(f"  Problem Type      : {self.report.get('problem_type', 'N/A')}")
        print(f"  Samples Explained : {self.report.get('n_samples_explained', 'N/A')}")
        print(f"  Features          : {self.report.get('n_features', 'N/A')}")

        if "top_features" in self.report:
            print(f"\n  Top Features by SHAP Importance:")
            for item in self.report["top_features"][:10]:
                pct = item.get("importance_pct", 0)
                print(f"    {item['feature']:30s}: {item['mean_abs_shap']:.4f} ({pct:.1f}%)")

        print("=" * 60 + "\n")