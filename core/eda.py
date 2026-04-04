"""
EDA Module (Segment 3)
=================================
Handles:
  1. Statistical summary (describe + skewness + kurtosis)
  2. Distribution plots for numeric columns
  3. Target variable analysis
  4. Correlation heatmap
  5. Pairwise scatter plots
  6. Aggregation (categorical vs target)
  7. Save all outputs
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
import json


class EDAAnalyzer:
    """Performs exploratory data analysis on the cleaned dataset."""

    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        self.df = None
        self.original_df = None  # Pre-cleaning version for raw distributions
        self.target_col = None
        self.problem_type = None
        self.column_types = {}
        self.eda_report = {}

    # ----------------------------------------------------------
    # SETUP
    # ----------------------------------------------------------
    def set_data(self, cleaned_df, original_df=None, column_types=None,
                 target_col=None, problem_type=None):
        """
        Set data for EDA.

        Args:
            cleaned_df: cleaned DataFrame (from Segment 2)
            original_df: pre-cleaning DataFrame (optional, for raw distributions)
            column_types: dict of column types
            target_col: target variable name
            problem_type: 'regression' or 'classification'
        """
        self.df = cleaned_df.copy()
        self.original_df = original_df.copy() if original_df is not None else None
        self.column_types = column_types or {}
        self.target_col = target_col
        self.problem_type = problem_type
        self.logger.section("EXPLORATORY DATA ANALYSIS")
        self.logger.log(f"Dataset: {self.df.shape[0]} rows x {self.df.shape[1]} columns")
        self.logger.log(f"Target: {self.target_col} ({self.problem_type})")

    # ----------------------------------------------------------
    # 1. STATISTICAL SUMMARY
    # ----------------------------------------------------------
    def generate_statistical_summary(self, output_dir=None):
        """
        Generate comprehensive statistical summary including
        describe(), skewness, and kurtosis.

        Returns:
            DataFrame with summary stats
        """
        self.logger.log("Generating statistical summary...")

        numeric_df = self.df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            self.logger.warning("No numeric columns for statistical summary.")
            return pd.DataFrame()

        # Basic describe
        summary = numeric_df.describe().T

        # Add skewness and kurtosis
        summary["skewness"] = numeric_df.skew()
        summary["kurtosis"] = numeric_df.kurtosis()

        # Add missing info
        summary["missing_count"] = self.df[numeric_df.columns].isnull().sum()
        summary["missing_pct"] = (summary["missing_count"] / len(self.df) * 100).round(2)

        # Interpret skewness
        def skew_label(val):
            if abs(val) < 0.5:
                return "symmetric"
            elif abs(val) < 1:
                return "moderate skew"
            else:
                return "high skew"

        summary["skew_interpretation"] = summary["skewness"].apply(skew_label)

        self.eda_report["statistical_summary"] = {
            "num_numeric_cols": len(numeric_df.columns),
            "columns": list(numeric_df.columns),
        }

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, "statistical_summary.csv")
            summary.to_csv(path)
            self.logger.success(f"Statistical summary saved to: {path}")

        self.logger.success(f"Summary generated for {len(numeric_df.columns)} numeric columns")
        return summary

    # ----------------------------------------------------------
    # 2. DISTRIBUTION PLOTS
    # ----------------------------------------------------------
    def plot_distributions(self, output_dir=None):
        """
        Plot histogram + KDE for every numeric column.
        Saves individual plots to output_dir/distributions/
        """
        self.logger.log("Plotting distributions...")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            self.logger.warning("No numeric columns for distribution plots.")
            return

        dist_dir = os.path.join(output_dir, "distributions") if output_dir else None
        if dist_dir:
            os.makedirs(dist_dir, exist_ok=True)

        plot_count = 0
        for col in numeric_cols:
            try:
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))

                # Histogram + KDE
                axes[0].hist(self.df[col].dropna(), bins=30, edgecolor="black",
                             alpha=0.7, color="steelblue")
                axes[0].set_title(f"{col} - Histogram")
                axes[0].set_xlabel(col)
                axes[0].set_ylabel("Frequency")

                # Box plot
                axes[1].boxplot(self.df[col].dropna(), vert=True)
                axes[1].set_title(f"{col} - Box Plot")
                axes[1].set_ylabel(col)

                plt.tight_layout()

                if dist_dir:
                    path = os.path.join(dist_dir, f"dist_{col}.png")
                    plt.savefig(path, dpi=100, bbox_inches="tight")
                    plot_count += 1

                plt.close(fig)
            except Exception as e:
                self.logger.warning(f"  Failed to plot '{col}': {str(e)}")
                plt.close("all")

        self.eda_report["distribution_plots"] = plot_count
        self.logger.success(f"Distribution plots saved: {plot_count}")

    # ----------------------------------------------------------
    # 3. TARGET VARIABLE ANALYSIS
    # ----------------------------------------------------------
    def analyze_target(self, output_dir=None):
        """
        Analyze the target variable:
        - Classification: class balance bar chart
        - Regression: distribution histogram
        """
        if not self.target_col or self.target_col not in self.df.columns:
            self.logger.log("No target variable set. Skipping target analysis.")
            return

        self.logger.log(f"Analyzing target variable: {self.target_col}...")

        fig, ax = plt.subplots(figsize=(8, 5))

        if self.problem_type == "classification":
            counts = self.df[self.target_col].value_counts()
            counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
            ax.set_title(f"Target: {self.target_col} - Class Distribution")
            ax.set_xlabel("Class")
            ax.set_ylabel("Count")

            # Add count labels on bars
            for i, (idx, val) in enumerate(counts.items()):
                ax.text(i, val + 0.5, str(val), ha="center", fontweight="bold")

            self.eda_report["target_analysis"] = {
                "type": "classification",
                "class_counts": counts.to_dict(),
                "num_classes": len(counts),
                "balance_ratio": round(counts.min() / counts.max(), 4),
            }

        elif self.problem_type == "regression":
            ax.hist(self.df[self.target_col].dropna(), bins=30,
                    edgecolor="black", alpha=0.7, color="steelblue")
            ax.set_title(f"Target: {self.target_col} - Distribution")
            ax.set_xlabel(self.target_col)
            ax.set_ylabel("Frequency")

            target_stats = self.df[self.target_col].describe().to_dict()
            self.eda_report["target_analysis"] = {
                "type": "regression",
                "stats": {k: round(v, 4) for k, v in target_stats.items()},
                "skewness": round(self.df[self.target_col].skew(), 4),
            }

        plt.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, "target_analysis.png")
            plt.savefig(path, dpi=100, bbox_inches="tight")
            self.logger.success(f"Target analysis saved to: {path}")

        plt.close(fig)

    # ----------------------------------------------------------
    # 4. CORRELATION HEATMAP
    # ----------------------------------------------------------
    def plot_correlation_heatmap(self, output_dir=None):
        """
        Plot correlation heatmap for all numeric columns.
        """
        self.logger.log("Generating correlation heatmap...")

        numeric_df = self.df.select_dtypes(include=[np.number])

        if numeric_df.shape[1] < 2:
            self.logger.warning("Need at least 2 numeric columns for correlation.")
            return None

        corr = numeric_df.corr()

        # Determine figure size based on number of columns
        n_cols = len(corr.columns)
        fig_size = max(8, n_cols * 0.6)

        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        # Use mask for upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        sns.heatmap(
            corr,
            mask=mask,
            annot=n_cols <= 20,  # Only show numbers if not too many columns
            fmt=".2f" if n_cols <= 20 else "",
            cmap="RdBu_r",
            center=0,
            square=True,
            linewidths=0.5,
            ax=ax,
            vmin=-1,
            vmax=1,
        )
        ax.set_title("Correlation Heatmap", fontsize=14)
        plt.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, "correlation_heatmap.png")
            plt.savefig(path, dpi=100, bbox_inches="tight")
            self.logger.success(f"Correlation heatmap saved to: {path}")

        plt.close(fig)

        # Find top correlations
        top_corr = self._get_top_correlations(corr, n=10)
        self.eda_report["correlation"] = {
            "top_positive": top_corr["positive"],
            "top_negative": top_corr["negative"],
        }

        return corr

    def _get_top_correlations(self, corr, n=10):
        """Extract top positive and negative correlations."""
        # Get upper triangle pairs
        pairs = []
        cols = corr.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                pairs.append({
                    "feature_1": cols[i],
                    "feature_2": cols[j],
                    "correlation": round(corr.iloc[i, j], 4),
                })

        pairs_sorted = sorted(pairs, key=lambda x: abs(x["correlation"]), reverse=True)

        positive = [p for p in pairs_sorted if p["correlation"] > 0][:n]
        negative = [p for p in pairs_sorted if p["correlation"] < 0][:n]

        return {"positive": positive, "negative": negative}

    # ----------------------------------------------------------
    # 5. PAIRWISE SCATTER PLOTS
    # ----------------------------------------------------------
    def plot_pairwise(self, output_dir=None, max_features=8):
        """
        Create pairwise scatter plot matrix for top numeric features.

        Args:
            max_features: limit number of features to avoid huge plots
        """
        self.logger.log(f"Generating pairwise plots (max {max_features} features)...")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        # Prioritize: target + highest variance columns
        if self.target_col and self.target_col in numeric_cols:
            # Put target first
            other_cols = [c for c in numeric_cols if c != self.target_col]
            # Sort by variance
            variances = self.df[other_cols].var().sort_values(ascending=False)
            selected = [self.target_col] + variances.index[:max_features - 1].tolist()
        else:
            variances = self.df[numeric_cols].var().sort_values(ascending=False)
            selected = variances.index[:max_features].tolist()

        if len(selected) < 2:
            self.logger.warning("Need at least 2 columns for pairwise plots.")
            return

        self.logger.log(f"  Using columns: {selected}")

        # Determine hue
        hue_col = None
        if self.target_col and self.target_col in self.df.columns:
            if self.problem_type == "classification" and self.df[self.target_col].nunique() <= 10:
                hue_col = self.target_col

        try:
            pair_grid = sns.pairplot(
                self.df[selected].dropna(),
                hue=hue_col,
                diag_kind="hist",
                plot_kws={"alpha": 0.5, "s": 15},
                height=2,
            )
            pair_grid.fig.suptitle("Pairwise Feature Relationships", y=1.02, fontsize=14)

            if output_dir:
                pair_dir = os.path.join(output_dir, "pairwise")
                os.makedirs(pair_dir, exist_ok=True)
                path = os.path.join(pair_dir, "pairwise_scatter.png")
                pair_grid.savefig(path, dpi=100, bbox_inches="tight")
                self.logger.success(f"Pairwise plot saved to: {path}")

            plt.close("all")
        except Exception as e:
            self.logger.warning(f"Pairwise plot failed: {str(e)}")
            plt.close("all")

        self.eda_report["pairwise_features"] = selected

    # ----------------------------------------------------------
    # 6. AGGREGATION: CATEGORICAL vs TARGET
    # ----------------------------------------------------------
    def aggregate_by_categories(self, output_dir=None):
        """
        For each categorical-like column, compute aggregated stats
        against the target variable and save summary + plot.
        """
        if not self.target_col or self.target_col not in self.df.columns:
            self.logger.log("No target variable. Skipping aggregation.")
            return {}

        self.logger.log("Aggregating categorical features vs target...")

        # Find columns with low cardinality (likely originally categorical)
        cat_like_cols = []
        for col in self.df.columns:
            if col == self.target_col:
                continue
            if self.df[col].nunique() <= 15 and self.df[col].nunique() >= 2:
                cat_like_cols.append(col)

        if not cat_like_cols:
            self.logger.log("  No suitable categorical columns for aggregation.")
            return {}

        agg_results = {}

        for col in cat_like_cols[:10]:  # Limit to 10 columns
            try:
                if self.problem_type == "regression":
                    agg = self.df.groupby(col)[self.target_col].agg(
                        ["mean", "median", "std", "count"]
                    ).round(4)
                elif self.problem_type == "classification":
                    agg = pd.crosstab(
                        self.df[col], self.df[self.target_col], normalize="index"
                    ).round(4)
                else:
                    continue

                agg_results[col] = agg

                # Plot
                fig, ax = plt.subplots(figsize=(8, 4))

                if self.problem_type == "regression":
                    agg["mean"].plot(kind="bar", ax=ax, color="steelblue",
                                     edgecolor="black")
                    ax.set_title(f"{col} vs {self.target_col} (Mean)")
                    ax.set_ylabel(f"Mean {self.target_col}")
                else:
                    agg.plot(kind="bar", stacked=True, ax=ax)
                    ax.set_title(f"{col} vs {self.target_col} (Proportions)")
                    ax.set_ylabel("Proportion")
                    ax.legend(title=self.target_col, bbox_to_anchor=(1.05, 1))

                ax.set_xlabel(col)
                plt.xticks(rotation=45)
                plt.tight_layout()

                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    path = os.path.join(output_dir, f"agg_{col}_vs_{self.target_col}.png")
                    plt.savefig(path, dpi=100, bbox_inches="tight")

                plt.close(fig)
                self.logger.log(f"  {col}: aggregated ({len(agg)} groups)")

            except Exception as e:
                self.logger.warning(f"  Aggregation failed for '{col}': {str(e)}")
                plt.close("all")

        self.eda_report["aggregation_columns"] = list(agg_results.keys())
        self.logger.success(f"Aggregation complete: {len(agg_results)} columns analyzed")
        return agg_results

    # ----------------------------------------------------------
    # 7. RUN FULL EDA
    # ----------------------------------------------------------
    def run_full_eda(self, output_dir=None):
        """
        Run all EDA steps and save everything.

        Args:
            output_dir: base EDA output directory
        """
        self.logger.section("FULL EDA PIPELINE")

        # 1. Statistical summary
        summary = self.generate_statistical_summary(output_dir)

        # 2. Distribution plots
        self.plot_distributions(output_dir)

        # 3. Target analysis
        self.analyze_target(output_dir)

        # 4. Correlation heatmap
        corr = self.plot_correlation_heatmap(output_dir)

        # 5. Pairwise plots
        self.plot_pairwise(output_dir)

        # 6. Aggregation
        agg = self.aggregate_by_categories(output_dir)

        # Save EDA report
        if output_dir:
            report_path = os.path.join(output_dir, "eda_report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(self.eda_report, f, indent=2, default=str)
            self.logger.success(f"EDA report saved to: {report_path}")

        self.logger.success("Full EDA complete")
        return {
            "summary": summary,
            "correlation": corr,
            "aggregation": agg,
        }

    # ----------------------------------------------------------
    # 8. PRINT EDA SUMMARY
    # ----------------------------------------------------------
    def print_eda_summary(self):
        """Print a human-readable EDA summary."""
        print("\n" + "=" * 60)
        print("  EDA SUMMARY")
        print("=" * 60)

        if "statistical_summary" in self.eda_report:
            info = self.eda_report["statistical_summary"]
            print(f"\n  Numeric Columns Analyzed: {info['num_numeric_cols']}")

        if "target_analysis" in self.eda_report:
            ta = self.eda_report["target_analysis"]
            print(f"\n  Target Variable Analysis:")
            print(f"    Type: {ta['type']}")
            if ta["type"] == "classification":
                print(f"    Classes: {ta['num_classes']}")
                print(f"    Balance Ratio: {ta['balance_ratio']}")
                for cls, cnt in ta.get("class_counts", {}).items():
                    print(f"      {cls}: {cnt}")
            elif ta["type"] == "regression":
                print(f"    Skewness: {ta.get('skewness', 'N/A')}")

        if "correlation" in self.eda_report:
            corr_info = self.eda_report["correlation"]
            if corr_info.get("top_positive"):
                print(f"\n  Top Positive Correlations:")
                for p in corr_info["top_positive"][:5]:
                    print(f"    {p['feature_1']} <-> {p['feature_2']}: {p['correlation']}")
            if corr_info.get("top_negative"):
                print(f"\n  Top Negative Correlations:")
                for p in corr_info["top_negative"][:5]:
                    print(f"    {p['feature_1']} <-> {p['feature_2']}: {p['correlation']}")

        if "distribution_plots" in self.eda_report:
            print(f"\n  Distribution Plots Generated: {self.eda_report['distribution_plots']}")

        if "pairwise_features" in self.eda_report:
            print(f"  Pairwise Features: {self.eda_report['pairwise_features']}")

        if "aggregation_columns" in self.eda_report:
            print(f"  Aggregation Columns: {self.eda_report['aggregation_columns']}")

        print("=" * 60 + "\n")