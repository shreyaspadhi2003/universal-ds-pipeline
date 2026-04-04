"""
Final Comparison Module (Segment 12)
======================================
Handles:
  1. Collecting results from all modeling phases
  2. Creating unified comparison tables
  3. Grand ranking across all model types
  4. Final report generation
  5. Summary visualizations
"""

import os
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class FinalComparison:
    """Aggregates and compares results from all pipeline phases."""

    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        self.all_results = {}  # {phase_name: results_df}
        self.grand_ranking = None
        self.problem_type = None

    def set_problem_type(self, problem_type):
        self.problem_type = problem_type

    # ----------------------------------------------------------
    # 1. COLLECT RESULTS
    # ----------------------------------------------------------
    def add_results(self, phase_name, results_df):
        """Add results from a pipeline phase."""
        if results_df is not None and len(results_df) > 0:
            results_df = results_df.copy()
            results_df["phase"] = phase_name
            self.all_results[phase_name] = results_df
            self.logger.log(f"  Added {len(results_df)} results from '{phase_name}'")

    # ----------------------------------------------------------
    # 2. BUILD GRAND RANKING
    # ----------------------------------------------------------
    def build_grand_ranking(self):
        """
        Combine all results into a single ranked table.

        Returns:
            DataFrame with all models ranked by primary metric
        """
        self.logger.section("FINAL COMPARISON")

        if not self.all_results:
            self.logger.warning("No results to compare.")
            return pd.DataFrame()

        frames = []

        for phase, df in self.all_results.items():
            subset = df.copy()
            subset["phase"] = phase

            if self.problem_type == "regression":
                if "r2" in subset.columns:
                    subset = subset[subset["r2"].notna()]
                    keep_cols = ["model", "r2", "rmse", "mae", "train_time_sec", "phase"]
                    keep_cols = [c for c in keep_cols if c in subset.columns]
                    frames.append(subset[keep_cols])

            elif self.problem_type == "classification":
                if "f1_score" in subset.columns:
                    subset = subset[subset["f1_score"].notna()]
                    keep_cols = ["model", "accuracy", "precision", "recall",
                                 "f1_score", "auc_roc", "train_time_sec", "phase"]
                    keep_cols = [c for c in keep_cols if c in subset.columns]
                    frames.append(subset[keep_cols])

        if not frames:
            self.logger.warning("No valid results to combine.")
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)

        # Sort by primary metric
        if self.problem_type == "regression":
            combined = combined.sort_values("r2", ascending=False).reset_index(drop=True)
        else:
            combined = combined.sort_values("f1_score", ascending=False).reset_index(drop=True)

        combined.insert(0, "rank", range(1, len(combined) + 1))

        self.grand_ranking = combined
        self.logger.success(f"Grand ranking: {len(combined)} models from {len(self.all_results)} phases")

        return combined

    # ----------------------------------------------------------
    # 3. SAVE RESULTS
    # ----------------------------------------------------------
    def save_results(self, output_dir):
        """Save grand ranking, plots, and final report."""
        os.makedirs(output_dir, exist_ok=True)

        if self.grand_ranking is None or self.grand_ranking.empty:
            self.logger.warning("No grand ranking to save.")
            return

        # Save grand ranking CSV
        csv_path = os.path.join(output_dir, "grand_ranking.csv")
        self.grand_ranking.to_csv(csv_path, index=False)
        self.logger.success(f"Grand ranking saved to: {csv_path}")

        # Plots
        self._plot_grand_ranking(output_dir)
        self._plot_phase_comparison(output_dir)
        self._plot_top_models(output_dir)

        # Final report
        self._save_final_report(output_dir)

    def _plot_grand_ranking(self, output_dir):
        """Plot top 30 models across all phases."""
        top_n = min(30, len(self.grand_ranking))
        top = self.grand_ranking.head(top_n)

        if self.problem_type == "regression":
            metric_col = "r2"
            xlabel = "R2 Score"
        else:
            metric_col = "f1_score"
            xlabel = "F1 Score"

        fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.4)))

        # Color by phase
        phase_colors = {
            "regression": "steelblue",
            "classification": "steelblue",
            "ensemble": "coral",
            "deep_learning": "mediumpurple",
        }

        colors = [phase_colors.get(p, "gray") for p in top["phase"]]

        ax.barh(range(top_n), top[metric_col], color=colors, edgecolor="black", alpha=0.8)
        ax.set_yticks(range(top_n))

        labels = [f"[{row['phase'][:3].upper()}] {row['model']}" for _, row in top.iterrows()]
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_title(f"Grand Ranking - Top {top_n} Models (All Phases)")
        ax.invert_yaxis()

        # Legend
        from matplotlib.patches import Patch
        legend_items = []
        for phase, color in phase_colors.items():
            if phase in top["phase"].values:
                legend_items.append(Patch(facecolor=color, edgecolor="black", label=phase.title()))
        if legend_items:
            ax.legend(handles=legend_items, loc="lower right")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "grand_ranking.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)

    def _plot_phase_comparison(self, output_dir):
        """Box plot comparing metrics across phases."""
        if self.problem_type == "regression":
            metric_col = "r2"
            title = "R2 Distribution by Phase"
        else:
            metric_col = "f1_score"
            title = "F1 Score Distribution by Phase"

        if metric_col not in self.grand_ranking.columns:
            return

        phases = self.grand_ranking["phase"].unique()
        if len(phases) < 2:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        data_by_phase = [
            self.grand_ranking[self.grand_ranking["phase"] == p][metric_col].dropna().values
            for p in phases
        ]

        bp = ax.boxplot(data_by_phase, labels=phases, patch_artist=True)

        colors = ["steelblue", "coral", "mediumpurple", "seagreen", "orange"]
        for patch, color in zip(bp["boxes"], colors[:len(phases)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel(metric_col.upper())
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "phase_comparison.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)

    def _plot_top_models(self, output_dir):
        """Detailed comparison of top 5 models."""
        top_5 = self.grand_ranking.head(5)

        if self.problem_type == "regression":
            metrics = ["r2", "rmse", "mae"]
        else:
            metrics = ["f1_score", "accuracy", "precision", "recall"]

        available_metrics = [m for m in metrics if m in top_5.columns]
        if not available_metrics:
            return

        fig, axes = plt.subplots(1, len(available_metrics),
                                  figsize=(5 * len(available_metrics), 5))
        if len(available_metrics) == 1:
            axes = [axes]

        for i, metric in enumerate(available_metrics):
            values = top_5[metric].fillna(0)
            axes[i].barh(range(len(top_5)), values, color="steelblue", edgecolor="black")
            axes[i].set_yticks(range(len(top_5)))
            axes[i].set_yticklabels(top_5["model"], fontsize=9)
            axes[i].set_xlabel(metric.upper())
            axes[i].set_title(f"Top 5 - {metric.upper()}")
            axes[i].invert_yaxis()

        plt.suptitle("Top 5 Models - Detailed Metrics", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "top_5_detailed.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)

    def _save_final_report(self, output_dir):
        """Save comprehensive final report."""
        if self.grand_ranking is None or self.grand_ranking.empty:
            return

        best = self.grand_ranking.iloc[0]

        report = {
            "problem_type": self.problem_type,
            "total_models_evaluated": len(self.grand_ranking),
            "phases_completed": list(self.all_results.keys()),
            "models_per_phase": {
                phase: len(df) for phase, df in self.all_results.items()
            },
            "best_model": {
                "name": best["model"],
                "phase": best["phase"],
            },
        }

        if self.problem_type == "regression":
            report["best_model"]["r2"] = best.get("r2")
            report["best_model"]["rmse"] = best.get("rmse")
            report["best_model"]["mae"] = best.get("mae")

            # Phase bests
            report["best_per_phase"] = {}
            for phase, df in self.all_results.items():
                if "r2" in df.columns and df["r2"].notna().any():
                    phase_best = df.loc[df["r2"].idxmax()]
                    report["best_per_phase"][phase] = {
                        "model": phase_best["model"],
                        "r2": round(float(phase_best["r2"]), 6),
                    }

        elif self.problem_type == "classification":
            report["best_model"]["f1_score"] = best.get("f1_score")
            report["best_model"]["accuracy"] = best.get("accuracy")
            report["best_model"]["auc_roc"] = best.get("auc_roc")

            report["best_per_phase"] = {}
            for phase, df in self.all_results.items():
                if "f1_score" in df.columns and df["f1_score"].notna().any():
                    phase_best = df.loc[df["f1_score"].idxmax()]
                    report["best_per_phase"][phase] = {
                        "model": phase_best["model"],
                        "f1_score": round(float(phase_best["f1_score"]), 6),
                    }

        report_path = os.path.join(output_dir, "final_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        self.logger.success(f"Final report saved to: {report_path}")

    # ----------------------------------------------------------
    # 4. PRINT GRAND SUMMARY
    # ----------------------------------------------------------
    def print_grand_summary(self):
        """Print the ultimate summary."""
        print("\n")
        print("=" * 95)
        print("  " + "=" * 91)
        print("    GRAND SUMMARY - UNIVERSAL DATA SCIENCE PIPELINE")
        print("  " + "=" * 91)
        print("=" * 95)

        if self.grand_ranking is None or self.grand_ranking.empty:
            print("\n  No models to summarize.")
            print("=" * 95)
            return

        print(f"\n  Problem Type        : {self.problem_type}")
        print(f"  Total Models Tested : {len(self.grand_ranking)}")
        print(f"  Phases Completed    : {list(self.all_results.keys())}")

        # Models per phase
        print(f"\n  Models Per Phase:")
        for phase, df in self.all_results.items():
            print(f"    {phase:<25s}: {len(df)} models")

        # Grand ranking top 10
        top_10 = self.grand_ranking.head(10)
        if self.problem_type == "regression":
            print(f"\n  TOP 10 MODELS:")
            print(f"  {'Rank':<6}{'Model':<35}{'Phase':<18}{'R2':<12}{'RMSE':<14}")
            print("  " + "-" * 83)
            for _, row in top_10.iterrows():
                print(
                    f"  {row['rank']:<6}{row['model']:<35}{row['phase']:<18}"
                    f"{row['r2']:<12.4f}{row.get('rmse', 0):<14.4f}"
                )
        else:
            print(f"\n  TOP 10 MODELS:")
            print(f"  {'Rank':<6}{'Model':<35}{'Phase':<18}{'F1':<10}{'Accuracy':<12}")
            print("  " + "-" * 83)
            for _, row in top_10.iterrows():
                acc = row.get('accuracy', 0) or 0
                print(
                    f"  {row['rank']:<6}{row['model']:<35}{row['phase']:<18}"
                    f"{row['f1_score']:<10.4f}{acc:<12.4f}"
                )

        # Best model
        best = self.grand_ranking.iloc[0]
        print(f"\n  {'*' * 50}")
        print(f"  BEST MODEL: {best['model']}")
        print(f"  Phase     : {best['phase']}")
        if self.problem_type == "regression":
            print(f"  R2        : {best['r2']:.6f}")
            print(f"  RMSE      : {best.get('rmse', 'N/A')}")
        else:
            print(f"  F1 Score  : {best['f1_score']:.6f}")
            print(f"  Accuracy  : {best.get('accuracy', 'N/A')}")
        print(f"  {'*' * 50}")

        print("\n" + "=" * 95 + "\n")