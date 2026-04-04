"""
Anomaly Detection Module (Segment 10)
=======================================
Handles:
  1. Statistical methods: IQR, Z-score, Modified Z-score
  2. ML methods: Isolation Forest, LOF, One-Class SVM, Elliptic Envelope
  3. Scoring and flagging anomalies
  4. Comparing methods
  5. Visualization on PCA projection
  6. Saving results
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
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class AnomalyDetector:
    """Detects anomalies using multiple methods and compares results."""

    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        self.X = None
        self.X_scaled = None
        self.feature_names = []
        self.results = {}          # {method_name: {"labels": array, "scores": array, ...}}
        self.summary = []          # list of dicts for comparison table
        self.report = {}

    # ----------------------------------------------------------
    # SETUP
    # ----------------------------------------------------------
    def set_data(self, df, feature_cols=None, target_col=None):
        """
        Prepare data for anomaly detection.

        Args:
            df: cleaned DataFrame
            feature_cols: columns to use (None = all numeric except target)
            target_col: excluded from features
        """
        self.logger.section("ANOMALY DETECTION")

        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != target_col]

        X = df[feature_cols].select_dtypes(include=[np.number]).copy()
        X = X.dropna()

        self.feature_names = list(X.columns)
        self.X = X

        scaler = StandardScaler()
        self.X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index,
        )

        self.logger.log(f"Anomaly detection data: {self.X.shape[0]} samples x {self.X.shape[1]} features")

    # ----------------------------------------------------------
    # 1. STATISTICAL METHODS
    # ----------------------------------------------------------
    def run_iqr_method(self, multiplier=1.5):
        """IQR-based anomaly detection across all features."""
        name = f"IQR (x{multiplier})"
        self.logger.log(f"Running {name}...")

        start = time.time()
        anomaly_mask = np.zeros(len(self.X), dtype=bool)

        for col in self.X.columns:
            Q1 = self.X[col].quantile(0.25)
            Q3 = self.X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - multiplier * IQR
            upper = Q3 + multiplier * IQR
            col_anomalies = (self.X[col] < lower) | (self.X[col] > upper)
            anomaly_mask = anomaly_mask | col_anomalies.values

        labels = np.where(anomaly_mask, -1, 1)
        train_time = time.time() - start

        n_anomalies = (labels == -1).sum()
        pct = n_anomalies / len(labels) * 100

        self.results[name] = {
            "labels": labels,
            "scores": None,
            "n_anomalies": int(n_anomalies),
            "pct_anomalies": round(pct, 2),
            "train_time": round(train_time, 4),
        }
        self.summary.append({
            "method": name,
            "n_anomalies": int(n_anomalies),
            "pct_anomalies": round(pct, 2),
            "train_time_sec": round(train_time, 4),
        })

        self.logger.log(f"  {name}: {n_anomalies} anomalies ({pct:.2f}%)")

    def run_zscore_method(self, threshold=3.0):
        """Z-score based anomaly detection."""
        name = f"Z-Score (>{threshold})"
        self.logger.log(f"Running {name}...")

        start = time.time()
        z_scores = np.abs(self.X_scaled.values)
        anomaly_mask = (z_scores > threshold).any(axis=1)
        labels = np.where(anomaly_mask, -1, 1)
        max_z = z_scores.max(axis=1)
        train_time = time.time() - start

        n_anomalies = (labels == -1).sum()
        pct = n_anomalies / len(labels) * 100

        self.results[name] = {
            "labels": labels,
            "scores": max_z,
            "n_anomalies": int(n_anomalies),
            "pct_anomalies": round(pct, 2),
            "train_time": round(train_time, 4),
        }
        self.summary.append({
            "method": name,
            "n_anomalies": int(n_anomalies),
            "pct_anomalies": round(pct, 2),
            "train_time_sec": round(train_time, 4),
        })

        self.logger.log(f"  {name}: {n_anomalies} anomalies ({pct:.2f}%)")

    def run_modified_zscore(self, threshold=3.5):
        """Modified Z-score using median and MAD."""
        name = f"Modified Z-Score (>{threshold})"
        self.logger.log(f"Running {name}...")

        start = time.time()
        anomaly_mask = np.zeros(len(self.X), dtype=bool)

        for col in self.X.columns:
            median = self.X[col].median()
            mad = np.median(np.abs(self.X[col] - median))
            if mad == 0:
                continue
            modified_z = 0.6745 * (self.X[col] - median) / mad
            col_anomalies = np.abs(modified_z) > threshold
            anomaly_mask = anomaly_mask | col_anomalies.values

        labels = np.where(anomaly_mask, -1, 1)
        train_time = time.time() - start

        n_anomalies = (labels == -1).sum()
        pct = n_anomalies / len(labels) * 100

        self.results[name] = {
            "labels": labels,
            "scores": None,
            "n_anomalies": int(n_anomalies),
            "pct_anomalies": round(pct, 2),
            "train_time": round(train_time, 4),
        }
        self.summary.append({
            "method": name,
            "n_anomalies": int(n_anomalies),
            "pct_anomalies": round(pct, 2),
            "train_time_sec": round(train_time, 4),
        })

        self.logger.log(f"  {name}: {n_anomalies} anomalies ({pct:.2f}%)")

    # ----------------------------------------------------------
    # 2. ML METHODS
    # ----------------------------------------------------------
    def run_isolation_forest(self, contamination=0.05):
        """Isolation Forest anomaly detection."""
        for cont in [0.01, 0.05, 0.1]:
            name = f"Isolation Forest (c={cont})"
            self.logger.log(f"Running {name}...")

            try:
                start = time.time()
                model = IsolationForest(
                    contamination=cont,
                    random_state=self.config.RANDOM_STATE,
                    n_jobs=-1,
                )
                labels = model.fit_predict(self.X_scaled)
                scores = model.decision_function(self.X_scaled)
                train_time = time.time() - start

                n_anomalies = (labels == -1).sum()
                pct = n_anomalies / len(labels) * 100

                self.results[name] = {
                    "labels": labels,
                    "scores": scores,
                    "n_anomalies": int(n_anomalies),
                    "pct_anomalies": round(pct, 2),
                    "train_time": round(train_time, 4),
                }
                self.summary.append({
                    "method": name,
                    "n_anomalies": int(n_anomalies),
                    "pct_anomalies": round(pct, 2),
                    "train_time_sec": round(train_time, 4),
                })

                self.logger.log(f"  {name}: {n_anomalies} anomalies ({pct:.2f}%)")
            except Exception as e:
                self.logger.warning(f"  {name} failed: {str(e)}")

    def run_local_outlier_factor(self):
        """Local Outlier Factor anomaly detection."""
        for n_neighbors in [10, 20, 30]:
            name = f"LOF (k={n_neighbors})"
            self.logger.log(f"Running {name}...")

            try:
                start = time.time()
                model = LocalOutlierFactor(
                    n_neighbors=n_neighbors,
                    contamination=0.05,
                    n_jobs=-1,
                )
                labels = model.fit_predict(self.X_scaled)
                scores = model.negative_outlier_factor_
                train_time = time.time() - start

                n_anomalies = (labels == -1).sum()
                pct = n_anomalies / len(labels) * 100

                self.results[name] = {
                    "labels": labels,
                    "scores": scores,
                    "n_anomalies": int(n_anomalies),
                    "pct_anomalies": round(pct, 2),
                    "train_time": round(train_time, 4),
                }
                self.summary.append({
                    "method": name,
                    "n_anomalies": int(n_anomalies),
                    "pct_anomalies": round(pct, 2),
                    "train_time_sec": round(train_time, 4),
                })

                self.logger.log(f"  {name}: {n_anomalies} anomalies ({pct:.2f}%)")
            except Exception as e:
                self.logger.warning(f"  {name} failed: {str(e)}")

    def run_one_class_svm(self):
        """One-Class SVM anomaly detection."""
        n_samples = len(self.X_scaled)
        if n_samples > 10000:
            self.logger.warning("  One-Class SVM skipped: dataset too large (>10k rows)")
            return

        name = "One-Class SVM"
        self.logger.log(f"Running {name}...")

        try:
            start = time.time()
            model = OneClassSVM(kernel="rbf", gamma="auto", nu=0.05)
            labels = model.fit_predict(self.X_scaled)
            scores = model.decision_function(self.X_scaled)
            train_time = time.time() - start

            n_anomalies = (labels == -1).sum()
            pct = n_anomalies / len(labels) * 100

            self.results[name] = {
                "labels": labels,
                "scores": scores,
                "n_anomalies": int(n_anomalies),
                "pct_anomalies": round(pct, 2),
                "train_time": round(train_time, 4),
            }
            self.summary.append({
                "method": name,
                "n_anomalies": int(n_anomalies),
                "pct_anomalies": round(pct, 2),
                "train_time_sec": round(train_time, 4),
            })

            self.logger.log(f"  {name}: {n_anomalies} anomalies ({pct:.2f}%)")
        except Exception as e:
            self.logger.warning(f"  {name} failed: {str(e)}")

    def run_elliptic_envelope(self):
        """Elliptic Envelope (robust covariance) anomaly detection."""
        name = "Elliptic Envelope"
        self.logger.log(f"Running {name}...")

        try:
            start = time.time()
            model = EllipticEnvelope(
                contamination=0.05,
                random_state=self.config.RANDOM_STATE,
            )
            labels = model.fit_predict(self.X_scaled)
            scores = model.decision_function(self.X_scaled)
            train_time = time.time() - start

            n_anomalies = (labels == -1).sum()
            pct = n_anomalies / len(labels) * 100

            self.results[name] = {
                "labels": labels,
                "scores": scores,
                "n_anomalies": int(n_anomalies),
                "pct_anomalies": round(pct, 2),
                "train_time": round(train_time, 4),
            }
            self.summary.append({
                "method": name,
                "n_anomalies": int(n_anomalies),
                "pct_anomalies": round(pct, 2),
                "train_time_sec": round(train_time, 4),
            })

            self.logger.log(f"  {name}: {n_anomalies} anomalies ({pct:.2f}%)")
        except Exception as e:
            self.logger.warning(f"  {name} failed: {str(e)}")

    # ----------------------------------------------------------
    # 3. RUN ALL METHODS
    # ----------------------------------------------------------
    def run_all(self):
        """
        Run every anomaly detection method.

        Returns:
            DataFrame comparing all methods
        """
        self.logger.log("Running all anomaly detection methods...\n")

        # Statistical
        self.run_iqr_method(multiplier=1.5)
        self.run_iqr_method(multiplier=2.0)
        self.run_zscore_method(threshold=3.0)
        self.run_zscore_method(threshold=2.5)
        self.run_modified_zscore(threshold=3.5)

        # ML
        self.run_isolation_forest()
        self.run_local_outlier_factor()
        self.run_one_class_svm()
        self.run_elliptic_envelope()

        summary_df = pd.DataFrame(self.summary)
        summary_df = summary_df.sort_values("n_anomalies").reset_index(drop=True)

        self.logger.success(f"\nAll methods complete: {len(summary_df)} methods ran")

        # Consensus anomalies (flagged by majority of methods)
        self._compute_consensus()

        return summary_df

    def _compute_consensus(self):
        """Find points flagged as anomalies by majority of methods."""
        if not self.results:
            return

        n_methods = len(self.results)
        n_samples = len(self.X)
        vote_matrix = np.zeros(n_samples, dtype=int)

        for method_name, result in self.results.items():
            labels = result["labels"]
            if len(labels) == n_samples:
                vote_matrix += (labels == -1).astype(int)

        # Consensus: flagged by more than half the methods
        threshold = max(1, n_methods // 2)
        consensus_anomalies = vote_matrix >= threshold

        n_consensus = consensus_anomalies.sum()
        self.report["consensus"] = {
            "threshold": f">={threshold}/{n_methods} methods",
            "n_anomalies": int(n_consensus),
            "pct_anomalies": round(n_consensus / n_samples * 100, 2),
            "indices": np.where(consensus_anomalies)[0].tolist()[:50],  # first 50
        }

        self.logger.success(
            f"Consensus anomalies: {n_consensus} points flagged by >= {threshold}/{n_methods} methods"
        )

    # ----------------------------------------------------------
    # 4. SAVE RESULTS
    # ----------------------------------------------------------
    def save_results(self, output_dir):
        """Save comparison table, anomaly flags, and visualizations."""
        os.makedirs(output_dir, exist_ok=True)

        # Save comparison CSV
        summary_df = pd.DataFrame(self.summary)
        csv_path = os.path.join(output_dir, "anomaly_comparison.csv")
        summary_df.to_csv(csv_path, index=False)
        self.logger.success(f"Comparison saved to: {csv_path}")

        # Save anomaly flags per method
        flags_df = self.X.copy()
        for method_name, result in self.results.items():
            safe_name = method_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace(">", "gt")
            labels = result["labels"]
            if len(labels) == len(flags_df):
                flags_df[f"anomaly_{safe_name}"] = (labels == -1).astype(int)

        flags_path = os.path.join(output_dir, "anomaly_flags.csv")
        flags_df.to_csv(flags_path, index=False)
        self.logger.success(f"Anomaly flags saved to: {flags_path}")

        # Generate plots
        self._plot_comparison(output_dir)
        self._plot_pca_anomalies(output_dir)

        # Save report
        self.report["methods"] = self.summary
        report_path = os.path.join(output_dir, "anomaly_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(self.report, f, indent=2, default=str)
        self.logger.success(f"Report saved to: {report_path}")

    def _plot_comparison(self, output_dir):
        """Bar chart comparing anomaly counts across methods."""
        summary_df = pd.DataFrame(self.summary).sort_values("n_anomalies")

        fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(summary_df) * 0.4)))

        # Anomaly count
        axes[0].barh(range(len(summary_df)), summary_df["n_anomalies"],
                      color="coral", edgecolor="black")
        axes[0].set_yticks(range(len(summary_df)))
        axes[0].set_yticklabels(summary_df["method"], fontsize=9)
        axes[0].set_xlabel("Number of Anomalies")
        axes[0].set_title("Anomalies Detected by Method")

        # Percentage
        axes[1].barh(range(len(summary_df)), summary_df["pct_anomalies"],
                      color="steelblue", edgecolor="black")
        axes[1].set_yticks(range(len(summary_df)))
        axes[1].set_yticklabels(summary_df["method"], fontsize=9)
        axes[1].set_xlabel("Anomaly Percentage (%)")
        axes[1].set_title("Anomaly Percentage by Method")

        plt.suptitle("Anomaly Detection Method Comparison", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "anomaly_comparison.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)

    def _plot_pca_anomalies(self, output_dir):
        """PCA 2D projection showing anomalies for top methods."""
        if self.X_scaled.shape[1] < 2:
            self.logger.warning("Skipping PCA anomaly plot: need at least 2 features.")
            return
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(self.X_scaled)

        # Pick up to 6 methods
        methods_to_plot = list(self.results.keys())[:6]
        n_plots = len(methods_to_plot)
        cols = min(3, n_plots)
        rows = (n_plots + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        if n_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for i, name in enumerate(methods_to_plot):
            if i >= len(axes):
                break
            result = self.results[name]
            labels = result["labels"]

            if len(labels) != len(X_2d):
                continue

            normal = labels == 1
            anomaly = labels == -1

            axes[i].scatter(X_2d[normal, 0], X_2d[normal, 1],
                            c="steelblue", alpha=0.4, s=10, label="Normal")
            axes[i].scatter(X_2d[anomaly, 0], X_2d[anomaly, 1],
                            c="red", alpha=0.8, s=30, marker="x", label="Anomaly")
            axes[i].set_title(f"{name}\n({result['n_anomalies']} anomalies)", fontsize=10)
            axes[i].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
            axes[i].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
            axes[i].legend(fontsize=8)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("Anomaly Detection (PCA 2D Projection)", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "anomaly_pca.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)

    # ----------------------------------------------------------
    # 5. PRINT SUMMARY
    # ----------------------------------------------------------
    def print_results_summary(self):
        """Print comparison table."""
        print("\n" + "=" * 75)
        print("  ANOMALY DETECTION RESULTS")
        print("=" * 75)

        if not self.summary:
            print("  No methods ran.")
            print("=" * 75)
            return

        print(f"\n  {'Method':<35}{'Anomalies':<12}{'Pct(%)':<10}{'Time(s)':<10}")
        print("  " + "-" * 70)

        sorted_summary = sorted(self.summary, key=lambda x: x["n_anomalies"])
        for row in sorted_summary:
            print(
                f"  {row['method']:<35}{row['n_anomalies']:<12}"
                f"{row['pct_anomalies']:<10.2f}{row['train_time_sec']:<10.4f}"
            )

        if "consensus" in self.report:
            c = self.report["consensus"]
            print(f"\n  CONSENSUS ({c['threshold']}):")
            print(f"  {c['n_anomalies']} anomalies ({c['pct_anomalies']:.2f}%)")

        print("=" * 75 + "\n")