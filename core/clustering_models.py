"""
Clustering Models Module (Segment 7)
======================================
Runs every suitable clustering model:
  - Partition: KMeans, Mini-Batch KMeans
  - Density: DBSCAN, OPTICS, Mean Shift
  - Hierarchical: Agglomerative (ward, complete, average)
  - Model-based: Gaussian Mixture
  - Other: Spectral, Birch, Affinity Propagation
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

from sklearn.cluster import (
    KMeans, MiniBatchKMeans, DBSCAN, OPTICS,
    AgglomerativeClustering, SpectralClustering,
    MeanShift, Birch, AffinityPropagation,
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class ClusteringModeler:
    """Trains and evaluates all suitable clustering models."""

    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        self.X = None
        self.results = []
        self.trained_models = {}
        self.cluster_labels = {}  # {model_name: labels}
        self.optimal_k = None

    # ----------------------------------------------------------
    # SETUP
    # ----------------------------------------------------------
    def set_data(self, df, feature_cols=None, target_col=None):
        """
        Prepare data for clustering.

        Args:
            df: cleaned DataFrame
            feature_cols: list of features (None = all numeric)
            target_col: excluded from features if present
        """
        self.logger.section("CLUSTERING MODELING")

        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != target_col]

        X = df[feature_cols].select_dtypes(include=[np.number]).copy()
        X = X.dropna()

        # Ensure data is scaled
        scaler = StandardScaler()
        self.X = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index,
        )

        self.logger.log(f"Clustering data: {self.X.shape[0]} samples x {self.X.shape[1]} features")

    # ----------------------------------------------------------
    # FIND OPTIMAL K (ELBOW METHOD)
    # ----------------------------------------------------------
    def find_optimal_k(self, max_k=15, output_dir=None):
        """
        Use elbow method and silhouette scores to find optimal k.

        Returns:
            optimal k value
        """
        self.logger.log("Finding optimal number of clusters...")

        max_k = min(max_k, len(self.X) - 1, 20)
        k_range = range(2, max_k + 1)

        inertias = []
        silhouettes = []

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.config.RANDOM_STATE, n_init=10)
            labels = km.fit_predict(self.X)
            inertias.append(km.inertia_)

            sil = silhouette_score(self.X, labels)
            silhouettes.append(sil)
            self.logger.log(f"  k={k}: inertia={km.inertia_:.2f}, silhouette={sil:.4f}")

        # Find optimal k: highest silhouette
        best_idx = np.argmax(silhouettes)
        self.optimal_k = list(k_range)[best_idx]

        self.logger.success(f"Optimal k = {self.optimal_k} (silhouette = {silhouettes[best_idx]:.4f})")

        # Plot elbow + silhouette
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Elbow plot
            axes[0].plot(list(k_range), inertias, "b-o", linewidth=2)
            axes[0].axvline(x=self.optimal_k, color="red", linestyle="--", alpha=0.7,
                            label=f"Optimal k={self.optimal_k}")
            axes[0].set_xlabel("Number of Clusters (k)")
            axes[0].set_ylabel("Inertia")
            axes[0].set_title("Elbow Method")
            axes[0].legend()

            # Silhouette plot
            axes[1].plot(list(k_range), silhouettes, "g-o", linewidth=2)
            axes[1].axvline(x=self.optimal_k, color="red", linestyle="--", alpha=0.7,
                            label=f"Optimal k={self.optimal_k}")
            axes[1].set_xlabel("Number of Clusters (k)")
            axes[1].set_ylabel("Silhouette Score")
            axes[1].set_title("Silhouette Analysis")
            axes[1].legend()

            plt.tight_layout()
            path = os.path.join(output_dir, "optimal_k_analysis.png")
            plt.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            self.logger.success(f"Optimal k plot saved to: {path}")

        return self.optimal_k

    # ----------------------------------------------------------
    # MODEL DEFINITIONS
    # ----------------------------------------------------------
    def _get_all_models(self):
        """Return dict of all clustering models suited to this dataset."""
        n_samples = len(self.X)
        k = self.optimal_k if self.optimal_k else 3

        models = {}

        # --- Partition-based ---
        models["KMeans"] = KMeans(
            n_clusters=k, random_state=self.config.RANDOM_STATE, n_init=10
        )
        models["Mini-Batch KMeans"] = MiniBatchKMeans(
            n_clusters=k, random_state=self.config.RANDOM_STATE, n_init=10
        )

        # KMeans with different k values for comparison
        for test_k in [2, 3, 4, 5, 6, 8]:
            if test_k != k and test_k < n_samples:
                models[f"KMeans (k={test_k})"] = KMeans(
                    n_clusters=test_k, random_state=self.config.RANDOM_STATE, n_init=10
                )

        # --- Density-based ---
        models["DBSCAN (eps=0.5)"] = DBSCAN(eps=0.5, min_samples=5)
        models["DBSCAN (eps=1.0)"] = DBSCAN(eps=1.0, min_samples=5)
        models["DBSCAN (eps=1.5)"] = DBSCAN(eps=1.5, min_samples=5)

        if n_samples <= 10000:
            models["OPTICS"] = OPTICS(min_samples=5)

        if n_samples <= 10000:
            models["Mean Shift"] = MeanShift()

        # --- Hierarchical ---
        if n_samples <= 15000:
            models["Agglomerative (Ward)"] = AgglomerativeClustering(
                n_clusters=k, linkage="ward"
            )
            models["Agglomerative (Complete)"] = AgglomerativeClustering(
                n_clusters=k, linkage="complete"
            )
            models["Agglomerative (Average)"] = AgglomerativeClustering(
                n_clusters=k, linkage="average"
            )

        # --- Spectral ---
        if n_samples <= 5000:
            models["Spectral Clustering"] = SpectralClustering(
                n_clusters=k, random_state=self.config.RANDOM_STATE,
                affinity="nearest_neighbors"
            )

        # --- Model-based ---
        models["Gaussian Mixture"] = GaussianMixture(
            n_components=k, random_state=self.config.RANDOM_STATE
        )

        # GMM with different component counts
        for test_k in [2, 3, 4, 5]:
            if test_k != k:
                models[f"Gaussian Mixture (k={test_k})"] = GaussianMixture(
                    n_components=test_k, random_state=self.config.RANDOM_STATE
                )

        # --- Birch ---
        models["Birch"] = Birch(n_clusters=k)

        # --- Affinity Propagation (small datasets) ---
        if n_samples <= 3000:
            models["Affinity Propagation"] = AffinityPropagation(
                random_state=self.config.RANDOM_STATE
            )

        # --- HDBSCAN (if installed) ---
        try:
            from sklearn.cluster import HDBSCAN
            models["HDBSCAN"] = HDBSCAN(min_cluster_size=10)
        except ImportError:
            try:
                import hdbscan
                models["HDBSCAN"] = hdbscan.HDBSCAN(min_cluster_size=10)
            except ImportError:
                self.logger.warning("HDBSCAN not available. Skipping.")

        return models

    # ----------------------------------------------------------
    # EVALUATION
    # ----------------------------------------------------------
    def _evaluate(self, model_name, labels, train_time):
        """Compute clustering evaluation metrics."""
        # Filter out noise labels (-1) for metrics
        valid_mask = labels >= 0
        n_clusters = len(set(labels[valid_mask]))
        n_noise = (labels == -1).sum()

        metrics = {
            "model": model_name,
            "n_clusters": n_clusters,
            "n_noise_points": int(n_noise),
            "train_time_sec": round(train_time, 4),
            "silhouette": None,
            "calinski_harabasz": None,
            "davies_bouldin": None,
        }

        # Need at least 2 clusters and some valid points for metrics
        if n_clusters < 2 or valid_mask.sum() < n_clusters + 1:
            self.logger.log(f"    Skipping metrics: {n_clusters} cluster(s) found")
            return metrics

        X_valid = self.X.values[valid_mask]
        labels_valid = labels[valid_mask]

        try:
            metrics["silhouette"] = round(silhouette_score(X_valid, labels_valid), 6)
        except Exception:
            pass

        try:
            metrics["calinski_harabasz"] = round(calinski_harabasz_score(X_valid, labels_valid), 4)
        except Exception:
            pass

        try:
            metrics["davies_bouldin"] = round(davies_bouldin_score(X_valid, labels_valid), 6)
        except Exception:
            pass

        return metrics

    # ----------------------------------------------------------
    # TRAIN ALL MODELS
    # ----------------------------------------------------------
    def train_all(self):
        """
        Train every suitable clustering model.

        Returns:
            DataFrame of results sorted by silhouette score
        """
        models = self._get_all_models()
        self.logger.log(f"Training {len(models)} clustering models...\n")

        self.results = []
        total = len(models)

        for i, (name, model) in enumerate(models.items(), 1):
            try:
                self.logger.log(f"  [{i}/{total}] {name}...")
                start = time.time()

                # Different models have different interfaces
                if hasattr(model, "fit_predict"):
                    labels = model.fit_predict(self.X)
                elif hasattr(model, "predict"):
                    model.fit(self.X)
                    labels = model.predict(self.X)
                else:
                    model.fit(self.X)
                    labels = model.labels_

                train_time = time.time() - start

                self.cluster_labels[name] = labels
                self.trained_models[name] = model

                metrics = self._evaluate(name, labels, train_time)
                self.results.append(metrics)

                sil_str = f"{metrics['silhouette']:.4f}" if metrics['silhouette'] is not None else "N/A"
                self.logger.log(
                    f"    Clusters={metrics['n_clusters']}, Silhouette={sil_str}, "
                    f"Time={metrics['train_time_sec']:.2f}s"
                )

            except Exception as e:
                self.logger.warning(f"    FAILED: {str(e)}")
                self.results.append({
                    "model": name,
                    "n_clusters": None,
                    "n_noise_points": None,
                    "train_time_sec": None,
                    "silhouette": None,
                    "calinski_harabasz": None,
                    "davies_bouldin": None,
                    "error": str(e),
                })

        results_df = pd.DataFrame(self.results)

        # Sort by silhouette (descending)
        successful = results_df[results_df["silhouette"].notna()].sort_values(
            "silhouette", ascending=False
        )
        failed_or_na = results_df[results_df["silhouette"].isna()]
        results_df = pd.concat([successful, failed_or_na]).reset_index(drop=True)

        self.logger.success(f"\nCompleted: {len(successful)} with valid metrics")

        if len(successful) > 0:
            best = successful.iloc[0]
            self.logger.success(
                f"Best model: {best['model']} "
                f"(Silhouette={best['silhouette']:.4f}, Clusters={best['n_clusters']})"
            )

        return results_df

    # ----------------------------------------------------------
    # SAVE RESULTS
    # ----------------------------------------------------------
    def save_results(self, results_df, output_dir):
        """Save comparison table, plots, and cluster visualizations."""
        trad_dir = os.path.join(output_dir, "traditional")
        os.makedirs(trad_dir, exist_ok=True)

        # Save CSV
        csv_path = os.path.join(trad_dir, "clustering_results.csv")
        results_df.to_csv(csv_path, index=False)
        self.logger.success(f"Results saved to: {csv_path}")

        successful = results_df[results_df["silhouette"].notna()].copy()
        if len(successful) > 0:
            self._plot_silhouette_comparison(successful, trad_dir)
            self._plot_cluster_visualizations(successful, trad_dir)
            self._plot_metric_comparison(successful, trad_dir)

        # Save best model summary
        if len(successful) > 0:
            best = successful.iloc[0]
            summary = {
                "best_model": best["model"],
                "n_clusters": int(best["n_clusters"]) if best["n_clusters"] is not None else None,
                "silhouette": best["silhouette"],
                "calinski_harabasz": best["calinski_harabasz"],
                "davies_bouldin": best["davies_bouldin"],
                "optimal_k": self.optimal_k,
                "total_models_tested": len(results_df),
                "models_with_valid_metrics": len(successful),
            }
            summary_path = os.path.join(trad_dir, "best_model_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, default=str)
            self.logger.success(f"Best model summary saved to: {summary_path}")

        return csv_path

    def _plot_silhouette_comparison(self, df, output_dir):
        """Bar chart of silhouette scores."""
        fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.35)))
        colors = ["green" if v > 0.5 else "steelblue" if v > 0.25 else "orange" if v > 0 else "red"
                   for v in df["silhouette"]]
        ax.barh(range(len(df)), df["silhouette"], color=colors, edgecolor="black")
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["model"], fontsize=9)
        ax.set_xlabel("Silhouette Score (higher is better)")
        ax.set_title("Clustering Models - Silhouette Score Comparison")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "silhouette_comparison.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)

    def _plot_cluster_visualizations(self, df, output_dir):
        """PCA 2D projection colored by cluster labels for top models."""
        # Need at least 2 features for PCA 2D
        if self.X.shape[1] < 2:
            self.logger.warning("Skipping cluster visualization: need at least 2 features for PCA.")
            return
        # PCA for 2D visualization
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(self.X)

        top_models = df.head(6)["model"].tolist()
        n_plots = len(top_models)
        cols = min(3, n_plots)
        rows = (n_plots + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        if n_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for i, name in enumerate(top_models):
            if name in self.cluster_labels and i < len(axes):
                labels = self.cluster_labels[name]
                scatter = axes[i].scatter(
                    X_2d[:, 0], X_2d[:, 1],
                    c=labels, cmap="Set1", alpha=0.6, s=15,
                )
                sil = df[df["model"] == name]["silhouette"].values[0]
                n_clust = df[df["model"] == name]["n_clusters"].values[0]
                axes[i].set_title(f"{name}\nClusters={n_clust}, Sil={sil:.4f}", fontsize=10)
                axes[i].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
                axes[i].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")

        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("Cluster Visualizations (PCA 2D Projection)", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "cluster_visualizations.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)

    def _plot_metric_comparison(self, df, output_dir):
        """Side-by-side comparison of all three clustering metrics."""
        fig, axes = plt.subplots(1, 3, figsize=(18, max(6, len(df) * 0.3)))

        # Silhouette (higher is better)
        axes[0].barh(range(len(df)), df["silhouette"], color="steelblue", edgecolor="black")
        axes[0].set_yticks(range(len(df)))
        axes[0].set_yticklabels(df["model"], fontsize=8)
        axes[0].set_title("Silhouette (higher = better)")
        axes[0].invert_yaxis()

        # Calinski-Harabasz (higher is better)
        ch_vals = df["calinski_harabasz"].fillna(0)
        axes[1].barh(range(len(df)), ch_vals, color="coral", edgecolor="black")
        axes[1].set_yticks(range(len(df)))
        axes[1].set_yticklabels(df["model"], fontsize=8)
        axes[1].set_title("Calinski-Harabasz (higher = better)")
        axes[1].invert_yaxis()

        # Davies-Bouldin (lower is better)
        db_vals = df["davies_bouldin"].fillna(999)
        sorted_db = df.sort_values("davies_bouldin", ascending=True)
        axes[2].barh(range(len(sorted_db)), sorted_db["davies_bouldin"].fillna(0),
                      color="seagreen", edgecolor="black")
        axes[2].set_yticks(range(len(sorted_db)))
        axes[2].set_yticklabels(sorted_db["model"], fontsize=8)
        axes[2].set_title("Davies-Bouldin (lower = better)")
        axes[2].invert_yaxis()

        plt.suptitle("Clustering Metrics Comparison", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "metric_comparison.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)

    # ----------------------------------------------------------
    # PRINT SUMMARY
    # ----------------------------------------------------------
    def print_results_summary(self, results_df):
        """Print ranking table."""
        print("\n" + "=" * 95)
        print("  CLUSTERING MODEL RESULTS")
        print("=" * 95)

        successful = results_df[results_df["silhouette"].notna()]

        if len(successful) == 0:
            print("  No models produced valid clustering metrics.")
            print("=" * 95)
            return

        print(f"\n  Optimal k (from elbow/silhouette): {self.optimal_k}")
        print(f"\n  {'Rank':<6}{'Model':<30}{'Clusters':<10}{'Silhouette':<14}"
              f"{'Calinski-H':<14}{'Davies-B':<14}{'Time(s)':<10}")
        print("  " + "-" * 93)

        for i, (_, row) in enumerate(successful.iterrows(), 1):
            ch_str = f"{row['calinski_harabasz']:.2f}" if row['calinski_harabasz'] is not None else "N/A"
            db_str = f"{row['davies_bouldin']:.4f}" if row['davies_bouldin'] is not None else "N/A"
            print(
                f"  {i:<6}{row['model']:<30}{row['n_clusters']:<10}"
                f"{row['silhouette']:<14.4f}{ch_str:<14}{db_str:<14}"
                f"{row['train_time_sec']:<10.3f}"
            )

        failed = results_df[results_df["silhouette"].isna()]
        if len(failed) > 0:
            has_error_col = "error" in failed.columns

            if has_error_col:
                errored = failed[failed["error"].notna()]
                no_metric = failed[failed["error"].isna()]
            else:
                errored = pd.DataFrame()
                no_metric = failed

            if len(no_metric) > 0:
                print(f"\n  Models with insufficient clusters for metrics ({len(no_metric)}):")
                for _, row in no_metric.iterrows():
                    n_c = row['n_clusters'] if row['n_clusters'] is not None else "?"
                    print(f"    - {row['model']}: {n_c} cluster(s)")

            if len(errored) > 0:
                print(f"\n  Failed Models ({len(errored)}):")
                for _, row in errored.iterrows():
                    print(f"    - {row['model']}: {row.get('error', 'Unknown')}")

        best = successful.iloc[0]
        print(f"\n  BEST MODEL: {best['model']}")
        print(f"  Clusters = {best['n_clusters']} | Silhouette = {best['silhouette']:.6f}")
        print("=" * 95 + "\n")