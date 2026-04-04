"""
Classification Models Module (Segment 6)
==========================================
Runs every suitable classification model:
  - Linear: Logistic Regression, Ridge, SGD, Passive Aggressive, Perceptron
  - SVM: SVC (linear, rbf, poly), LinearSVC
  - Tree: Decision Tree, Random Forest, Extra Trees, Gradient Boosting, AdaBoost, Bagging
  - Boosting: XGBoost, LightGBM, CatBoost
  - Neighbors: KNN
  - Naive Bayes: Gaussian, Multinomial, Bernoulli, Complement
  - Discriminant: LDA, QDA
  - Neural Network: MLP
  - Other: Nearest Centroid, Gaussian Process
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

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.preprocessing import label_binarize

from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier, SGDClassifier,
    PassiveAggressiveClassifier, Perceptron,
)
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier,
    BaggingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

warnings.filterwarnings("ignore")


class ClassificationModeler:
    """Trains and evaluates all suitable classification models."""

    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results = []
        self.trained_models = {}
        self.n_classes = 0
        self.class_labels = []

    # ----------------------------------------------------------
    # SETUP
    # ----------------------------------------------------------
    def set_data(self, df, target_col, feature_cols=None):
        """
        Prepare train/test split for classification.

        Args:
            df: cleaned DataFrame
            target_col: target variable
            feature_cols: list of feature columns (None = all except target)
        """
        self.logger.section("CLASSIFICATION MODELING")

        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != target_col]

        X = df[feature_cols].select_dtypes(include=[np.number])
        y = df[target_col]

        valid = X.notna().all(axis=1) & y.notna()
        X = X[valid]
        y = y[valid]

        self.n_classes = y.nunique()
        self.class_labels = sorted(y.unique())

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=y,
        )

        self.logger.log(f"Train: {self.X_train.shape}, Test: {self.X_test.shape}")
        self.logger.log(f"Classes: {self.n_classes} -> {self.class_labels}")
        self.logger.log(f"Features: {list(X.columns)}")

    # ----------------------------------------------------------
    # MODEL DEFINITIONS
    # ----------------------------------------------------------
    def _get_all_models(self):
        """Return dict of all classification models suited to this dataset."""
        n_samples = len(self.X_train)
        n_features = self.X_train.shape[1]
        is_binary = self.n_classes == 2

        models = {}

        # --- Linear Models ---
        models["Logistic Regression"] = LogisticRegression(
            max_iter=2000, random_state=self.config.RANDOM_STATE, n_jobs=-1
        )
        models["Ridge Classifier"] = RidgeClassifier(
            random_state=self.config.RANDOM_STATE
        )
        models["SGD Classifier"] = SGDClassifier(
            random_state=self.config.RANDOM_STATE, max_iter=1000
        )
        models["Passive Aggressive"] = PassiveAggressiveClassifier(
            random_state=self.config.RANDOM_STATE, max_iter=1000
        )
        models["Perceptron"] = Perceptron(
            random_state=self.config.RANDOM_STATE, max_iter=1000
        )

        # --- SVM ---
        if n_samples <= 10000:
            models["SVC (RBF)"] = SVC(
                kernel="rbf", random_state=self.config.RANDOM_STATE, probability=True
            )
            models["SVC (Poly)"] = SVC(
                kernel="poly", degree=3, random_state=self.config.RANDOM_STATE, probability=True
            )
            models["SVC (Linear)"] = SVC(
                kernel="linear", random_state=self.config.RANDOM_STATE, probability=True
            )
        models["Linear SVC"] = LinearSVC(
            random_state=self.config.RANDOM_STATE, max_iter=5000, dual="auto"
        )

        # --- Tree-Based ---
        models["Decision Tree"] = DecisionTreeClassifier(
            random_state=self.config.RANDOM_STATE, max_depth=10
        )
        models["Random Forest"] = RandomForestClassifier(
            n_estimators=100, random_state=self.config.RANDOM_STATE, n_jobs=-1
        )
        models["Extra Trees"] = ExtraTreesClassifier(
            n_estimators=100, random_state=self.config.RANDOM_STATE, n_jobs=-1
        )
        models["Gradient Boosting"] = GradientBoostingClassifier(
            n_estimators=100, random_state=self.config.RANDOM_STATE
        )
        models["AdaBoost"] = AdaBoostClassifier(
            n_estimators=100, random_state=self.config.RANDOM_STATE
        )
        models["Bagging"] = BaggingClassifier(
            n_estimators=100, random_state=self.config.RANDOM_STATE, n_jobs=-1
        )

        # --- KNN ---
        models["KNN (k=5)"] = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        models["KNN (k=10)"] = KNeighborsClassifier(n_neighbors=10, n_jobs=-1)
        models["KNN (k=3)"] = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)

        # --- Naive Bayes ---
        models["Gaussian NB"] = GaussianNB()
        models["Bernoulli NB"] = BernoulliNB()
        models["Complement NB"] = ComplementNB()

        # --- Discriminant Analysis ---
        models["LDA"] = LinearDiscriminantAnalysis()
        if n_features < n_samples:
            models["QDA"] = QuadraticDiscriminantAnalysis()

        # --- Nearest Centroid ---
        models["Nearest Centroid"] = NearestCentroid()

        # --- Neural Network ---
        models["MLP Classifier"] = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=self.config.RANDOM_STATE,
            early_stopping=True,
        )

        # --- Gaussian Process (small datasets only) ---
        if n_samples <= self.config.MAX_ROWS_FOR_GP:
            models["Gaussian Process"] = GaussianProcessClassifier(
                random_state=self.config.RANDOM_STATE
            )

        # --- XGBoost ---
        try:
            from xgboost import XGBClassifier
            models["XGBoost"] = XGBClassifier(
                n_estimators=100,
                random_state=self.config.RANDOM_STATE,
                verbosity=0,
                n_jobs=-1,
                eval_metric="logloss",
            )
        except ImportError:
            self.logger.warning("XGBoost not installed. Skipping.")

        # --- LightGBM ---
        try:
            from lightgbm import LGBMClassifier
            models["LightGBM"] = LGBMClassifier(
                n_estimators=100,
                random_state=self.config.RANDOM_STATE,
                verbose=-1,
                n_jobs=-1,
            )
        except ImportError:
            self.logger.warning("LightGBM not installed. Skipping.")

        # --- CatBoost ---
        try:
            from catboost import CatBoostClassifier
            models["CatBoost"] = CatBoostClassifier(
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
        """Compute all classification metrics."""
        y_pred = model.predict(self.X_test)

        # Average method for multiclass
        avg = "binary" if self.n_classes == 2 else "weighted"

        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average=avg, zero_division=0)
        recall = recall_score(self.y_test, y_pred, average=avg, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average=avg, zero_division=0)

        # AUC-ROC
        auc = None
        try:
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(self.X_test)
                if self.n_classes == 2:
                    auc = roc_auc_score(self.y_test, y_proba[:, 1])
                else:
                    y_bin = label_binarize(self.y_test, classes=self.class_labels)
                    auc = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="weighted")
            elif hasattr(model, "decision_function"):
                y_dec = model.decision_function(self.X_test)
                if self.n_classes == 2:
                    auc = roc_auc_score(self.y_test, y_dec)
                else:
                    y_bin = label_binarize(self.y_test, classes=self.class_labels)
                    if y_dec.ndim == 1:
                        auc = None
                    else:
                        auc = roc_auc_score(y_bin, y_dec, multi_class="ovr", average="weighted")
        except Exception:
            auc = None

        return {
            "model": model_name,
            "accuracy": round(accuracy, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1_score": round(f1, 6),
            "auc_roc": round(auc, 6) if auc is not None else None,
            "train_time_sec": round(train_time, 4),
        }

    # ----------------------------------------------------------
    # TRAIN ALL MODELS
    # ----------------------------------------------------------
    def train_all(self):
        """
        Train every suitable classification model.

        Returns:
            DataFrame of results sorted by F1 score
        """
        models = self._get_all_models()
        self.logger.log(f"Training {len(models)} classification models...\n")

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

                auc_str = f"{metrics['auc_roc']:.4f}" if metrics['auc_roc'] else "N/A"
                self.logger.log(
                    f"    Acc={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}, "
                    f"AUC={auc_str}, Time={metrics['train_time_sec']:.2f}s"
                )

            except Exception as e:
                self.logger.warning(f"    FAILED: {str(e)}")
                self.results.append({
                    "model": name,
                    "accuracy": None,
                    "precision": None,
                    "recall": None,
                    "f1_score": None,
                    "auc_roc": None,
                    "train_time_sec": None,
                    "error": str(e),
                })

        results_df = pd.DataFrame(self.results)

        successful = results_df[results_df["accuracy"].notna()].sort_values("f1_score", ascending=False)
        failed = results_df[results_df["accuracy"].isna()]
        results_df = pd.concat([successful, failed]).reset_index(drop=True)

        self.logger.success(
            f"\nCompleted: {len(successful)} succeeded, {len(failed)} failed"
        )

        if len(successful) > 0:
            best = successful.iloc[0]
            auc_str = f"{best['auc_roc']:.4f}" if best['auc_roc'] else "N/A"
            self.logger.success(
                f"Best model: {best['model']} (F1={best['f1_score']:.4f}, "
                f"Acc={best['accuracy']:.4f}, AUC={auc_str})"
            )

        return results_df

    # ----------------------------------------------------------
    # SAVE RESULTS
    # ----------------------------------------------------------
    def save_results(self, results_df, output_dir):
        """Save comparison table, plots, confusion matrices."""
        trad_dir = os.path.join(output_dir, "traditional")
        os.makedirs(trad_dir, exist_ok=True)

        # Save CSV
        csv_path = os.path.join(trad_dir, "classification_results.csv")
        results_df.to_csv(csv_path, index=False)
        self.logger.success(f"Results saved to: {csv_path}")

        successful = results_df[results_df["accuracy"].notna()].copy()
        if len(successful) > 0:
            self._plot_accuracy_comparison(successful, trad_dir)
            self._plot_f1_comparison(successful, trad_dir)
            self._plot_time_comparison(successful, trad_dir)
            self._plot_confusion_matrices(successful, trad_dir)
            self._save_classification_reports(successful, trad_dir)
            self._plot_metric_radar(successful, trad_dir)

        # Save best model summary
        if len(successful) > 0:
            best = successful.iloc[0]
            summary = {
                "best_model": best["model"],
                "accuracy": best["accuracy"],
                "precision": best["precision"],
                "recall": best["recall"],
                "f1_score": best["f1_score"],
                "auc_roc": best["auc_roc"],
                "n_classes": self.n_classes,
                "class_labels": [int(c) if isinstance(c, (np.integer,)) else c for c in self.class_labels],
                "total_models_tested": len(results_df),
                "successful_models": len(successful),
                "failed_models": len(results_df) - len(successful),
            }
            summary_path = os.path.join(trad_dir, "best_model_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, default=str)
            self.logger.success(f"Best model summary saved to: {summary_path}")

        return csv_path

    def _plot_accuracy_comparison(self, df, output_dir):
        """Bar chart of accuracy scores."""
        fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.35)))
        colors = ["green" if v > 0.9 else "steelblue" if v > 0.7 else "orange" if v > 0.5 else "red"
                   for v in df["accuracy"]]
        ax.barh(range(len(df)), df["accuracy"], color=colors, edgecolor="black")
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["model"], fontsize=9)
        ax.set_xlabel("Accuracy")
        ax.set_title("Classification Models - Accuracy Comparison")
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)

    def _plot_f1_comparison(self, df, output_dir):
        """Bar chart of F1 scores."""
        fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.35)))
        sorted_df = df.sort_values("f1_score", ascending=True)
        ax.barh(range(len(sorted_df)), sorted_df["f1_score"], color="coral", edgecolor="black")
        ax.set_yticks(range(len(sorted_df)))
        ax.set_yticklabels(sorted_df["model"], fontsize=9)
        ax.set_xlabel("F1 Score")
        ax.set_title("Classification Models - F1 Score Comparison")
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "f1_comparison.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)

    def _plot_time_comparison(self, df, output_dir):
        """Bar chart of training times."""
        fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.35)))
        sorted_df = df.sort_values("train_time_sec", ascending=True)
        ax.barh(range(len(sorted_df)), sorted_df["train_time_sec"], color="seagreen", edgecolor="black")
        ax.set_yticks(range(len(sorted_df)))
        ax.set_yticklabels(sorted_df["model"], fontsize=9)
        ax.set_xlabel("Training Time (seconds)")
        ax.set_title("Classification Models - Training Time")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_time.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)

    def _plot_confusion_matrices(self, df, output_dir):
        """Plot confusion matrices for top 3 models."""
        top_models = df.head(3)["model"].tolist()
        n_models = len(top_models)

        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
        if n_models == 1:
            axes = [axes]

        for i, name in enumerate(top_models):
            if name in self.trained_models:
                model = self.trained_models[name]
                y_pred = model.predict(self.X_test)
                cm = confusion_matrix(self.y_test, y_pred)

                sns.heatmap(
                    cm, annot=True, fmt="d", cmap="Blues",
                    ax=axes[i], cbar=False,
                    xticklabels=self.class_labels,
                    yticklabels=self.class_labels,
                )
                axes[i].set_title(f"{name}\nF1={df[df['model']==name]['f1_score'].values[0]:.4f}")
                axes[i].set_xlabel("Predicted")
                axes[i].set_ylabel("Actual")

        plt.suptitle("Confusion Matrices (Top 3 Models)", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrices.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)

    def _save_classification_reports(self, df, output_dir):
        """Save detailed classification report for top 3 models."""
        top_models = df.head(3)["model"].tolist()
        reports = {}

        for name in top_models:
            if name in self.trained_models:
                model = self.trained_models[name]
                y_pred = model.predict(self.X_test)
                report = classification_report(
                    self.y_test, y_pred, output_dict=True, zero_division=0
                )
                reports[name] = report

        report_path = os.path.join(output_dir, "classification_reports.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2, default=str)
        self.logger.success(f"Classification reports saved to: {report_path}")

    def _plot_metric_radar(self, df, output_dir):
        """Plot radar/spider chart comparing top 5 models across metrics."""
        top = df.head(5).copy()
        metrics = ["accuracy", "precision", "recall", "f1_score"]

        # Only include auc_roc if available for all top models
        if top["auc_roc"].notna().all():
            metrics.append("auc_roc")

        n_metrics = len(metrics)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # close the polygon

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for _, row in top.iterrows():
            values = [row[m] if row[m] is not None else 0 for m in metrics]
            values += values[:1]
            ax.plot(angles, values, linewidth=2, label=row["model"])
            ax.fill(angles, values, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title("Top 5 Models - Metric Comparison", fontsize=14, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "metric_radar.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)

    # ----------------------------------------------------------
    # PRINT SUMMARY
    # ----------------------------------------------------------
    def print_results_summary(self, results_df):
        """Print ranking table."""
        print("\n" + "=" * 100)
        print("  CLASSIFICATION MODEL RESULTS")
        print("=" * 100)

        successful = results_df[results_df["accuracy"].notna()]

        if len(successful) == 0:
            print("  No models completed successfully.")
            print("=" * 100)
            return

        print(f"\n  {'Rank':<6}{'Model':<28}{'Accuracy':<12}{'Precision':<12}"
              f"{'Recall':<12}{'F1':<12}{'AUC':<12}{'Time(s)':<10}")
        print("  " + "-" * 98)

        for i, (_, row) in enumerate(successful.iterrows(), 1):
            auc_str = f"{row['auc_roc']:.4f}" if row['auc_roc'] is not None else "N/A"
            print(
                f"  {i:<6}{row['model']:<28}{row['accuracy']:<12.4f}"
                f"{row['precision']:<12.4f}{row['recall']:<12.4f}"
                f"{row['f1_score']:<12.4f}{auc_str:<12}{row['train_time_sec']:<10.3f}"
            )

        failed = results_df[results_df["accuracy"].isna()]
        if len(failed) > 0:
            print(f"\n  Failed Models ({len(failed)}):")
            for _, row in failed.iterrows():
                err = row.get("error", "Unknown error")
                print(f"    - {row['model']}: {err}")

        best = successful.iloc[0]
        auc_str = f"{best['auc_roc']:.4f}" if best['auc_roc'] is not None else "N/A"
        print(f"\n  BEST MODEL: {best['model']}")
        print(f"  Accuracy = {best['accuracy']:.6f} | F1 = {best['f1_score']:.6f} | AUC = {auc_str}")
        print("=" * 100 + "\n")