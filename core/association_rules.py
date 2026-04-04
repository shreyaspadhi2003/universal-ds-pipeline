"""
Association Rules / Market Basket Analysis (Segment 8)
=======================================================
Handles:
  1. Detecting if data is transactional/binary
  2. Converting data to transaction format
  3. Running Apriori algorithm
  4. Running FP-Growth algorithm
  5. Generating association rules
  6. Visualizing rules (support vs confidence, network graph)
  7. Saving results
"""

import os
import pandas as pd
import numpy as np
import json
import time
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# Check for mlxtend
try:
    from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False


class AssociationAnalyzer:
    """Performs market basket analysis using association rule mining."""

    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        self.transaction_df = None  # Binary encoded transaction matrix
        self.frequent_itemsets = {}  # {method: itemsets_df}
        self.rules = {}  # {method: rules_df}
        self.report = {
            "is_suitable": False,
            "conversion_method": None,
            "methods_run": [],
            "total_rules": 0,
        }

    # ----------------------------------------------------------
    # 1. CHECK DATA SUITABILITY
    # ----------------------------------------------------------
    def check_suitability(self, df):
        """
        Check if the dataset is suitable for market basket analysis.

        Suitable data looks like:
        - Binary columns (0/1) representing item presence
        - Categorical columns that can be one-hot encoded into items
        - Transaction-like structure (rows = transactions, cols = items)

        Returns:
            bool, str (is_suitable, reason)
        """
        self.logger.section("ASSOCIATION RULES / MARKET BASKET ANALYSIS")
        self.logger.log("Checking data suitability...")

        n_rows, n_cols = df.shape

        # Check 1: Are most columns binary (0/1)?
        binary_cols = []
        for col in df.columns:
            unique_vals = set(df[col].dropna().unique())
            if unique_vals <= {0, 1} or unique_vals <= {0.0, 1.0} or unique_vals <= {True, False}:
                binary_cols.append(col)

        binary_ratio = len(binary_cols) / max(n_cols, 1)

        if binary_ratio >= 0.5:
            self.logger.success(
                f"Data appears transactional: {len(binary_cols)}/{n_cols} binary columns"
            )
            self.report["is_suitable"] = True
            self.report["conversion_method"] = "already_binary"
            return True, "already_binary"

        # Check 2: Are there categorical columns that look like items?
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        low_card_cols = [c for c in df.columns if df[c].nunique() <= 20 and df[c].nunique() >= 2]

        if len(cat_cols) >= 2 or len(low_card_cols) >= 3:
            self.logger.success(
                f"Data can be converted: {len(cat_cols)} categorical, "
                f"{len(low_card_cols)} low-cardinality columns"
            )
            self.report["is_suitable"] = True
            self.report["conversion_method"] = "encode_categoricals"
            return True, "encode_categoricals"

        # Check 3: Numeric data that can be binned
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 3:
            self.logger.log("Numeric data can be binned into categories for association rules")
            self.report["is_suitable"] = True
            self.report["conversion_method"] = "bin_numerics"
            return True, "bin_numerics"

        self.logger.warning("Data does not appear suitable for market basket analysis.")
        self.report["is_suitable"] = False
        return False, "not_suitable"

    # ----------------------------------------------------------
    # 2. CONVERT TO TRANSACTION FORMAT
    # ----------------------------------------------------------
    def prepare_transactions(self, df, method="auto"):
        """
        Convert DataFrame to binary transaction matrix.

        Args:
            df: input DataFrame
            method: 'already_binary', 'encode_categoricals', 'bin_numerics', 'auto'

        Returns:
            binary DataFrame (rows=transactions, cols=items, values=0/1)
        """
        self.logger.log(f"Preparing transaction matrix (method='{method}')...")

        if method == "auto":
            _, method = self.check_suitability(df)

        if method == "already_binary":
            # Use binary columns directly
            binary_cols = []
            for col in df.columns:
                unique_vals = set(df[col].dropna().unique())
                if unique_vals <= {0, 1, 0.0, 1.0, True, False}:
                    binary_cols.append(col)
            self.transaction_df = df[binary_cols].astype(bool)

        elif method == "encode_categoricals":
            # One-hot encode categorical and low-cardinality columns
            cols_to_encode = []
            for col in df.columns:
                if df[col].dtype == "object" or df[col].nunique() <= 15:
                    cols_to_encode.append(col)

            if not cols_to_encode:
                # Fallback: bin numerics
                return self.prepare_transactions(df, method="bin_numerics")

            encoded = pd.get_dummies(df[cols_to_encode], prefix_sep="=")
            self.transaction_df = encoded.astype(bool)

        elif method == "bin_numerics":
            # Bin numeric columns into quartile categories, then one-hot encode
            binned_frames = []
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            for col in numeric_cols:
                try:
                    binned = pd.qcut(df[col], q=4, labels=["low", "med_low", "med_high", "high"],
                                     duplicates="drop")
                    dummies = pd.get_dummies(binned, prefix=col)
                    binned_frames.append(dummies)
                except Exception:
                    pass

            if binned_frames:
                self.transaction_df = pd.concat(binned_frames, axis=1).astype(bool)
            else:
                self.logger.error("Failed to create transaction matrix.")
                return None

        else:
            self.logger.error(f"Unknown method: {method}")
            return None

        # Remove columns with very low support (< 1%)
        min_support_count = max(1, int(len(self.transaction_df) * 0.01))
        col_sums = self.transaction_df.sum()
        valid_cols = col_sums[col_sums >= min_support_count].index
        self.transaction_df = self.transaction_df[valid_cols]

        self.logger.success(
            f"Transaction matrix: {self.transaction_df.shape[0]} transactions x "
            f"{self.transaction_df.shape[1]} items"
        )
        return self.transaction_df

    # ----------------------------------------------------------
    # 3. RUN APRIORI
    # ----------------------------------------------------------
    def run_apriori(self, min_support=0.05, min_confidence=0.3, min_lift=1.0):
        """
        Run Apriori algorithm to find frequent itemsets and rules.

        Args:
            min_support: minimum support threshold
            min_confidence: minimum confidence for rules
            min_lift: minimum lift for rules

        Returns:
            tuple (frequent_itemsets_df, rules_df)
        """
        if not MLXTEND_AVAILABLE:
            self.logger.error("mlxtend not installed. Run: pip install mlxtend")
            return None, None

        if self.transaction_df is None:
            self.logger.error("No transaction matrix. Call prepare_transactions() first.")
            return None, None

        self.logger.log(f"Running Apriori (min_support={min_support})...")

        try:
            start = time.time()
            freq_items = apriori(
                self.transaction_df,
                min_support=min_support,
                use_colnames=True,
                max_len=4,
            )
            apriori_time = time.time() - start

            if len(freq_items) == 0:
                self.logger.warning("No frequent itemsets found. Try lowering min_support.")
                return pd.DataFrame(), pd.DataFrame()

            self.logger.log(f"  Frequent itemsets found: {len(freq_items)} (in {apriori_time:.2f}s)")

            # Generate rules
            rules = association_rules(
                freq_items, metric="confidence", min_threshold=min_confidence
            )

            # Filter by lift
            rules = rules[rules["lift"] >= min_lift]

            self.frequent_itemsets["apriori"] = freq_items
            self.rules["apriori"] = rules.sort_values("lift", ascending=False).reset_index(drop=True)

            self.logger.success(
                f"  Apriori rules: {len(rules)} (confidence >= {min_confidence}, lift >= {min_lift})"
            )
            return freq_items, rules

        except Exception as e:
            self.logger.error(f"Apriori failed: {str(e)}")
            return None, None

    # ----------------------------------------------------------
    # 4. RUN FP-GROWTH
    # ----------------------------------------------------------
    def run_fpgrowth(self, min_support=0.05, min_confidence=0.3, min_lift=1.0):
        """
        Run FP-Growth algorithm (faster than Apriori).

        Returns:
            tuple (frequent_itemsets_df, rules_df)
        """
        if not MLXTEND_AVAILABLE:
            self.logger.error("mlxtend not installed. Run: pip install mlxtend")
            return None, None

        if self.transaction_df is None:
            self.logger.error("No transaction matrix.")
            return None, None

        self.logger.log(f"Running FP-Growth (min_support={min_support})...")

        try:
            start = time.time()
            freq_items = fpgrowth(
                self.transaction_df,
                min_support=min_support,
                use_colnames=True,
                max_len=4,
            )
            fp_time = time.time() - start

            if len(freq_items) == 0:
                self.logger.warning("No frequent itemsets found with FP-Growth.")
                return pd.DataFrame(), pd.DataFrame()

            self.logger.log(f"  Frequent itemsets found: {len(freq_items)} (in {fp_time:.2f}s)")

            rules = association_rules(
                freq_items, metric="confidence", min_threshold=min_confidence
            )
            rules = rules[rules["lift"] >= min_lift]

            self.frequent_itemsets["fpgrowth"] = freq_items
            self.rules["fpgrowth"] = rules.sort_values("lift", ascending=False).reset_index(drop=True)

            self.logger.success(
                f"  FP-Growth rules: {len(rules)} (confidence >= {min_confidence}, lift >= {min_lift})"
            )
            return freq_items, rules

        except Exception as e:
            self.logger.error(f"FP-Growth failed: {str(e)}")
            return None, None

    # ----------------------------------------------------------
    # 5. RUN ALL
    # ----------------------------------------------------------
    def run_all(self, df, min_support=0.05, min_confidence=0.3, min_lift=1.0):
        """
        Full pipeline: check suitability, prepare, run both algorithms.

        Returns:
            dict with all results
        """
        is_suitable, method = self.check_suitability(df)

        if not is_suitable:
            self.logger.warning("Skipping association rules - data not suitable.")
            return {"suitable": False}

        if not MLXTEND_AVAILABLE:
            self.logger.error(
                "mlxtend library required for association rules. "
                "Install with: pip install mlxtend"
            )
            return {"suitable": True, "error": "mlxtend not installed"}

        self.prepare_transactions(df, method=method)

        if self.transaction_df is None or self.transaction_df.empty:
            self.logger.warning("Failed to create transaction matrix.")
            return {"suitable": True, "error": "transaction_creation_failed"}

        # Run both algorithms
        apriori_items, apriori_rules = self.run_apriori(min_support, min_confidence, min_lift)
        fp_items, fp_rules = self.run_fpgrowth(min_support, min_confidence, min_lift)

        total_rules = 0
        methods_run = []

        if apriori_rules is not None and len(apriori_rules) > 0:
            total_rules += len(apriori_rules)
            methods_run.append("apriori")

        if fp_rules is not None and len(fp_rules) > 0:
            total_rules += len(fp_rules)
            methods_run.append("fpgrowth")

        self.report["methods_run"] = methods_run
        self.report["total_rules"] = total_rules

        self.logger.success(f"Total rules found: {total_rules} from {len(methods_run)} method(s)")

        return {
            "suitable": True,
            "apriori_itemsets": apriori_items,
            "apriori_rules": apriori_rules,
            "fpgrowth_itemsets": fp_items,
            "fpgrowth_rules": fp_rules,
        }

    # ----------------------------------------------------------
    # 6. SAVE RESULTS
    # ----------------------------------------------------------
    def save_results(self, output_dir):
        """Save rules, itemsets, and visualizations."""
        os.makedirs(output_dir, exist_ok=True)

        # Save rules for each method
        for method, rules in self.rules.items():
            if rules is not None and len(rules) > 0:
                # Convert frozensets to strings for CSV
                rules_save = rules.copy()
                rules_save["antecedents"] = rules_save["antecedents"].apply(
                    lambda x: ", ".join(list(x)) if isinstance(x, frozenset) else str(x)
                )
                rules_save["consequents"] = rules_save["consequents"].apply(
                    lambda x: ", ".join(list(x)) if isinstance(x, frozenset) else str(x)
                )

                csv_path = os.path.join(output_dir, f"{method}_rules.csv")
                rules_save.to_csv(csv_path, index=False)
                self.logger.success(f"{method} rules saved to: {csv_path}")

        # Save frequent itemsets
        for method, items in self.frequent_itemsets.items():
            if items is not None and len(items) > 0:
                items_save = items.copy()
                items_save["itemsets"] = items_save["itemsets"].apply(
                    lambda x: ", ".join(list(x)) if isinstance(x, frozenset) else str(x)
                )
                csv_path = os.path.join(output_dir, f"{method}_frequent_itemsets.csv")
                items_save.to_csv(csv_path, index=False)

        # Generate plots
        self._plot_rules(output_dir)

        # Save report
        report_path = os.path.join(output_dir, "association_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(self.report, f, indent=2, default=str)
        self.logger.success(f"Report saved to: {report_path}")

    def _plot_rules(self, output_dir):
        """Visualize association rules."""
        # Use the method with more rules
        best_method = None
        best_count = 0
        for method, rules in self.rules.items():
            if rules is not None and len(rules) > best_count:
                best_count = len(rules)
                best_method = method

        if best_method is None or best_count == 0:
            self.logger.log("No rules to plot.")
            return

        rules = self.rules[best_method]

        # Plot 1: Support vs Confidence scatter (colored by lift)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        scatter = axes[0].scatter(
            rules["support"], rules["confidence"],
            c=rules["lift"], cmap="YlOrRd", alpha=0.7,
            s=50, edgecolors="black", linewidth=0.5,
        )
        axes[0].set_xlabel("Support")
        axes[0].set_ylabel("Confidence")
        axes[0].set_title(f"Support vs Confidence (color=Lift)\n{best_method} - {len(rules)} rules")
        plt.colorbar(scatter, ax=axes[0], label="Lift")

        # Plot 2: Top 15 rules by lift
        top_rules = rules.head(15).copy()
        top_rules["rule"] = top_rules.apply(
            lambda r: f"{', '.join(list(r['antecedents']))} -> {', '.join(list(r['consequents']))}",
            axis=1
        )
        # Truncate long rule names
        top_rules["rule"] = top_rules["rule"].apply(lambda x: x[:50] + "..." if len(x) > 50 else x)

        axes[1].barh(range(len(top_rules)), top_rules["lift"], color="coral", edgecolor="black")
        axes[1].set_yticks(range(len(top_rules)))
        axes[1].set_yticklabels(top_rules["rule"], fontsize=8)
        axes[1].set_xlabel("Lift")
        axes[1].set_title(f"Top {len(top_rules)} Rules by Lift")
        axes[1].invert_yaxis()

        plt.tight_layout()
        path = os.path.join(output_dir, "association_rules_plots.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        self.logger.success(f"Rules plot saved to: {path}")

        # Plot 3: Heatmap of support x confidence x lift
        if len(rules) >= 5:
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(
                rules["support"], rules["confidence"],
                s=rules["lift"] * 50, alpha=0.5,
                c=rules["lift"], cmap="viridis",
                edgecolors="black", linewidth=0.5,
            )
            ax.set_xlabel("Support")
            ax.set_ylabel("Confidence")
            ax.set_title("Association Rules\n(size & color = Lift)")
            plt.colorbar(scatter, ax=ax, label="Lift")
            plt.tight_layout()
            path = os.path.join(output_dir, "rules_bubble_chart.png")
            plt.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)

    # ----------------------------------------------------------
    # 7. PRINT SUMMARY
    # ----------------------------------------------------------
    def print_summary(self):
        """Print association rules summary."""
        print("\n" + "=" * 80)
        print("  ASSOCIATION RULES / MARKET BASKET ANALYSIS SUMMARY")
        print("=" * 80)

        print(f"\n  Data Suitable    : {self.report['is_suitable']}")
        print(f"  Conversion Method: {self.report['conversion_method']}")
        print(f"  Methods Run      : {self.report['methods_run']}")
        print(f"  Total Rules      : {self.report['total_rules']}")

        if self.transaction_df is not None:
            print(f"  Transaction Shape: {self.transaction_df.shape}")

        for method, rules in self.rules.items():
            if rules is not None and len(rules) > 0:
                print(f"\n  --- {method.upper()} ---")
                print(f"  Rules found: {len(rules)}")
                print(f"  Avg Support   : {rules['support'].mean():.4f}")
                print(f"  Avg Confidence: {rules['confidence'].mean():.4f}")
                print(f"  Avg Lift      : {rules['lift'].mean():.4f}")
                print(f"  Max Lift      : {rules['lift'].max():.4f}")

                print(f"\n  Top 5 Rules by Lift:")
                for i, (_, row) in enumerate(rules.head(5).iterrows(), 1):
                    ant = ", ".join(list(row["antecedents"]))
                    con = ", ".join(list(row["consequents"]))
                    print(
                        f"    {i}. {ant} -> {con}"
                        f"  (sup={row['support']:.3f}, conf={row['confidence']:.3f}, "
                        f"lift={row['lift']:.3f})"
                    )

        print("=" * 80 + "\n")