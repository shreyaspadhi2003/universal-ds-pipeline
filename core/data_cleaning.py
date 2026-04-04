"""
Data Cleaning Module (Segment 2)
=================================
Handles:
  1. Dropping high-missing columns
  2. Missing value imputation (mean/median/mode/KNN/iterative)
  3. Outlier detection and treatment (IQR / Z-score)
  4. Categorical encoding (one-hot / label / target)
  5. Numeric scaling (standard / minmax / robust)
  6. Datetime feature extraction
  7. Saving cleaned data and cleaning report
"""

import os
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import KNNImputer, SimpleImputer


class DataCleaner:
    """Cleans and transforms the working dataset."""

    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        self.df = None
        self.target_col = None
        self.column_types = {}
        self.cleaning_report = {
            "dropped_columns": [],
            "missing_value_actions": {},
            "outlier_actions": {},
            "encoding_actions": {},
            "scaling_actions": {},
            "datetime_extractions": {},
        }
        self.label_encoders = {}   # Store for inverse transform later
        self.scaler = None

    # ----------------------------------------------------------
    # SETUP
    # ----------------------------------------------------------
    def set_data(self, df, column_types, target_col=None):
        """
        Set the DataFrame and column info to work with.

        Args:
            df: pandas DataFrame
            column_types: dict from DataIngestor.detect_column_types()
            target_col: name of target variable (excluded from some transforms)
        """
        self.df = df.copy()
        self.column_types = column_types.copy()
        self.target_col = target_col
        self.logger.section("DATA CLEANING")
        self.logger.log(f"Starting with {self.df.shape[0]} rows x {self.df.shape[1]} columns")
        return self.df

    # ----------------------------------------------------------
    # 1. DROP HIGH-MISSING COLUMNS
    # ----------------------------------------------------------
    def drop_high_missing(self, threshold=None):
        """
        Drop columns where missing percentage exceeds threshold.

        Args:
            threshold: float 0-1, default from config.MISSING_THRESHOLD
        """
        if threshold is None:
            threshold = self.config.MISSING_THRESHOLD

        self.logger.log(f"Checking for columns with >{threshold*100:.0f}% missing values...")

        dropped = []
        for col in self.df.columns:
            missing_pct = self.df[col].isnull().mean()
            if missing_pct > threshold:
                dropped.append(col)
                self.logger.warning(
                    f"  Dropping '{col}': {missing_pct*100:.1f}% missing"
                )

        if dropped:
            self.df.drop(columns=dropped, inplace=True)
            # Remove from column_types too
            for col in dropped:
                self.column_types.pop(col, None)
            self.cleaning_report["dropped_columns"] = dropped
        else:
            self.logger.log("  No columns exceed missing threshold.")

        self.logger.success(f"After drop: {self.df.shape[1]} columns remain")
        return self.df

    # ----------------------------------------------------------
    # 2. HANDLE MISSING VALUES
    # ----------------------------------------------------------
    def handle_missing_values(self, strategy="auto"):
        """
        Fill missing values based on column type.

        Args:
            strategy: 'auto', 'mean', 'median', 'mode', 'knn', 'drop_rows'
                - 'auto': mean for numeric, mode for categorical
                - 'knn': KNN imputer for numeric columns
                - 'drop_rows': drop any row with missing values
        """
        self.logger.log(f"Handling missing values (strategy='{strategy}')...")

        total_missing_before = self.df.isnull().sum().sum()
        if total_missing_before == 0:
            self.logger.log("  No missing values found. Skipping.")
            return self.df

        self.logger.log(f"  Total missing cells: {total_missing_before}")

        if strategy == "drop_rows":
            before = len(self.df)
            self.df.dropna(inplace=True)
            after = len(self.df)
            self.logger.log(f"  Dropped {before - after} rows with missing values")
            self.cleaning_report["missing_value_actions"]["strategy"] = "drop_rows"
            self.cleaning_report["missing_value_actions"]["rows_dropped"] = before - after
            return self.df

        if strategy == "knn":
            self._knn_impute()
            # Still fill categorical with mode after KNN (KNN only handles numeric)
            self._fill_categorical_mode()
            return self.df

        # Auto / mean / median / mode strategies
        for col in self.df.columns:
            if self.df[col].isnull().sum() == 0:
                continue

            ctype = self.column_types.get(col, "unknown")
            action = ""

            if ctype == "numeric":
                if strategy in ["auto", "mean"]:
                    fill_val = self.df[col].mean()
                    self.df[col].fillna(fill_val, inplace=True)
                    action = f"mean ({fill_val:.4f})"
                elif strategy == "median":
                    fill_val = self.df[col].median()
                    self.df[col].fillna(fill_val, inplace=True)
                    action = f"median ({fill_val:.4f})"
                elif strategy == "mode":
                    fill_val = self.df[col].mode()[0] if not self.df[col].mode().empty else 0
                    self.df[col].fillna(fill_val, inplace=True)
                    action = f"mode ({fill_val})"

            elif ctype in ["categorical", "boolean", "text"]:
                fill_val = self.df[col].mode()[0] if not self.df[col].mode().empty else "Unknown"
                self.df[col].fillna(fill_val, inplace=True)
                action = f"mode ({fill_val})"

            elif ctype == "datetime":
                # Forward fill for datetime
                self.df[col].fillna(method="ffill", inplace=True)
                self.df[col].fillna(method="bfill", inplace=True)
                action = "forward/backward fill"

            else:
                fill_val = self.df[col].mode()[0] if not self.df[col].mode().empty else "Unknown"
                self.df[col].fillna(fill_val, inplace=True)
                action = f"mode ({fill_val})"

            if action:
                self.cleaning_report["missing_value_actions"][col] = action
                self.logger.log(f"  {col}: filled with {action}")

        total_missing_after = self.df.isnull().sum().sum()
        self.logger.success(
            f"Missing values: {total_missing_before} -> {total_missing_after}"
        )
        return self.df

    def _knn_impute(self):
        """KNN imputation for numeric columns only."""
        numeric_cols = [
            col for col, ctype in self.column_types.items()
            if ctype == "numeric" and col in self.df.columns
        ]
        if not numeric_cols:
            return

        self.logger.log("  Running KNN imputation on numeric columns...")
        imputer = KNNImputer(n_neighbors=5)
        self.df[numeric_cols] = imputer.fit_transform(self.df[numeric_cols])

        for col in numeric_cols:
            self.cleaning_report["missing_value_actions"][col] = "KNN (k=5)"

        self.logger.log(f"  KNN imputed {len(numeric_cols)} numeric columns")

    def _fill_categorical_mode(self):
        """Fill remaining categorical missing values with mode."""
        for col in self.df.columns:
            if self.df[col].isnull().sum() == 0:
                continue
            ctype = self.column_types.get(col, "unknown")
            if ctype in ["categorical", "boolean", "text"]:
                fill_val = self.df[col].mode()[0] if not self.df[col].mode().empty else "Unknown"
                self.df[col].fillna(fill_val, inplace=True)
                self.cleaning_report["missing_value_actions"][col] = f"mode ({fill_val})"

    # ----------------------------------------------------------
    # 3. OUTLIER DETECTION AND TREATMENT
    # ----------------------------------------------------------
    def handle_outliers(self, method=None, action="clip"):
        """
        Detect and handle outliers in numeric columns.

        Args:
            method: 'iqr' or 'zscore', default from config
            action: 'clip' (cap to bounds), 'remove' (drop rows), 'flag' (add column)
        """
        if method is None:
            method = self.config.OUTLIER_METHOD

        self.logger.log(f"Handling outliers (method='{method}', action='{action}')...")

        numeric_cols = [
            col for col, ctype in self.column_types.items()
            if ctype == "numeric" and col in self.df.columns and col != self.target_col
        ]

        total_outliers = 0

        for col in numeric_cols:
            if method == "iqr":
                outlier_count, lower, upper = self._iqr_outliers(col)
            elif method == "zscore":
                outlier_count, lower, upper = self._zscore_outliers(col)
            else:
                self.logger.warning(f"  Unknown method '{method}', skipping outliers.")
                return self.df

            if outlier_count == 0:
                continue

            total_outliers += outlier_count

            if action == "clip":
                self.df[col] = self.df[col].clip(lower=lower, upper=upper)
                act_desc = f"clipped to [{lower:.2f}, {upper:.2f}]"
            elif action == "remove":
                mask = (self.df[col] >= lower) & (self.df[col] <= upper)
                self.df = self.df[mask]
                act_desc = f"removed {outlier_count} rows"
            elif action == "flag":
                flag_col = f"{col}_outlier"
                self.df[flag_col] = ((self.df[col] < lower) | (self.df[col] > upper)).astype(int)
                act_desc = f"flagged in '{flag_col}'"
            else:
                act_desc = "no action"

            self.cleaning_report["outlier_actions"][col] = {
                "method": method,
                "outliers_found": outlier_count,
                "action": act_desc,
                "bounds": [round(lower, 4), round(upper, 4)],
            }
            self.logger.log(f"  {col}: {outlier_count} outliers -> {act_desc}")

        self.logger.success(f"Total outliers handled: {total_outliers}")
        return self.df

    def _iqr_outliers(self, col):
        """Detect outliers using IQR method."""
        Q1 = self.df[col].quantile(0.25)
        Q3 = self.df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_count = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
        return outlier_count, lower, upper

    def _zscore_outliers(self, col):
        """Detect outliers using Z-score method (threshold=3)."""
        mean = self.df[col].mean()
        std = self.df[col].std()
        if std == 0:
            return 0, mean, mean
        lower = mean - 3 * std
        upper = mean + 3 * std
        outlier_count = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
        return outlier_count, lower, upper

    # ----------------------------------------------------------
    # 4. ENCODE CATEGORICAL COLUMNS
    # ----------------------------------------------------------
    def encode_categoricals(self, max_onehot=None):
        """
        Encode categorical columns.
        - One-hot encoding if unique values <= max_onehot
        - Label encoding if unique values > max_onehot

        Args:
            max_onehot: threshold for one-hot vs label encoding
        """
        if max_onehot is None:
            max_onehot = self.config.MAX_CATEGORIES_ONEHOT

        self.logger.log(f"Encoding categoricals (one-hot threshold={max_onehot})...")

        cat_cols = [
            col for col, ctype in self.column_types.items()
            if ctype in ["categorical", "boolean"] and col in self.df.columns and col != self.target_col
        ]

        if not cat_cols:
            self.logger.log("  No categorical columns to encode.")
            return self.df

        for col in cat_cols:
            nunique = self.df[col].nunique()

            if self.df[col].dtype == bool or self.column_types.get(col) == "boolean":
                # Boolean -> 0/1
                self.df[col] = self.df[col].astype(int)
                self.cleaning_report["encoding_actions"][col] = "boolean -> int"
                self.logger.log(f"  {col}: boolean -> int")

            elif nunique <= max_onehot:
                # One-hot encoding
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                dummies = dummies.astype(int)
                self.df = pd.concat([self.df.drop(columns=[col]), dummies], axis=1)
                self.cleaning_report["encoding_actions"][col] = f"one-hot ({nunique} categories)"
                self.logger.log(f"  {col}: one-hot encoded ({nunique} categories -> {len(dummies.columns)} columns)")

            else:
                # Label encoding
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
                self.cleaning_report["encoding_actions"][col] = f"label encoded ({nunique} categories)"
                self.logger.log(f"  {col}: label encoded ({nunique} categories)")

        # Also encode the target if it's categorical (for classification)
        if self.target_col and self.target_col in self.df.columns:
            target_type = self.column_types.get(self.target_col, "")
            if target_type in ["categorical", "boolean"] and self.df[self.target_col].dtype == "object":
                le = LabelEncoder()
                self.df[self.target_col] = le.fit_transform(self.df[self.target_col].astype(str))
                self.label_encoders[self.target_col] = le
                self.cleaning_report["encoding_actions"][self.target_col] = "target label encoded"
                self.logger.log(f"  {self.target_col} (target): label encoded")

        self.logger.success(f"Encoding complete. Shape: {self.df.shape}")
        return self.df

    # ----------------------------------------------------------
    # 5. SCALE NUMERIC COLUMNS
    # ----------------------------------------------------------
    def scale_numerics(self, method=None):
        """
        Scale numeric columns.

        Args:
            method: 'standard', 'minmax', or 'robust'
        """
        if method is None:
            method = self.config.SCALING_METHOD

        self.logger.log(f"Scaling numeric columns (method='{method}')...")

        numeric_cols = [
            col for col, ctype in self.column_types.items()
            if ctype == "numeric" and col in self.df.columns and col != self.target_col
        ]

        if not numeric_cols:
            self.logger.log("  No numeric columns to scale.")
            return self.df

        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        elif method == "robust":
            self.scaler = RobustScaler()
        else:
            self.logger.warning(f"  Unknown scaling method '{method}'")
            return self.df

        self.df[numeric_cols] = self.scaler.fit_transform(self.df[numeric_cols])

        for col in numeric_cols:
            self.cleaning_report["scaling_actions"][col] = method
            self.logger.log(f"  {col}: {method} scaled")

        self.logger.success(f"Scaled {len(numeric_cols)} numeric columns")
        return self.df

    # ----------------------------------------------------------
    # 6. EXTRACT DATETIME FEATURES
    # ----------------------------------------------------------
    def extract_datetime_features(self):
        """
        Convert datetime columns into useful numeric features:
        year, month, day, day_of_week, is_weekend
        Then drop the original datetime column.
        """
        self.logger.log("Extracting datetime features...")

        dt_cols = [
            col for col, ctype in self.column_types.items()
            if ctype == "datetime" and col in self.df.columns
        ]

        if not dt_cols:
            self.logger.log("  No datetime columns found.")
            return self.df

        for col in dt_cols:
            try:
                # Convert to datetime if not already
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    dt_series = pd.to_datetime(self.df[col], errors="coerce")

                self.df[f"{col}_year"] = dt_series.dt.year
                self.df[f"{col}_month"] = dt_series.dt.month
                self.df[f"{col}_day"] = dt_series.dt.day
                self.df[f"{col}_dayofweek"] = dt_series.dt.dayofweek
                self.df[f"{col}_is_weekend"] = (dt_series.dt.dayofweek >= 5).astype(int)

                # If time component exists
                if dt_series.dt.hour.sum() > 0:
                    self.df[f"{col}_hour"] = dt_series.dt.hour

                # Drop original
                self.df.drop(columns=[col], inplace=True)
                self.column_types.pop(col, None)

                features_added = [c for c in self.df.columns if c.startswith(f"{col}_")]
                self.cleaning_report["datetime_extractions"][col] = features_added
                self.logger.log(f"  {col}: extracted {len(features_added)} features -> dropped original")

            except Exception as e:
                self.logger.warning(f"  Failed to extract from '{col}': {str(e)}")

        self.logger.success("Datetime extraction complete")
        return self.df

    # ----------------------------------------------------------
    # 7. DROP TEXT COLUMNS
    # ----------------------------------------------------------
    def drop_text_columns(self):
        """
        Drop text/high-cardinality columns that can't be used for modeling.
        """
        text_cols = [
            col for col, ctype in self.column_types.items()
            if ctype == "text" and col in self.df.columns and col != self.target_col
        ]

        if text_cols:
            self.logger.log(f"Dropping text columns not usable for modeling: {text_cols}")
            self.df.drop(columns=text_cols, inplace=True)
            for col in text_cols:
                self.column_types.pop(col, None)
                self.cleaning_report["dropped_columns"].append(col)
        else:
            self.logger.log("No text columns to drop.")

        return self.df

    # ----------------------------------------------------------
    # 8. RUN FULL CLEANING PIPELINE
    # ----------------------------------------------------------
    def run_full_cleaning(self, missing_strategy="auto", outlier_action="clip"):
        """
        Run all cleaning steps in the correct order.

        Args:
            missing_strategy: 'auto', 'mean', 'median', 'mode', 'knn', 'drop_rows'
            outlier_action: 'clip', 'remove', 'flag'

        Returns:
            cleaned DataFrame
        """
        self.logger.section("FULL CLEANING PIPELINE")

        # Step 1: Drop high-missing columns
        self.drop_high_missing()

        # Step 2: Extract datetime features (before encoding)
        self.extract_datetime_features()

        # Step 3: Drop text columns
        self.drop_text_columns()

        # Step 4: Handle missing values
        self.handle_missing_values(strategy=missing_strategy)

        # Step 5: Handle outliers (before scaling)
        self.handle_outliers(action=outlier_action)

        # Step 6: Encode categoricals
        self.encode_categoricals()

        # Step 7: Scale numerics
        self.scale_numerics()

        self.logger.success(
            f"Cleaning complete: {self.df.shape[0]} rows x {self.df.shape[1]} columns"
        )
        return self.df

    # ----------------------------------------------------------
    # 9. SAVE CLEANED DATA AND REPORT
    # ----------------------------------------------------------
    def save_cleaned_data(self, output_dir):
        """Save cleaned DataFrame and cleaning report."""
        os.makedirs(output_dir, exist_ok=True)

        # Save cleaned CSV
        csv_path = os.path.join(output_dir, "cleaned_data.csv")
        self.df.to_csv(csv_path, index=False)
        self.logger.success(f"Cleaned data saved to: {csv_path}")

        # Save cleaning report
        report_path = os.path.join(output_dir, "cleaning_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(self.cleaning_report, f, indent=2, default=str)
        self.logger.success(f"Cleaning report saved to: {report_path}")

        return csv_path, report_path

    # ----------------------------------------------------------
    # 10. PRINT CLEANING SUMMARY
    # ----------------------------------------------------------
    def print_cleaning_summary(self):
        """Print what was done during cleaning."""
        print("\n" + "=" * 60)
        print("  CLEANING SUMMARY")
        print("=" * 60)

        print(f"\n  Final Shape: {self.df.shape[0]} rows x {self.df.shape[1]} columns")

        if self.cleaning_report["dropped_columns"]:
            print(f"\n  Dropped Columns ({len(self.cleaning_report['dropped_columns'])}):")
            for col in self.cleaning_report["dropped_columns"]:
                print(f"    - {col}")

        if self.cleaning_report["missing_value_actions"]:
            print(f"\n  Missing Value Actions ({len(self.cleaning_report['missing_value_actions'])}):")
            for col, action in self.cleaning_report["missing_value_actions"].items():
                print(f"    {col:30s} : {action}")

        if self.cleaning_report["outlier_actions"]:
            print(f"\n  Outlier Actions ({len(self.cleaning_report['outlier_actions'])}):")
            for col, info in self.cleaning_report["outlier_actions"].items():
                print(f"    {col:30s} : {info['outliers_found']} outliers -> {info['action']}")

        if self.cleaning_report["encoding_actions"]:
            print(f"\n  Encoding Actions ({len(self.cleaning_report['encoding_actions'])}):")
            for col, action in self.cleaning_report["encoding_actions"].items():
                print(f"    {col:30s} : {action}")

        if self.cleaning_report["scaling_actions"]:
            print(f"\n  Scaling Actions ({len(self.cleaning_report['scaling_actions'])}):")
            for col, method in self.cleaning_report["scaling_actions"].items():
                print(f"    {col:30s} : {method}")

        if self.cleaning_report["datetime_extractions"]:
            print(f"\n  Datetime Extractions:")
            for col, features in self.cleaning_report["datetime_extractions"].items():
                print(f"    {col} -> {features}")

        remaining_missing = self.df.isnull().sum().sum()
        print(f"\n  Remaining Missing Values: {remaining_missing}")
        print("=" * 60 + "\n")