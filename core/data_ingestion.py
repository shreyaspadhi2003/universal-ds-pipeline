"""
Data Ingestion Module (Segment 1)
=================================
Handles:
  1. Loading single or multiple CSV/Excel files
  2. Detecting if tables are relational (shared key columns)
  3. Offering merge options for relational tables
  4. Auto-detecting column types (numeric, categorical, datetime, boolean, text)
  5. Generating metadata report
  6. Target variable selection
"""

import os
import pandas as pd
import numpy as np
import json


class DataIngestor:
    """Loads, inspects, and optionally merges datasets."""

    def __init__(self, logger):
        self.logger = logger
        self.dataframes = {}       # {filename: DataFrame}
        self.merged_df = None      # Final merged/selected DataFrame
        self.column_types = {}     # {col_name: detected_type}
        self.metadata = {}         # Full metadata dictionary
        self.target_variable = None
        self.file_paths = []

    # ----------------------------------------------------------
    # 1. LOAD FILES
    # ----------------------------------------------------------
    def load_files(self, file_paths):
        """
        Load one or more CSV/Excel files into DataFrames.

        Args:
            file_paths: list of file path strings

        Returns:
            dict of {filename: DataFrame}
        """
        self.file_paths = file_paths
        self.logger.section("DATA INGESTION")

        for path in file_paths:
            if not os.path.exists(path):
                self.logger.error(f"File not found: {path}")
                continue

            filename = os.path.basename(path)
            ext = os.path.splitext(path)[1].lower()

            try:
                if ext == ".csv":
                    df = pd.read_csv(path)
                elif ext in [".xlsx", ".xls"]:
                    df = pd.read_excel(path)
                else:
                    self.logger.warning(f"Unsupported file type: {ext} for {filename}")
                    continue

                self.dataframes[filename] = df
                self.logger.success(
                    f"Loaded '{filename}': {df.shape[0]} rows x {df.shape[1]} columns"
                )
            except Exception as e:
                self.logger.error(f"Failed to load '{filename}': {str(e)}")

        return self.dataframes

    # ----------------------------------------------------------
    # 2. DETECT RELATIONAL KEYS
    # ----------------------------------------------------------
    def detect_relational_keys(self):
        """
        Check if multiple loaded tables share common column names
        that could serve as join keys.

        Returns:
            dict: {(file1, file2): [shared_columns]}
        """
        if len(self.dataframes) < 2:
            self.logger.log("Single dataset loaded — no relational detection needed.")
            return {}

        self.logger.section("RELATIONAL KEY DETECTION")
        filenames = list(self.dataframes.keys())
        shared_keys = {}

        for i in range(len(filenames)):
            for j in range(i + 1, len(filenames)):
                f1, f2 = filenames[i], filenames[j]
                cols1 = set(self.dataframes[f1].columns)
                cols2 = set(self.dataframes[f2].columns)
                common = cols1.intersection(cols2)

                if common:
                    shared_keys[(f1, f2)] = list(common)
                    self.logger.success(
                        f"Shared columns between '{f1}' and '{f2}': {list(common)}"
                    )
                else:
                    self.logger.log(
                        f"No shared columns between '{f1}' and '{f2}'."
                    )

        return shared_keys

    # ----------------------------------------------------------
    # 3. MERGE TABLES
    # ----------------------------------------------------------
    def merge_tables(self, file1, file2, on_columns, how="inner"):
        """
        Merge two DataFrames on specified columns.

        Args:
            file1, file2: filenames (keys in self.dataframes)
            on_columns: list of column names to join on
            how: merge type — 'inner', 'left', 'right', 'outer'

        Returns:
            merged DataFrame
        """
        self.logger.section("MERGING TABLES")
        try:
            df1 = self.dataframes[file1]
            df2 = self.dataframes[file2]
            merged = pd.merge(df1, df2, on=on_columns, how=how)
            self.logger.success(
                f"Merged '{file1}' + '{file2}' on {on_columns} ({how}): "
                f"{merged.shape[0]} rows x {merged.shape[1]} columns"
            )
            self.merged_df = merged
            return merged
        except Exception as e:
            self.logger.error(f"Merge failed: {str(e)}")
            return None

    # ----------------------------------------------------------
    # 4. SELECT WORKING DATASET
    # ----------------------------------------------------------
    def set_working_dataframe(self, df=None, filename=None):
        """
        Set the main DataFrame for the pipeline.
        Either pass a df directly or a filename from loaded files.
        """
        if df is not None:
            self.merged_df = df
        elif filename and filename in self.dataframes:
            self.merged_df = self.dataframes[filename].copy()
        elif len(self.dataframes) == 1:
            # If only one file loaded, use it automatically
            key = list(self.dataframes.keys())[0]
            self.merged_df = self.dataframes[key].copy()
            self.logger.log(f"Auto-selected single dataset: '{key}'")
        else:
            self.logger.error("No working DataFrame set. Provide df or filename.")
            return None

        self.logger.success(
            f"Working dataset set: {self.merged_df.shape[0]} rows x "
            f"{self.merged_df.shape[1]} columns"
        )
        return self.merged_df

    # ----------------------------------------------------------
    # 5. AUTO-DETECT COLUMN TYPES
    # ----------------------------------------------------------
    def detect_column_types(self):
        """
        Classify each column as: numeric, categorical, datetime, boolean, or text.

        Logic:
          - If dtype is bool → boolean
          - If dtype is datetime → datetime
          - If dtype is numeric and unique values > 20 → numeric
          - If dtype is numeric and unique values <= 20 → categorical (likely encoded)
          - If dtype is object and unique values <= MAX threshold → categorical
          - If dtype is object and unique values > MAX threshold → text
          - Try parsing object columns as datetime
        """
        if self.merged_df is None:
            self.logger.error("No working DataFrame. Call set_working_dataframe() first.")
            return {}

        self.logger.section("COLUMN TYPE DETECTION")
        df = self.merged_df
        type_map = {}

        for col in df.columns:
            dtype = df[col].dtype
            nunique = df[col].nunique()
            total = len(df[col])

            if dtype == bool or (dtype == "object" and set(df[col].dropna().unique()) <= {"True", "False", "true", "false", "0", "1"}):
                type_map[col] = "boolean"

            elif pd.api.types.is_datetime64_any_dtype(dtype):
                type_map[col] = "datetime"

            elif pd.api.types.is_numeric_dtype(dtype):
                # If very few unique values relative to total, likely categorical
                if nunique <= 20 and nunique / max(total, 1) < 0.05:
                    type_map[col] = "categorical"
                else:
                    type_map[col] = "numeric"

            elif dtype == "object" or str(dtype) in ["string", "str"]:
                # Try parsing as datetime
                try:
                    sample = df[col].dropna().head(20)
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pd.to_datetime(sample)
                    type_map[col] = "datetime"
                except (ValueError, TypeError):
                    if nunique <= 50 and nunique / max(total, 1) < 0.3:
                        type_map[col] = "categorical"
                    else:
                        type_map[col] = "text"
            else:
                type_map[col] = "unknown"

            self.logger.log(f"  {col}: {type_map[col]} (dtype={dtype}, unique={nunique})")

        self.column_types = type_map
        return type_map

    # ----------------------------------------------------------
    # 6. GENERATE METADATA
    # ----------------------------------------------------------
    def generate_metadata(self, output_dir=None):
        """
        Produce a full metadata report for the working dataset.

        Returns:
            dict with metadata and saves CSV + JSON to output_dir
        """
        if self.merged_df is None:
            self.logger.error("No working DataFrame.")
            return {}

        self.logger.section("METADATA GENERATION")
        df = self.merged_df

        metadata = {
            "shape": {"rows": df.shape[0], "columns": df.shape[1]},
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            "columns": {},
        }

        for col in df.columns:
            col_info = {
                "dtype": str(df[col].dtype),
                "detected_type": self.column_types.get(col, "unknown"),
                "missing_count": int(df[col].isnull().sum()),
                "missing_pct": round(df[col].isnull().mean() * 100, 2),
                "unique_count": int(df[col].nunique()),
                "unique_pct": round(df[col].nunique() / max(len(df), 1) * 100, 2),
            }

            # Add stats for numeric columns (exclude booleans)
            if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
                col_info["mean"] = round(float(df[col].mean()), 4) if not df[col].isnull().all() else None
                col_info["std"] = round(float(df[col].std()), 4) if not df[col].isnull().all() else None
                col_info["min"] = round(float(df[col].min()), 4) if not df[col].isnull().all() else None
                col_info["max"] = round(float(df[col].max()), 4) if not df[col].isnull().all() else None

            # Add top values for categorical
            if self.column_types.get(col) == "categorical":
                top_vals = df[col].value_counts().head(5).to_dict()
                col_info["top_values"] = {str(k): int(v) for k, v in top_vals.items()}

            metadata["columns"][col] = col_info

        self.metadata = metadata

        # Save to files
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            # Save as JSON
            json_path = os.path.join(output_dir, "metadata_report.json")
            with open(json_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)
            self.logger.success(f"Metadata saved to: {json_path}")

            # Save summary as CSV
            csv_path = os.path.join(output_dir, "metadata_summary.csv")
            summary_rows = []
            for col, info in metadata["columns"].items():
                row = {
                    "column": col,
                    "dtype": info["dtype"],
                    "detected_type": info["detected_type"],
                    "missing_count": info["missing_count"],
                    "missing_pct": info["missing_pct"],
                    "unique_count": info["unique_count"],
                    "unique_pct": info["unique_pct"],
                }
                summary_rows.append(row)
            pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
            self.logger.success(f"Summary CSV saved to: {csv_path}")

        self.logger.log(
            f"Dataset: {metadata['shape']['rows']} rows x "
            f"{metadata['shape']['columns']} cols, "
            f"{metadata['memory_usage_mb']} MB"
        )

        return metadata

    # ----------------------------------------------------------
    # 7. TARGET VARIABLE SELECTION
    # ----------------------------------------------------------
    def get_target_candidates(self):
        """
        Suggest possible target variables based on column types.
        Numeric and categorical columns are candidates.

        Returns:
            dict: {"regression": [...], "classification": [...]}
        """
        if not self.column_types:
            self.detect_column_types()

        candidates = {"regression": [], "classification": []}

        for col, ctype in self.column_types.items():
            if ctype == "numeric":
                candidates["regression"].append(col)
                # If few unique values, also a classification candidate
                if self.merged_df[col].nunique() <= 20:
                    candidates["classification"].append(col)
            elif ctype in ["categorical", "boolean"]:
                candidates["classification"].append(col)

        return candidates

    def set_target(self, target_col):
        """Set the target variable and determine problem type."""
        if self.merged_df is None:
            self.logger.error("No working DataFrame.")
            return None

        if target_col not in self.merged_df.columns:
            self.logger.error(f"Column '{target_col}' not found in dataset.")
            return None

        self.target_variable = target_col
        ctype = self.column_types.get(target_col, "unknown")

        if ctype == "numeric" and self.merged_df[target_col].nunique() > 20:
            problem_type = "regression"
        elif ctype in ["categorical", "boolean"] or self.merged_df[target_col].nunique() <= 20:
            problem_type = "classification"
        else:
            problem_type = "unknown"

        self.logger.success(
            f"Target variable set: '{target_col}' -> Problem type: {problem_type}"
        )
        return problem_type

    # ----------------------------------------------------------
    # 8. QUICK SUMMARY (print-friendly)
    # ----------------------------------------------------------
    def print_summary(self):
        """Print a human-readable summary of the loaded data."""
        if self.merged_df is None:
            print("No data loaded yet.")
            return

        df = self.merged_df
        print("\n" + "=" * 60)
        print("  DATASET SUMMARY")
        print("=" * 60)
        print(f"  Shape        : {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"  Memory       : {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        print(f"  Target       : {self.target_variable or 'Not set'}")
        print()

        # Type breakdown
        type_counts = {}
        for t in self.column_types.values():
            type_counts[t] = type_counts.get(t, 0) + 1
        print("  Column Types :")
        for t, c in sorted(type_counts.items()):
            print(f"    {t:15s} : {c}")

        # Missing data overview
        total_missing = df.isnull().sum().sum()
        total_cells = df.shape[0] * df.shape[1]
        print(f"\n  Missing Cells : {total_missing} / {total_cells} "
              f"({total_missing / max(total_cells, 1) * 100:.2f}%)")

        # Columns with most missing
        missing_cols = df.isnull().sum()
        missing_cols = missing_cols[missing_cols > 0].sort_values(ascending=False)
        if len(missing_cols) > 0:
            print(f"\n  Top Missing Columns:")
            for col, count in missing_cols.head(5).items():
                pct = count / len(df) * 100
                print(f"    {col:30s} : {count} ({pct:.1f}%)")

        print("=" * 60 + "\n")