"""
Time Series Models Module (Segment 9)
=======================================
Handles:
  1. Detecting time series data
  2. Creating lag/rolling features for ML models
  3. Traditional models: ARIMA, SARIMA, Exponential Smoothing, Theta
  4. ML models on lag features: RF, XGBoost, LightGBM
  5. Auto-ARIMA for automatic order selection
  6. Evaluation: MAE, RMSE, MAPE
  7. Forecast visualization
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

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

# Optional imports
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False


class TimeSeriesModeler:
    """Trains and evaluates time series models."""

    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        self.series = None           # The time series (pandas Series with datetime index)
        self.train = None
        self.test = None
        self.results = []
        self.forecasts = {}          # {model_name: forecast_array}
        self.report = {}
        self.is_seasonal = False
        self.seasonal_period = None

    # ----------------------------------------------------------
    # 1. DETECT TIME SERIES
    # ----------------------------------------------------------
    def detect_time_series(self, df, column_types):
        """
        Check if the dataset has time series characteristics.

        Returns:
            dict: {is_timeseries, datetime_col, value_cols, reason}
        """
        self.logger.section("TIME SERIES DETECTION")

        datetime_cols = [c for c, t in column_types.items() if t == "datetime"]
        numeric_cols = [c for c, t in column_types.items() if t == "numeric"]

        # Also check for columns that look like dates
        for col in df.columns:
            if col not in datetime_cols:
                # Skip numeric columns — integers/floats are not dates
                if pd.api.types.is_numeric_dtype(df[col]):
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        sample = df[col].dropna().head(20)
                        # Skip if values look like plain numbers
                        if sample.apply(lambda x: str(x).replace('.','').replace('-','').isdigit()).all():
                            continue
                        parsed = pd.to_datetime(sample, errors="coerce")
                    if parsed.notna().sum() >= 15:
                        datetime_cols.append(col)
                except Exception:
                    pass

        if not datetime_cols:
            self.logger.log("No datetime columns found. Not a time series dataset.")
            return {"is_timeseries": False, "reason": "no_datetime_column"}

        if not numeric_cols:
            self.logger.log("No numeric columns to forecast.")
            return {"is_timeseries": False, "reason": "no_numeric_columns"}

        self.logger.success(
            f"Time series detected: datetime={datetime_cols}, numeric={numeric_cols}"
        )
        return {
            "is_timeseries": True,
            "datetime_cols": datetime_cols,
            "value_cols": numeric_cols,
        }

    # ----------------------------------------------------------
    # 2. PREPARE TIME SERIES
    # ----------------------------------------------------------
    def prepare_series(self, df, datetime_col, value_col, freq=None):
        """
        Prepare a single time series for modeling.

        Args:
            df: DataFrame
            datetime_col: column with datetime values
            value_col: column to forecast
            freq: frequency string ('D', 'M', 'H', etc.) or None for auto
        """
        self.logger.log(f"Preparing time series: {datetime_col} -> {value_col}")

        df_ts = df[[datetime_col, value_col]].copy()

        # Convert to datetime
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_ts[datetime_col] = pd.to_datetime(df_ts[datetime_col], errors="coerce")

        df_ts = df_ts.dropna()
        df_ts = df_ts.sort_values(datetime_col)
        df_ts = df_ts.set_index(datetime_col)

        # Infer frequency
        if freq is None:
            try:
                inferred = pd.infer_freq(df_ts.index)
                if inferred:
                    freq = inferred
                    self.logger.log(f"  Inferred frequency: {freq}")
                else:
                    # Guess from median time diff
                    diffs = df_ts.index.to_series().diff().dropna()
                    median_diff = diffs.median()
                    if median_diff <= pd.Timedelta(hours=2):
                        freq = "h"
                    elif median_diff <= pd.Timedelta(days=2):
                        freq = "D"
                    elif median_diff <= pd.Timedelta(days=35):
                        freq = "MS"
                    else:
                        freq = "YS"
                    self.logger.log(f"  Estimated frequency: {freq} (median diff={median_diff})")
            except Exception:
                freq = "D"
                self.logger.log(f"  Defaulting to daily frequency")

        # Resample if needed (aggregate duplicates)
        try:
            self.series = df_ts[value_col].resample(freq).mean().dropna()
        except Exception:
            self.series = df_ts[value_col]

        # Check seasonality
        self._detect_seasonality()

        # Train/test split (last 20% for test)
        n = len(self.series)
        split_idx = int(n * (1 - self.config.TEST_SIZE))
        self.train = self.series[:split_idx]
        self.test = self.series[split_idx:]

        self.logger.success(
            f"Series prepared: {n} observations, Train={len(self.train)}, Test={len(self.test)}"
        )
        return self.series

    def _detect_seasonality(self):
        """Detect if the series has seasonal patterns."""
        n = len(self.series)

        if n < 14:
            self.is_seasonal = False
            return

        # Try common seasonal periods
        freq = self.series.index.freq
        freq_str = str(freq) if freq else ""

        if "h" in freq_str.lower() or "H" in freq_str:
            self.seasonal_period = 24  # hourly -> daily cycle
        elif "D" in freq_str or "d" in freq_str:
            self.seasonal_period = 7   # daily -> weekly cycle
        elif "M" in freq_str or "MS" in freq_str:
            self.seasonal_period = 12  # monthly -> yearly cycle
        else:
            self.seasonal_period = 7   # default

        if n >= 2 * self.seasonal_period:
            self.is_seasonal = True
            self.logger.log(f"  Seasonality detected: period={self.seasonal_period}")
        else:
            self.is_seasonal = False
            self.logger.log(f"  Series too short for seasonal models (need {2 * self.seasonal_period}, have {n})")

    # ----------------------------------------------------------
    # 3. CREATE LAG FEATURES
    # ----------------------------------------------------------
    def _create_lag_features(self, series, n_lags=10):
        """Create lag and rolling features for ML models."""
        df = pd.DataFrame({"value": series})

        # Lag features
        for i in range(1, n_lags + 1):
            df[f"lag_{i}"] = df["value"].shift(i)

        # Rolling features
        for window in [3, 5, 7]:
            if window < len(series):
                df[f"rolling_mean_{window}"] = df["value"].shift(1).rolling(window).mean()
                df[f"rolling_std_{window}"] = df["value"].shift(1).rolling(window).std()

        # Diff features
        df["diff_1"] = df["value"].diff()
        df["diff_2"] = df["value"].diff(2)

        df = df.dropna()
        X = df.drop(columns=["value"])
        y = df["value"]
        return X, y

    # ----------------------------------------------------------
    # 4. EVALUATION
    # ----------------------------------------------------------
    def _evaluate(self, model_name, actual, predicted, train_time):
        """Compute time series metrics."""
        actual = np.array(actual)
        predicted = np.array(predicted)

        # Align lengths
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]

        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)

        # MAPE (avoid division by zero)
        mask = actual != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        else:
            mape = None

        return {
            "model": model_name,
            "mae": round(mae, 6),
            "mse": round(mse, 6),
            "rmse": round(rmse, 6),
            "mape": round(mape, 4) if mape is not None else None,
            "train_time_sec": round(train_time, 4),
        }

    # ----------------------------------------------------------
    # 5. TRAIN ALL MODELS
    # ----------------------------------------------------------
    def train_all(self):
        """
        Train every suitable time series model.

        Returns:
            DataFrame of results
        """
        if self.train is None or self.test is None:
            self.logger.error("No time series prepared. Call prepare_series() first.")
            return pd.DataFrame()

        self.logger.section("TIME SERIES MODELING")
        self.logger.log(f"Training models on {len(self.train)} observations, testing on {len(self.test)}...\n")

        self.results = []
        test_len = len(self.test)

        # ==========================
        # STATISTICAL MODELS
        # ==========================
        if STATSMODELS_AVAILABLE:
            # --- Naive (last value) ---
            self._train_naive(test_len)

            # --- Moving Average ---
            self._train_moving_average(test_len)

            # --- Simple Exponential Smoothing ---
            self._train_ses(test_len)

            # --- Double Exponential Smoothing (Holt) ---
            self._train_holt(test_len)

            # --- Holt-Winters (if seasonal) ---
            if self.is_seasonal:
                self._train_holt_winters(test_len)

            # --- ARIMA variants ---
            self._train_arima(test_len)

            # --- SARIMA (if seasonal) ---
            if self.is_seasonal:
                self._train_sarima(test_len)

            # --- Auto-ARIMA ---
            if PMDARIMA_AVAILABLE:
                self._train_auto_arima(test_len)
        else:
            self.logger.warning("statsmodels not installed. Skipping statistical models.")
            self.logger.warning("Install with: pip install statsmodels")

        # ==========================
        # ML MODELS ON LAG FEATURES
        # ==========================
        self._train_ml_models()

        # Compile results
        results_df = pd.DataFrame(self.results)

        if len(results_df) > 0:
            successful = results_df[results_df["rmse"].notna()].sort_values("rmse")
            failed = results_df[results_df["rmse"].isna()]
            results_df = pd.concat([successful, failed]).reset_index(drop=True)

            if len(successful) > 0:
                best = successful.iloc[0]
                self.logger.success(
                    f"\nBest model: {best['model']} (RMSE={best['rmse']:.4f}, MAE={best['mae']:.4f})"
                )

        return results_df

    def _train_naive(self, test_len):
        """Naive forecast: last known value."""
        try:
            start = time.time()
            forecast = np.full(test_len, self.train.iloc[-1])
            t = time.time() - start

            self.forecasts["Naive (Last Value)"] = forecast
            metrics = self._evaluate("Naive (Last Value)", self.test.values, forecast, t)
            self.results.append(metrics)
            self.logger.log(f"  Naive: RMSE={metrics['rmse']:.4f}")
        except Exception as e:
            self.logger.warning(f"  Naive failed: {str(e)}")

    def _train_moving_average(self, test_len):
        """Moving average forecast."""
        for window in [3, 5, 7]:
            try:
                name = f"Moving Average ({window})"
                start = time.time()
                forecast = np.full(test_len, self.train.iloc[-window:].mean())
                t = time.time() - start

                self.forecasts[name] = forecast
                metrics = self._evaluate(name, self.test.values, forecast, t)
                self.results.append(metrics)
                self.logger.log(f"  {name}: RMSE={metrics['rmse']:.4f}")
            except Exception as e:
                self.logger.warning(f"  {name} failed: {str(e)}")

    def _train_ses(self, test_len):
        """Simple Exponential Smoothing."""
        try:
            name = "Simple Exp Smoothing"
            start = time.time()
            model = ExponentialSmoothing(self.train, trend=None, seasonal=None)
            fit = model.fit(optimized=True)
            forecast = fit.forecast(test_len).values
            t = time.time() - start

            self.forecasts[name] = forecast
            metrics = self._evaluate(name, self.test.values, forecast, t)
            self.results.append(metrics)
            self.logger.log(f"  {name}: RMSE={metrics['rmse']:.4f}")
        except Exception as e:
            self.logger.warning(f"  SES failed: {str(e)}")

    def _train_holt(self, test_len):
        """Holt's Double Exponential Smoothing."""
        try:
            name = "Holt (Double Exp)"
            start = time.time()
            model = ExponentialSmoothing(self.train, trend="add", seasonal=None)
            fit = model.fit(optimized=True)
            forecast = fit.forecast(test_len).values
            t = time.time() - start

            self.forecasts[name] = forecast
            metrics = self._evaluate(name, self.test.values, forecast, t)
            self.results.append(metrics)
            self.logger.log(f"  {name}: RMSE={metrics['rmse']:.4f}")
        except Exception as e:
            self.logger.warning(f"  Holt failed: {str(e)}")

    def _train_holt_winters(self, test_len):
        """Holt-Winters with seasonality."""
        for seasonal_type in ["add", "mul"]:
            try:
                name = f"Holt-Winters ({seasonal_type})"
                start = time.time()

                # Ensure positive values for multiplicative
                train_data = self.train.copy()
                if seasonal_type == "mul" and (train_data <= 0).any():
                    continue

                model = ExponentialSmoothing(
                    train_data,
                    trend="add",
                    seasonal=seasonal_type,
                    seasonal_periods=self.seasonal_period,
                )
                fit = model.fit(optimized=True)
                forecast = fit.forecast(test_len).values
                t = time.time() - start

                self.forecasts[name] = forecast
                metrics = self._evaluate(name, self.test.values, forecast, t)
                self.results.append(metrics)
                self.logger.log(f"  {name}: RMSE={metrics['rmse']:.4f}")
            except Exception as e:
                self.logger.warning(f"  {name} failed: {str(e)}")

    def _train_arima(self, test_len):
        """ARIMA with a few common orders."""
        orders = [(1, 1, 0), (1, 1, 1), (2, 1, 1), (2, 1, 2), (0, 1, 1)]

        for order in orders:
            try:
                name = f"ARIMA{order}"
                start = time.time()
                model = ARIMA(self.train, order=order)
                fit = model.fit()
                forecast = fit.forecast(steps=test_len).values
                t = time.time() - start

                self.forecasts[name] = forecast
                metrics = self._evaluate(name, self.test.values, forecast, t)
                self.results.append(metrics)
                self.logger.log(f"  {name}: RMSE={metrics['rmse']:.4f}")
            except Exception as e:
                self.logger.warning(f"  {name} failed: {str(e)}")

    def _train_sarima(self, test_len):
        """SARIMA with seasonal component."""
        seasonal_orders = [
            (1, 1, 1, self.seasonal_period),
            (0, 1, 1, self.seasonal_period),
        ]

        for s_order in seasonal_orders:
            try:
                name = f"SARIMA(1,1,1)x{s_order}"
                start = time.time()
                model = SARIMAX(self.train, order=(1, 1, 1), seasonal_order=s_order)
                fit = model.fit(disp=False)
                forecast = fit.forecast(steps=test_len).values
                t = time.time() - start

                self.forecasts[name] = forecast
                metrics = self._evaluate(name, self.test.values, forecast, t)
                self.results.append(metrics)
                self.logger.log(f"  {name}: RMSE={metrics['rmse']:.4f}")
            except Exception as e:
                self.logger.warning(f"  {name} failed: {str(e)}")

    def _train_auto_arima(self, test_len):
        """Auto-ARIMA for automatic order selection."""
        try:
            name = "Auto-ARIMA"
            start = time.time()
            model = auto_arima(
                self.train,
                seasonal=self.is_seasonal,
                m=self.seasonal_period if self.is_seasonal else 1,
                suppress_warnings=True,
                stepwise=True,
                max_p=3, max_q=3, max_d=2,
            )
            forecast = model.predict(n_periods=test_len)
            t = time.time() - start

            self.forecasts[name] = forecast
            metrics = self._evaluate(name, self.test.values, forecast, t)
            self.results.append(metrics)
            self.logger.log(f"  {name}: RMSE={metrics['rmse']:.4f} (order={model.order})")
        except Exception as e:
            self.logger.warning(f"  Auto-ARIMA failed: {str(e)}")

    def _train_ml_models(self):
        """Train ML models on lag features."""
        self.logger.log("\n  Training ML models on lag features...")

        n_lags = min(10, len(self.train) // 3)
        if n_lags < 3:
            self.logger.warning("  Not enough data for lag features.")
            return

        X_full, y_full = self._create_lag_features(self.series, n_lags=n_lags)

        if len(X_full) < 10:
            self.logger.warning("  Not enough data after lag creation.")
            return

        # Split aligned with time
        train_end = len(self.train) - n_lags - 5  # account for lost rows from lags
        if train_end < 5:
            self.logger.warning("  Train set too small after lag creation.")
            return

        X_train_ml = X_full.iloc[:train_end]
        y_train_ml = y_full.iloc[:train_end]
        X_test_ml = X_full.iloc[train_end:]
        y_test_ml = y_full.iloc[train_end:]

        if len(X_test_ml) == 0:
            self.logger.warning("  No test data for ML models.")
            return

        ml_models = {
            "Linear Reg (Lag)": LinearRegression(),
            "Random Forest (Lag)": RandomForestRegressor(
                n_estimators=100, random_state=self.config.RANDOM_STATE, n_jobs=-1
            ),
            "Gradient Boosting (Lag)": GradientBoostingRegressor(
                n_estimators=100, random_state=self.config.RANDOM_STATE
            ),
        }

        # XGBoost
        try:
            from xgboost import XGBRegressor
            ml_models["XGBoost (Lag)"] = XGBRegressor(
                n_estimators=100, random_state=self.config.RANDOM_STATE,
                verbosity=0, n_jobs=-1,
            )
        except ImportError:
            pass

        # LightGBM
        try:
            from lightgbm import LGBMRegressor
            ml_models["LightGBM (Lag)"] = LGBMRegressor(
                n_estimators=100, random_state=self.config.RANDOM_STATE,
                verbose=-1, n_jobs=-1,
            )
        except ImportError:
            pass

        for name, model in ml_models.items():
            try:
                start = time.time()
                model.fit(X_train_ml, y_train_ml)
                forecast = model.predict(X_test_ml)
                t = time.time() - start

                self.forecasts[name] = forecast
                metrics = self._evaluate(name, y_test_ml.values, forecast, t)
                self.results.append(metrics)
                self.logger.log(f"  {name}: RMSE={metrics['rmse']:.4f}")
            except Exception as e:
                self.logger.warning(f"  {name} failed: {str(e)}")

    # ----------------------------------------------------------
    # 6. SAVE RESULTS
    # ----------------------------------------------------------
    def save_results(self, results_df, output_dir):
        """Save results, plots, and forecasts."""
        trad_dir = os.path.join(output_dir, "traditional")
        os.makedirs(trad_dir, exist_ok=True)

        # Save CSV
        csv_path = os.path.join(trad_dir, "timeseries_results.csv")
        results_df.to_csv(csv_path, index=False)
        self.logger.success(f"Results saved to: {csv_path}")

        successful = results_df[results_df["rmse"].notna()].copy()
        if len(successful) > 0:
            self._plot_rmse_comparison(successful, trad_dir)
            self._plot_forecasts(successful, trad_dir)

        # Save summary
        if len(successful) > 0:
            best = successful.iloc[0]
            summary = {
                "best_model": best["model"],
                "rmse": best["rmse"],
                "mae": best["mae"],
                "mape": best["mape"],
                "series_length": len(self.series),
                "train_size": len(self.train),
                "test_size": len(self.test),
                "is_seasonal": self.is_seasonal,
                "seasonal_period": self.seasonal_period,
                "total_models_tested": len(results_df),
                "successful_models": len(successful),
            }
            summary_path = os.path.join(trad_dir, "best_model_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, default=str)

        return csv_path

    def _plot_rmse_comparison(self, df, output_dir):
        """Bar chart of RMSE."""
        fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.35)))
        sorted_df = df.sort_values("rmse", ascending=True)
        colors = ["green" if i == 0 else "steelblue" for i in range(len(sorted_df))]
        ax.barh(range(len(sorted_df)), sorted_df["rmse"], color=colors, edgecolor="black")
        ax.set_yticks(range(len(sorted_df)))
        ax.set_yticklabels(sorted_df["model"], fontsize=9)
        ax.set_xlabel("RMSE (lower is better)")
        ax.set_title("Time Series Models - RMSE Comparison")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "rmse_comparison.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)

    def _plot_forecasts(self, df, output_dir):
        """Plot actual vs forecast for top models."""
        top_models = df.head(4)["model"].tolist()
        n_plots = len(top_models)

        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))
        if n_plots == 1:
            axes = [axes]

        for i, name in enumerate(top_models):
            if name in self.forecasts:
                forecast = self.forecasts[name]

                # Plot train
                axes[i].plot(range(len(self.train)), self.train.values,
                             label="Train", color="steelblue", alpha=0.7)

                # Plot test
                test_start = len(self.train)
                test_range = range(test_start, test_start + len(self.test))
                axes[i].plot(test_range, self.test.values,
                             label="Actual", color="black", linewidth=2)

                # Plot forecast
                forecast_len = min(len(forecast), len(self.test))
                forecast_range = range(test_start, test_start + forecast_len)
                axes[i].plot(forecast_range, forecast[:forecast_len],
                             label="Forecast", color="red", linestyle="--", linewidth=2)

                rmse = df[df["model"] == name]["rmse"].values[0]
                axes[i].set_title(f"{name} (RMSE={rmse:.4f})")
                axes[i].legend()
                axes[i].set_xlabel("Time Index")
                axes[i].set_ylabel("Value")

        plt.suptitle("Time Series Forecasts (Top Models)", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "forecast_plots.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)

    # ----------------------------------------------------------
    # 7. PRINT SUMMARY
    # ----------------------------------------------------------
    def print_results_summary(self, results_df):
        """Print ranking table."""
        print("\n" + "=" * 85)
        print("  TIME SERIES MODEL RESULTS")
        print("=" * 85)

        successful = results_df[results_df["rmse"].notna()]

        if len(successful) == 0:
            print("  No models completed successfully.")
            print("=" * 85)
            return

        print(f"\n  Series Length: {len(self.series)} | Train: {len(self.train)} | Test: {len(self.test)}")
        print(f"  Seasonal: {self.is_seasonal} (period={self.seasonal_period})")

        print(f"\n  {'Rank':<6}{'Model':<35}{'RMSE':<14}{'MAE':<14}{'MAPE(%)':<12}{'Time(s)':<10}")
        print("  " + "-" * 85)

        for i, (_, row) in enumerate(successful.iterrows(), 1):
            mape_str = f"{row['mape']:.2f}" if row['mape'] is not None else "N/A"
            print(
                f"  {i:<6}{row['model']:<35}{row['rmse']:<14.4f}"
                f"{row['mae']:<14.4f}{mape_str:<12}{row['train_time_sec']:<10.3f}"
            )

        failed = results_df[results_df["rmse"].isna()]
        if len(failed) > 0:
            print(f"\n  Failed Models ({len(failed)}):")
            for _, row in failed.iterrows():
                err = row.get("error", "Unknown")
                print(f"    - {row['model']}: {err}")

        best = successful.iloc[0]
        print(f"\n  BEST MODEL: {best['model']}")
        print(f"  RMSE = {best['rmse']:.6f} | MAE = {best['mae']:.6f}")
        print("=" * 85 + "\n")