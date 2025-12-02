import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import timedelta
import pickle
import os
from python.utils import project_path


class MultiZoneLoadPredictor:
    """
    Predicts electricity loads for multiple zones simultaneously using
    a multi-output regression model.
    """

    def __init__(
        self,
        model,
        zones=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"],
        model_description="custom",
    ):
        """
        Initialize the multi-zone load predictor with a specified model.

        Parameters:
        -----------
        model : scikit-learn model, optional
            The ML model to use for prediction.
        zones : list
            List of zones to predict.
        model_description : str, optional
            Description of model.
        """
        self.model = model
        self.model_description = model_description
        self.zones = zones

        self.scaler = StandardScaler()

    def preprocess_data(self, temp_data, load_data, lags=[1, 2], temp_varname="T2C"):
        """
        Preprocess the temperature and load data for all zones.

        Parameters:
        -----------
        temp_data : DataFrame
            Temperature data with columns 'zone', 'time', temp_varname
        load_data : DataFrame
            Load data with columns 'time', 'zone', 'load_MW'

        Returns:
        --------
        DataFrame
            Preprocessed data with features and multi-zone targets
        """
        # Convert time columns to datetime if they aren't already
        temp_data["time"] = pd.to_datetime(temp_data["time"])
        load_data["time"] = pd.to_datetime(load_data["time"])

        # Rename columns for consistency
        temp_data = temp_data.rename(columns={"time": "datetime"})
        load_data = load_data.rename(columns={"time": "datetime"})

        # Pivot the load data to have one column per zone
        load_pivot = load_data.pivot_table(
            index="datetime",
            columns="zone",
            values="load_MW",
        ).reset_index()

        # Create datetime-indexed dataframe for joining
        temp_data_wide = temp_data.pivot_table(
            index="datetime",
            columns="zone",
            values=temp_varname,
        )

        # Rename temperature columns to avoid confusion with load columns
        temp_columns = {
            zone: f"{temp_varname}_{zone}" for zone in temp_data_wide.columns
        }
        temp_data_wide = temp_data_wide.rename(columns=temp_columns)
        temp_data_wide = temp_data_wide.reset_index()

        # Merge load and temperature data
        data = pd.merge(load_pivot, temp_data_wide, on="datetime", how="inner")

        # Extract temporal features
        data["day_of_week"] = data["datetime"].dt.dayofweek
        data["day_of_year"] = data["datetime"].dt.dayofyear
        data["hour"] = data["datetime"].dt.hour
        data["month"] = data["datetime"].dt.month
        data["year"] = data["datetime"].dt.year

        # Calculate previous day's average load for each zone
        data = data.sort_values("datetime")
        data["date"] = data["datetime"].dt.date

        # Create previous day average load features for each zone
        for zone in self.zones:
            for lag in lags:
                # Group by date and calculate daily average for each zone
                zone_daily_avg = data.groupby("date")[zone].mean().reset_index()
                zone_daily_avg[f"lag{lag}_date"] = pd.to_datetime(
                    zone_daily_avg["date"]
                ) + timedelta(days=lag)
                zone_daily_avg = zone_daily_avg.rename(
                    columns={zone: f"lag{lag}_avg_{zone}"}
                )

                # Merge with previous day's average
                data["date"] = pd.to_datetime(data["date"])
                data = pd.merge(
                    data,
                    zone_daily_avg[[f"lag{lag}_date", f"lag{lag}_avg_{zone}"]],
                    left_on="date",
                    right_on=f"lag{lag}_date",
                    how="left",
                )
                data = data.drop(columns=[f"lag{lag}_date"], errors="ignore")

                # Fill missing values for the first day with the zone's mean
                if data[f"lag{lag}_avg_{zone}"].isna().any():
                    data[f"lag{lag}_avg_{zone}"] = data[f"lag{lag}_avg_{zone}"].fillna(
                        data[zone].mean()
                    )

        return data

    def prepare_features_target(self, data, temp_varname="T2C"):
        """
        Prepare features and multi-zone target variables from preprocessed data.

        Parameters:
        -----------
        data : DataFrame
            Preprocessed data
        temp_varname : str, optional
            Name of the temperature variable. Default is "T2C" (for TGW).

        Returns:
        --------
        X : DataFrame
            Feature matrix
        y : DataFrame
            Multi-zone target matrix
        """
        # Basic temporal features
        base_features = ["hour", "day_of_year"]

        # Temperature features for each zone
        temp_features = [
            col for col in data.columns if col.startswith(f"{temp_varname}_")
        ]

        # Previous day average load features
        prev_load_features = [col for col in data.columns if col.startswith("lag")]

        # Combine all features
        feature_cols = base_features + temp_features + prev_load_features

        # Select features and targets
        X = data[feature_cols]
        y = data[self.zones]

        return X, y

    def train(
        self,
        temp_data,
        load_data,
        test_split=0.2,
        temp_varname="T2C",
        lags=[1, 2],
        random_state=42,
    ):
        """
        Train the model to predict loads for all zones simultaneously.

        Parameters:
        -----------
        temp_data : DataFrame
            Temperature data
        load_data : DataFrame
            Load data
        test_split : float, optional
            If float, the proportion of data to use for testing.
            If list, the years to use for testing.
        random_state : int, optional
            Random state for reproducibility.
        lags : list, optional
            Number of lagged days to include as temperature predictors. Default is [1, 2].
        temp_varname : str, optional
            Name of the temperature variable. Default is "T2C" (for TGW).

        Returns:
        --------
        dict
            Training results including metrics
        """
        # Preprocess data
        data = self.preprocess_data(temp_data, load_data)

        # Prepare features and target
        X, y = self.prepare_features_target(data)

        # Sort data chronologically for time-series split
        data = data.sort_values("datetime")
        X = X.loc[data.index]
        y = y.loc[data.index]

        # Train/test split - using chronological split for time series data
        if type(test_split) is float:
            split_idx = int(len(X) * (1 - test_split))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        elif type(test_split) is list:
            split_idx = data["year"].isin(test_split)
            X_train, X_test = X[~split_idx], X[split_idx]
            y_train, y_test = y[~split_idx], y[split_idx]
        else:
            print("Invalid split type")
            return None

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train the model
        self.model.fit(X_train_scaled, y_train)

        # Get predictions
        y_pred_test = self.predict(X_test_scaled)
        y_pred_train = self.model.predict(X_train_scaled)

        # Calculate metrics for each zone
        metrics = {}
        for i, zone in enumerate(self.zones):
            zone_metrics = {
                "rmse_train": np.sqrt(
                    mean_squared_error(y_train[zone], y_pred_train[:, i])
                ),
                "mae_train": mean_absolute_error(y_train[zone], y_pred_train[:, i]),
                "r2_train": r2_score(y_train[zone], y_pred_train[:, i]),
                "rmse_test": np.sqrt(
                    mean_squared_error(y_test[zone], y_pred_test[:, i])
                ),
                "mae_test": mean_absolute_error(y_test[zone], y_pred_test[:, i]),
                "r2_test": r2_score(y_test[zone], y_pred_test[:, i]),
            }
            metrics[zone] = zone_metrics

        # Calculate overall metrics
        overall_metrics = {
            "rmse_train": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "mae_train": mean_absolute_error(y_train, y_pred_train),
            "r2_train": r2_score(y_train, y_pred_train),
            "rmse_test": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "mae_test": mean_absolute_error(y_test, y_pred_test),
            "r2_test": r2_score(y_test, y_pred_test),
        }
        metrics["overall"] = overall_metrics

        # Store test results and metrics for visualization
        self.results = {
            "y_true_train": y_train.to_numpy(),
            "y_pred_train": y_pred_train,
            "train_datetimes": data[~split_idx]["datetime"].to_numpy(),
            "train_temps": data[~split_idx][
                [f"{temp_varname}_{zone}" for zone in self.zones]
            ].to_numpy(),
            "y_true_test": y_test.to_numpy(),
            "y_pred_test": y_pred_test,
            "test_datetimes": data[split_idx]["datetime"].to_numpy(),
            "test_temps": data[split_idx][
                [f"{temp_varname}_{zone}" for zone in self.zones]
            ].to_numpy(),
            "metrics": metrics,
            "feature_names": X.columns.tolist(),
        }

        # Store Dec 31st averages for new predictions
        self.store_training_statistics(data, lags)

        return metrics

    def store_training_statistics(self, training_data, lags=[1, 2]):
        """
        Store training data statistics for future prediction initialization.
        Call this after training to store necessary statistics.

        Parameters:
        -----------
        training_data : DataFrame
            The preprocessed training data used for model training
        """

        if not hasattr(self, "results"):
            raise ValueError("Model has not been trained yet")

        # Store December 31st daily averages for each zone and each lag
        self.dec31_averages = {}

        # Sort data by datetime to get chronological order
        training_data_sorted = training_data.sort_values("datetime")

        # Get Dec 31st averages
        for lag in lags:
            for zone in self.zones:
                # Find December 31st values for each year in training data
                dec31_values = []

                # Get unique years in training data
                years = training_data_sorted["datetime"].dt.year.unique()

                for year in years:
                    # Get December 31st data for this year
                    dec31_data = training_data_sorted[
                        (training_data_sorted["datetime"].dt.month == 12)
                        & (training_data_sorted["datetime"].dt.day == 31)
                        & (training_data_sorted["datetime"].dt.year == year)
                    ]

                    if not dec31_data.empty and zone in dec31_data.columns:
                        # Calculate daily average for Dec 31st
                        daily_avg = dec31_data[zone].mean()
                        if not pd.isna(daily_avg):
                            dec31_values.append(daily_avg)

                # Store the mean of all Dec 31st daily averages
                if dec31_values:
                    self.dec31_averages[f"lag{lag}_avg_{zone}"] = np.mean(dec31_values)
                else:
                    # Fallback to overall mean if no Dec 31st data found
                    print(
                        f"Warning: No Dec 31st data found for zone {zone}, using overall mean"
                    )
                    zone_idx = self.zones.index(zone)
                    self.dec31_averages[f"lag{lag}_avg_{zone}"] = np.mean(
                        self.results["y_true"][:, zone_idx]
                    )

    def predict(self, X):
        """
        Make predictions for all zones and ensure they are non-negative.

        Parameters:
        -----------
        X : array-like or DataFrame
            Feature matrix

        Returns:
        --------
        array
            Non-negative predictions for all zones
        """
        # Check if X is a DataFrame
        if isinstance(X, pd.DataFrame):
            # Scale features
            X_scaled = self.scaler.transform(X)
        else:
            # Assume X is already scaled
            X_scaled = X

        # Make predictions
        predictions = self.model.predict(X_scaled)

        # Ensure non-negative predictions
        if isinstance(predictions, pd.DataFrame):
            predictions = predictions.clip(lower=0)
        else:
            predictions = np.maximum(0, predictions)

        return predictions

    def evaluate(self):
        """
        Return evaluation metrics for all zones.

        Returns:
        --------
        dict
            Evaluation metrics by zone
        """
        if not hasattr(self, "results"):
            raise ValueError("Model has not been trained yet")

        return self.results["metrics"]

    def plot_results(self, zone, filepath=None):
        """
        Plot actual vs predicted values for specified zones.

        Parameters:
        -----------
        zones : list of str, optional
            zones to plot. If None, plots all zones.
        """
        if not hasattr(self, "results"):
            raise ValueError("Model has not been trained yet")

        # Get required data
        zone_idx = self.zones.index(zone)
        y_true = self.results["y_true_test"][:, zone_idx]
        y_pred = self.results["y_pred_test"][:, zone_idx]
        temp = self.results["test_temps"][:, zone_idx]
        # Drop zeros for plot
        inds = y_true > 0
        y_true = y_true[inds]
        y_pred = y_pred[inds]
        temp = temp[inds]
        # datetimes = self.results['test_datetimes']

        fig, axs = plt.subplots(3, 1, figsize=(8, 8))
        fig.suptitle(f"Actual vs Predicted Load for Zone {zone}")

        # Timeseries plot
        ax = axs[0]
        ax.plot(y_true, label="Actual", alpha=0.75)
        ax.plot(y_pred, label="Prediction", alpha=0.75)
        ax.set_xlabel("Hour")
        ax.set_ylabel("Load (MW)")
        ax.grid(alpha=0.4)
        ax.legend()

        # Temperature scatter plot
        ax = axs[1]
        ax.scatter(temp, y_true, label="Actual", s=5, alpha=0.5)
        ax.scatter(temp, y_pred, label="Prediction", s=5, alpha=0.5)
        ax.set_xlabel("Zonal temperature (C)")
        ax.set_ylabel("Load (MW)")
        ax.grid(alpha=0.4)
        ax.legend()

        # Scatter plot
        ax = axs[2]
        ax.scatter(y_true, y_pred, s=5, alpha=0.5)
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", color="black")
        ax.set_xlabel("Actual load (MW)")
        ax.set_ylabel("Predicted load (MW)")
        ax.grid(alpha=0.4)

        plt.tight_layout()

        if filepath is not None:
            plt.savefig(filepath, bbox_inches="tight")
        else:
            plt.show()

    def plot_feature_importance(self):
        """
        Plot feature importance for all zones (if the model supports it).
        """
        if not hasattr(self, "results"):
            raise ValueError("Model has not been trained yet")

        # Check if the model has feature_importances_ attribute
        if hasattr(self.model, "feature_importances_"):
            # Direct access for RandomForestRegressor
            importances = self.model.feature_importances_
            feature_names = self.results["feature_names"]
        elif self.model_description == "random_forest":
            # Direct access for RandomForestRegressor
            importances = self.model.feature_importances_
            feature_names = self.results["feature_names"]
        elif hasattr(self.model, "estimators_"):
            # For MultiOutputRegressor, get average importance across all outputs
            all_importances = []
            for estimator in self.model.estimators_:
                if hasattr(estimator, "feature_importances_"):
                    all_importances.append(estimator.feature_importances_)

            if all_importances:
                importances = np.mean(all_importances, axis=0)
                feature_names = self.results["feature_names"]
            else:
                print("Model doesn't support feature importance visualization")
                return
        else:
            print("Model doesn't support feature importance visualization")
            return

        # Sort feature importances
        indices = np.argsort(importances)[::-1]
        sorted_importances = importances[indices]
        sorted_features = [feature_names[i] for i in indices]

        # Create a plot
        plt.figure(figsize=(12, 8))
        plt.title("Feature Importance (All zones)")
        plt.bar(range(len(sorted_importances)), sorted_importances)
        plt.xticks(range(len(sorted_importances)), sorted_features, rotation=90)
        plt.tight_layout()
        plt.show()

    def save_model(self, filepath="multi_zone_model.pkl"):
        """
        Save the trained model and related data.

        Parameters:
        -----------
        filepath : str, optional
            The filepath to save the model to.
        """
        if not hasattr(self, "results"):
            raise ValueError("Model has not been trained yet")

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "zones": self.zones,
            "results": self.results,
            "model_description": self.model_description,
            "dec31_averages": self.dec31_averages,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath="multi_zone_model.pkl"):
        """
        Load a trained model and related data.

        Parameters:
        -----------
        filepath : str, optional
            The filepath to load the model from.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} not found")

        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.zones = model_data["zones"]
        self.results = model_data["results"]
        self.model_description = model_data.get("model_description", "custom")
        self.dec31_averages = model_data.get("dec31_averages", {})

        print(f"Model loaded from {filepath}")

    def get_zone_prediction(self, X, zone):
        """
        Get predictions for a specific zone.

        Parameters:
        -----------
        X : array-like or DataFrame
            Feature matrix
        zone : str
            zone to get predictions for

        Returns:
        --------
        array
            Predictions for the specified zone
        """
        if zone not in self.zones:
            raise ValueError(f"zone {zone} not found in trained model")

        # Get zone index
        zone_idx = self.zones.index(zone)

        # Get predictions for all zones
        all_predictions = self.predict(X)

        # Extract predictions for specified zone
        return all_predictions[:, zone_idx]

    def predict_future_loads(self, temp_data, temp_varname="T2C", lags=[1, 2]):
        """
        Predict electricity loads for future periods using new temperature data.
        Assumes predictions start from January 1st and uses December 31st historical
        averages for lag feature initialization.

        Parameters:
        -----------
        temp_data : DataFrame
            Future temperature data with columns 'zone', 'time', temp_varname
        temp_varname : str, optional
            Name of the temperature variable. Default is "T2C"
        lags : list, optional
            Number of lagged days to include. Should match training configuration.

        Returns:
        --------
        DataFrame
            Predictions with columns: datetime, zone_A, zone_B, ..., zone_K
        """
        if not hasattr(self, "results"):
            raise ValueError("Model has not been trained yet")

        if not hasattr(self, "dec31_averages"):
            raise ValueError(
                "December 31st statistics not found. Call store_training_statistics() after training."
            )

        # Prepare temperature data
        temp_data = temp_data.copy()
        temp_data["time"] = pd.to_datetime(temp_data["time"])
        temp_data = temp_data.rename(columns={"time": "datetime"})

        # Pivot temperature data to wide format
        temp_data_wide = temp_data.pivot_table(
            index="datetime",
            columns="zone",
            values=temp_varname,
        )

        # Rename temperature columns
        temp_columns = {
            zone: f"{temp_varname}_{zone}" for zone in temp_data_wide.columns
        }
        temp_data_wide = temp_data_wide.rename(columns=temp_columns)
        temp_data_wide = temp_data_wide.reset_index()

        # Sort by datetime
        temp_data_wide = temp_data_wide.sort_values("datetime")

        # Extract temporal features
        temp_data_wide["day_of_week"] = temp_data_wide["datetime"].dt.dayofweek
        temp_data_wide["day_of_year"] = temp_data_wide["datetime"].dt.dayofyear
        temp_data_wide["hour"] = temp_data_wide["datetime"].dt.hour
        temp_data_wide["month"] = temp_data_wide["datetime"].dt.month
        temp_data_wide["year"] = temp_data_wide["datetime"].dt.year
        temp_data_wide["date"] = temp_data_wide["datetime"].dt.date

        # Initialize lag features using December 31st averages
        lag_features = self.dec31_averages.copy()

        # Create containers for predictions
        predictions_list = []

        # Get unique dates for daily aggregation
        unique_dates = sorted(temp_data_wide["date"].unique())

        # Process each day
        for current_date in unique_dates:
            # Get data for current day
            day_data = temp_data_wide[temp_data_wide["date"] == current_date].copy()

            # Add current lag features to the day's data
            for lag_col, lag_val in lag_features.items():
                day_data[lag_col] = lag_val

            # Use the exact feature names and order from training
            if not hasattr(self, "results") or "feature_names" not in self.results:
                raise ValueError(
                    "Feature names from training not found. Ensure model was trained properly."
                )

            feature_cols = self.results["feature_names"]

            # Ensure all required features are present, add missing ones with defaults
            for col in feature_cols:
                if col not in day_data.columns:
                    if col.startswith("lag"):
                        # This shouldn't happen as we add lag features above
                        raise ValueError(
                            f"Missing lag feature: {col}. Check lag feature initialization."
                        )
                    elif col.startswith(f"{temp_varname}_"):
                        # This shouldn't happen as we pivot temperature data above
                        raise ValueError(
                            f"Missing temperature feature: {col}. Check temperature data."
                        )
                    else:
                        # Handle any other missing features with warning
                        print(f"Warning: Missing feature {col}, setting to 0")
                        day_data[col] = 0

            # Select features in the exact same order as training
            X_day = day_data[feature_cols]

            # Make predictions for the day
            day_predictions = self.predict(X_day)

            # Store predictions with datetime
            for i, datetime_val in enumerate(day_data["datetime"]):
                pred_row = {"datetime": datetime_val}
                for j, zone in enumerate(self.zones):
                    pred_row[zone] = day_predictions[i, j]
                predictions_list.append(pred_row)

            # Update lag features with daily average of predictions
            daily_avg_predictions = np.mean(day_predictions, axis=0)

            # Shift lag features (lag2 becomes lag3, lag1 becomes lag2, current becomes lag1)
            for zone_idx, zone in enumerate(self.zones):
                # Update lag features for next day
                for lag in sorted(lags, reverse=True):  # Process in reverse order
                    if lag > 1:
                        # Shift older lags
                        if f"lag{lag - 1}_avg_{zone}" in lag_features:
                            lag_features[f"lag{lag}_avg_{zone}"] = lag_features[
                                f"lag{lag - 1}_avg_{zone}"
                            ]
                    else:
                        # Set lag1 to current day's average prediction
                        lag_features[f"lag{lag}_avg_{zone}"] = daily_avg_predictions[
                            zone_idx
                        ]

        # Convert predictions to DataFrame
        predictions_df = pd.DataFrame(predictions_list)
        predictions_df = predictions_df.sort_values("datetime").reset_index(drop=True)

        return predictions_df


def load_and_prepare_data(temp_file, load_file):
    """
    Preprocess temperature and load data from files.

    Parameters:
    -----------
    temp_file : str
        Path to temperature data file
    load_file : str
        Path to load data file

    Returns:
    --------
    tuple
        Temperature and load DataFrames
    """
    try:
        # Temperature data
        temp_data = pd.read_csv(temp_file)
        # Ensure datetime columns are properly formatted
        temp_data["time"] = pd.to_datetime(temp_data["time"])

        # Add timezone indicator for temperature data
        temp_data["time"] = temp_data["time"].dt.tz_localize("UTC")

        if load_file is not None:
            # Load data
            load_data = pd.read_csv(load_file)
            # Ensure datetime columns are properly formatted
            load_data["time"] = pd.to_datetime(load_data["time"])
            load_data["time"] = load_data["time"].dt.tz_localize(
                "America/New_York", ambiguous="NaT"
            )
            load_data["time"] = load_data["time"].dt.tz_convert("UTC")
        else:
            load_data = None

        return temp_data, load_data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def baseline_load_to_bus(predictor, temp_data_path, temp_varname="T2C"):
    """
    Disaggregate baseline load to bus level.

    Parameters:
    -----------
    predictor : MultiZoneLoadPredictor
        Trained model
    temp_data : DataFrame
        Temperature data
    """
    # Get future predictions
    temp_data, _ = load_and_prepare_data(temp_data_path, None)

    future_predictions = predictor.predict_future_loads(
        temp_data=temp_data,
        temp_varname=temp_varname,
    )

    # Get bus load data
    df_npcc = pd.read_csv(f"{project_path}/data/grid/npcc_new.csv")

    # Make sure sum of load ratios > 0.
    mask = df_npcc.groupby("zoneID")["sumLoadP0"].transform("sum") == 0
    df_npcc.loc[mask, "sumLoadP0"] = 0.01

    # Get zonal ratios
    df_npcc = pd.merge(
        df_npcc,
        pd.DataFrame(
            data={"zonal_sumLoadP0": df_npcc.groupby("zoneID")["sumLoadP0"].sum()}
        ),
        on="zoneID",
        how="outer",
    )
    df_npcc["ratio"] = df_npcc["sumLoadP0"] / df_npcc["zonal_sumLoadP0"]
    ratios = df_npcc[["busIdx", "ratio", "zoneID"]].rename(
        columns={"busIdx": "bus_id", "zoneID": "zone"}
    )

    # Merge predictions with bus load data
    df_out = pd.merge(
        future_predictions.melt(
            id_vars=["datetime"], var_name="zone", value_name="zonal_load_MW"
        ),
        ratios,
        on="zone",
        how="outer",
    )
    df_out["load_MW"] = df_out["zonal_load_MW"] * df_out["ratio"]

    # Drop ratio
    df_out = df_out.drop(columns=["ratio"])

    return df_out
