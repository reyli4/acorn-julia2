import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import timedelta
import pickle
import os


class LoadPredictor:
    """
    A class for predicting electricity loads using machine learning models.
    Supports different zones and can easily switch between different models.
    """

    def __init__(self, model=None):
        """
        Initialize the load predictor with a specified model.

        Parameters:
        -----------
        model : scikit-learn model, optional
            The ML model to use for prediction. Default is RandomForestRegressor.
        """
        self.model = model or RandomForestRegressor(n_estimators=100, random_state=42)
        self.zone_models = {}

    def preprocess_data(self, temp_data, load_data, zone, temp_varname="T2C"):
        """
        Preprocess the temperature and load data for a specific zone.

        Parameters:
        -----------
        temp_data : DataFrame
            Temperature data with columns 'zone', 'time', temp_varname
        load_data : DataFrame
            Load data with columns 'time', 'Zone', 'load_MW'
        zone : str
            The zone to filter data for

        Returns:
        --------
        DataFrame
            Preprocessed data with features and target
        """
        # Filter data for the specified zone
        zone_temp = temp_data[temp_data["zone"] == zone].copy()
        zone_load = load_data[load_data["zone"] == zone].copy()

        # Convert time columns to datetime if they aren't already
        zone_temp["time"] = pd.to_datetime(zone_temp["time"])
        zone_load["time"] = pd.to_datetime(zone_load["time"])

        # Rename columns for consistency
        zone_temp = zone_temp.rename(columns={"time": "datetime"})
        zone_load = zone_load.rename(columns={"time": "datetime"})

        # Merge the datasets on datetime
        data = pd.merge(
            zone_load, zone_temp[["datetime", temp_varname]], on="datetime", how="inner"
        )

        # Extract temporal features
        data["day_of_week"] = data["datetime"].dt.dayofweek
        data["day_of_year"] = data["datetime"].dt.dayofyear
        data["hour"] = data["datetime"].dt.hour
        data["month"] = data["datetime"].dt.month
        data["year"] = data["datetime"].dt.year
        print(f"Modeling years: {data['year'].min()} - {data['year'].max()}")

        # Calculate previous day's average load
        data = data.sort_values("datetime")
        data["date"] = data["datetime"].dt.date

        # Group by date and calculate daily average
        daily_avg = data.groupby("date")["load_MW"].mean().reset_index()
        daily_avg["date"] = pd.to_datetime(daily_avg["date"])
        daily_avg["prev_date"] = daily_avg["date"] - timedelta(days=1)
        daily_avg = daily_avg.rename(columns={"load_MW": "prev_day_avg_load"})

        # Merge with previous day's average
        data["date"] = pd.to_datetime(data["date"])
        data = pd.merge(
            data,
            daily_avg[["prev_date", "prev_day_avg_load"]],
            left_on="date",
            right_on="prev_date",
            how="left",
        )

        # Fill missing values for the first day with the overall mean
        if data["prev_day_avg_load"].isna().any():
            data["prev_day_avg_load"] = data["prev_day_avg_load"].fillna(
                data["load_MW"].mean()
            )

        return data

    def prepare_features_target(self, data, temp_varname="T2C"):
        """
        Prepare features and target variables from preprocessed data.

        Parameters:
        -----------
        data : DataFrame
            Preprocessed data

        Returns:
        --------
        X : DataFrame
            Feature matrix
        y : Series
            Target variable
        """
        # Select features
        features = [temp_varname, "day_of_week", "day_of_year", "prev_day_avg_load"]
        X = data[features]
        y = data["load_MW"]

        return X, y, features

    def train(
        self,
        temp_data,
        load_data,
        zone,
        test_split=0.2,
        temp_varname="T2C",
        random_state=None,
    ):
        """
        Train the model for a specific zone.

        Parameters:
        -----------
        temp_data : DataFrame
            Temperature data
        load_data : DataFrame
            Load data
        zone : str
            The zone to train for
        test_split : float, optional
            Testing split: if fraction between 0-1, proportion of data to use for testing;
                            if list of ints, years to use as testing
        random_state : int, optional
            Random state for reproducibility.

        Returns:
        --------
        dict
            Training results including model, metrics, and preprocessor
        """
        # Preprocess data
        data = self.preprocess_data(temp_data, load_data, zone)

        # Prepare features and target
        X, y, features = self.prepare_features_target(data)

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
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train the model
        self.model.fit(X_train_scaled, y_train)

        # Get predictions
        y_pred_test = self.predict(X_test_scaled)
        y_pred_train = self.model.predict(X_train_scaled)

        # Calculate metrics
        metrics = {
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "test_mae": mean_absolute_error(y_test, y_pred_test),
            "test_r2": r2_score(y_test, y_pred_test),
            "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "train_mae": mean_absolute_error(y_train, y_pred_train),
            "train_r2": r2_score(y_train, y_pred_train),
        }

        # Store model and preprocessor for this zone
        test_results = pd.DataFrame(
            data={
                "datetime": data[split_idx]["datetime"].to_numpy(),
                "y_true": y_test.to_numpy(),
                "y_pred": y_pred_test,
                "temp": X_test[temp_varname].to_numpy(),
            }
        )
        self.zone_models[zone] = {
            "model": self.model,
            "scaler": scaler,
            "features": features,
            "metrics": metrics,
            "test_results": test_results,
            "test_split": test_split,
        }

        return self.zone_models[zone]

    def predict(self, X):
        """
        Make predictions and ensure they are non-negative.

        Parameters:
        -----------
        X : array-like
            Feature matrix

        Returns:
        --------
        array
            Non-negative predictions
        """
        # Make predictions
        predictions = self.model.predict(X)

        # Ensure non-negative predictions
        predictions = np.maximum(0, predictions)

        return predictions

    def predict_for_zone(self, features, zone):
        """
        Make predictions for a specific zone.

        Parameters:
        -----------
        features : DataFrame
            Feature matrix
        zone : str
            The zone to predict for

        Returns:
        --------
        array
            Non-negative predictions for the zone
        """
        if zone not in self.zone_models:
            raise ValueError(f"Model for zone {zone} has not been trained yet")

        # Get zone-specific model and scaler
        zone_model = self.zone_models[zone]["model"]
        zone_scaler = self.zone_models[zone]["scaler"]

        # Scale features
        features_scaled = zone_scaler.transform(features)

        # Make predictions
        predictions = zone_model.predict(features_scaled)

        # Ensure non-negative predictions
        predictions = np.maximum(0, predictions)

        return predictions

    def evaluate(self, zone):
        """
        Evaluate the model for a specific zone.

        Parameters:
        -----------
        zone : str
            The zone to evaluate

        Returns:
        --------
        dict
            Evaluation metrics
        """
        if zone not in self.zone_models:
            raise ValueError(f"Model for zone {zone} has not been trained yet")

        return self.zone_models[zone]["metrics"]

    def plot_results(self, zone, filepath=None):
        """
        Plot actual vs predicted values for a specific zone.

        Parameters:
        -----------
        zone : str
            The zone to plot results for
        """
        if zone not in self.zone_models:
            raise ValueError(f"Model for zone {zone} has not been trained yet")

        # Drop zeros for plot
        df = self.zone_models[zone]["test_results"]
        df = df[df["y_true"] > 0]

        fig, axs = plt.subplots(3, 1, figsize=(8, 8))
        fig.suptitle(f"Actual vs Predicted Load for Zone {zone}")

        # Timeseries plot
        ax = axs[0]
        ax.plot(df["y_true"], label="Actual")
        ax.plot(df["y_pred"], label="Prediction")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Load (MW)")
        ax.grid(alpha=0.4)
        ax.legend()

        # Temperature scatter plot
        ax = axs[1]
        ax.scatter(df["temp"], df["y_true"], label="Actual", s=5, alpha=0.5)
        ax.scatter(df["temp"], df["y_pred"], label="Prediction", s=5, alpha=0.5)
        ax.set_xlabel("Zonal temperature (C)")
        ax.set_ylabel("Load (MW)")
        ax.grid(alpha=0.4)
        ax.legend()

        # Scatter plot
        ax = axs[2]
        ax.scatter(df["y_true"], df["y_pred"], s=5, alpha=0.5)
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", color="black")
        ax.set_xlabel("Actual load (MW)")
        ax.set_ylabel("Predicted load (MW)")
        ax.grid(alpha=0.4)

        plt.tight_layout()
        if filepath is not None:
            plt.savefig(filepath, bbox_inches="tight")
        else:
            plt.show()

    def feature_importance(self, zone):
        """
        Plot feature importance for a specific zone (if the model supports it).

        Parameters:
        -----------
        zone : str
            The zone to plot feature importance for
        """
        if zone not in self.zone_models:
            raise ValueError(f"Model for zone {zone} has not been trained yet")

        model = self.zone_models[zone]["model"]
        features = self.zone_models[zone]["features"]

        # Check if the model has feature_importances_ attribute
        if hasattr(model, "feature_importances_"):
            # Get feature importances
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]

            # Create a plot
            plt.figure(figsize=(8, 4))
            plt.title(f"Feature Importance for Zone {zone}")
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(
                range(len(importances)), [features[i] for i in indices], rotation=0
            )
            plt.tight_layout()
            plt.show()
        else:
            print(f"Model for zone {zone} does not support feature importance")

    def save_model(self, zone, filepath=None):
        """
        Save the model for a specific zone.

        Parameters:
        -----------
        zone : str
            The zone to save the model for
        filepath : str, optional
            The filepath to save the model to. Default is 'zone_{zone}_model.pkl'.
        """
        if zone not in self.zone_models:
            raise ValueError(f"Model for zone {zone} has not been trained yet")

        if filepath is None:
            filepath = f"zone_{zone}_model.pkl"

        with open(filepath, "wb") as f:
            pickle.dump(self.zone_models[zone], f)

        print(f"Model for zone {zone} saved to {filepath}")

    def load_model(self, zone, filepath=None):
        """
        Load the model for a specific zone.

        Parameters:
        -----------
        zone : str
            The zone to load the model for
        filepath : str, optional
            The filepath to load the model from. Default is 'zone_{zone}_model.pkl'.
        """
        if filepath is None:
            filepath = f"zone_{zone}_model.pkl"

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} not found")

        with open(filepath, "rb") as f:
            self.zone_models[zone] = pickle.load(f)

        print(f"Model for zone {zone} loaded from {filepath}")


def load_and_prepare_data(temp_file, load_file):
    """
    Load temperature and load data from files.

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
        # Load data
        temp_data = pd.read_csv(temp_file)
        load_data = pd.read_csv(load_file)

        # Ensure datetime columns are properly formatted
        temp_data["time"] = pd.to_datetime(temp_data["time"])
        load_data["time"] = pd.to_datetime(load_data["time"])

        return temp_data, load_data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
