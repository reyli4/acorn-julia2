import os
import requests
import zipfile
import pandas as pd
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
from python.utils import project_path

"""
This script downloads and processes historical load data from NYISO. The data is
downloaded in CSV format and processed to aggregate the data into hourly averages
and store the results in a CSV file. This script is self contained and does not 
require any web-based interactions. 

Assumptions:
- Load zones seem to change over time, from NYC_LongIsland to NYC and LongIsland. We assume
  that the NYC_LongIsland zone can be split into NYC and LongIsland, using a ratio of 2.5:1.
- We use "integrated" real-time actual load, which corresponds to integrating over the hour.

NOTE: Time index is in local time (EST/EDT) and not UTC.
"""

###########################
## Preliminaries
###########################
# Define the base URL and destination folder
base_url = "http://mis.nyiso.com/public/csv/palIntegrated/"
data_folder = f"{project_path}/data/nyiso/historical_load"

# Create the relevant folders
os.makedirs(data_folder, exist_ok=True)
os.makedirs(f"{data_folder}/zipped", exist_ok=True)
os.makedirs(f"{data_folder}/extracted", exist_ok=True)
os.makedirs(f"{data_folder}/combined", exist_ok=True)


###########################
## User-defined functions
###########################
# Function to download a file
def download_file(url, destination):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(destination, "wb") as f:
            f.write(response.content)
        return True
    return False


# Unzip a zipped file
def unzip_file(zip_path, extract_path):
    os.makedirs(extract_path, exist_ok=True)  # ensure the extraction directory exists
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)


# Function to process and aggregate CSV files
def process_load_file(file_path):
    # Define the mapping for zones -> zone numbers
    name_mapping = {
        "WEST": "A",
        "GENESE": "B",
        "CENTRL": "C",
        "NORTH": "D",
        "MHK VL": "E",
        "CAPITL": "F",
        "HUD VL": "G",
        "MILLWD": "H",
        "DUNWOD": "I",
        "N.Y.C.": "J",
        "LONGIL": "K",
        "N.Y.C._LONGIL": "NYC_Long_Island",  # Placeholder for handling separately
    }

    # Read the CSV file
    df = pd.read_csv(
        file_path,
        dtype={"Time Stamp": str, "Name": str, "PTID": str, "Integrated Load": float},
    )

    # Replace missing values in the Load column with 0.0 before processing
    df["Integrated Load"] = df["Integrated Load"].fillna(0.0)

    # Parse the "Time Stamp" column
    df["Time Stamp"] = pd.to_datetime(df["Time Stamp"], format="%m/%d/%Y %H:%M:%S")
    df["Hourly Time"] = df["Time Stamp"].dt.floor("h")

    # Create a new DataFrame for the transformed data
    transformed_rows = pd.DataFrame(columns=["time", "zone", "load_MW"])

    # Create separate dataframes for NYC_LONGIL and other zones
    nyc_longil_df = df[df["Name"] == "N.Y.C._LONGIL"].copy()
    other_zones_df = df[df["Name"] != "N.Y.C._LONGIL"].copy()

    # Process NYC_LONGIL rows
    nyc_rows = pd.DataFrame(
        {
            "time": nyc_longil_df["Hourly Time"],
            "zone": "J",
            "load_MW": nyc_longil_df["Integrated Load"] * (2.5 / 3.5),
        }
    )

    longisland_rows = pd.DataFrame(
        {
            "time": nyc_longil_df["Hourly Time"],
            "zone": "K",
            "load_MW": nyc_longil_df["Integrated Load"] * (1 / 3.5),
        }
    )

    # Process other zones
    other_zones_df["zone"] = other_zones_df["Name"].map(name_mapping)
    other_rows = pd.DataFrame(
        {
            "time": other_zones_df["Hourly Time"],
            "zone": other_zones_df["zone"],
            "load_MW": other_zones_df["Integrated Load"],
        }
    )

    # Combine all rows
    transformed_rows = pd.concat(
        [nyc_rows, longisland_rows, other_rows], ignore_index=True
    )

    # Calculate hourly averages and group by "Hourly Time" and "zone"
    df_hourly = (
        transformed_rows.groupby(["time", "zone"])["load_MW"].mean().reset_index()
    )

    return df_hourly


###########################
## Download and unzip data
###########################
def download_and_extract_data():
    # Loop over each year and month
    for year in range(2002, 2024):
        for month in range(1, 13):
            # Use the first day of the month as the date format for URL (e.g., YYYYMM01)
            date_str = date(year, month, 1).strftime("%Y%m%d")[:6] + "01"
            file_url = base_url + date_str + "palIntegrated_csv.zip"
            zip_file_path = f"{data_folder}/zipped/{date_str}.zip"
            extracted_folder_path = f"{data_folder}/extracted/"

            # Check if the zip file or extracted folder already exists
            if os.path.exists(zip_file_path):
                print(f"Data for {date_str} already exists. Skipping download.")
                continue

            try:
                # Download the zip file
                print(f"Downloading data for {date_str}...")
                print(file_url)
                if download_file(file_url, zip_file_path):
                    # Unzip the file
                    print(f"Unzipping data for {date_str}...")
                    unzip_file(zip_file_path, extracted_folder_path)
                else:
                    print(f"Failed to download data for {date_str}")
            except Exception as e:
                print(f"Failed to download or unzip data for {date_str}: {e}")
                # Optionally, remove the zip file if it failed
                if os.path.exists(zip_file_path):
                    os.remove(zip_file_path)


############################
## Combine the data files
############################
def combine_load_data():
    # Check if done
    combined_file_path = f"{data_folder}/combined/historical_load.csv"
    if os.path.exists(combined_file_path):
        print("Combined data already exists. Skipping.")
        return None

    # Get list of all files in extracted folder
    files = os.listdir(f"{data_folder}/extracted")

    # Loop through and process each file
    combined_df = pd.DataFrame(columns=["time", "zone", "load_MW"])
    for file in files:
        file_path = f"{data_folder}/extracted/{file}"
        try:
            processed_df = process_load_file(file_path)
            if len(processed_df) > 0:
                combined_df = pd.concat([combined_df, processed_df], ignore_index=True)
        except Exception as e:
            print(f"Error processing {file}: {e}")

    # Sort by time and Zone
    combined_df = combined_df.sort_values(["time", "zone"]).reset_index(drop=True)

    # Store the combined dataframe
    combined_df.to_csv(combined_file_path, index=False)

    return combined_df


############################
## Plotting functions
############################
def plot_historical_loads(year):
    # Load the combined CSV file
    df = pd.read_csv(f"{data_folder}/combined/historical_load.csv")
    df["time"] = pd.to_datetime(df["time"])

    # Group by each hour and calculate the total load across all zones
    df_agg = df.groupby("time")["load_MW"].sum().reset_index()
    df_agg["load_GW"] = df_agg["load_MW"] / 1000

    # Add new columns for day, month, and hour
    df_agg["Year"] = df_agg["time"].dt.year
    df_agg["Month"] = df_agg["time"].dt.month
    df_agg["Day"] = df_agg["time"].dt.day
    df_agg["Hour"] = df_agg["time"].dt.hour

    # Plot specified year
    df_year = df_agg[df_agg["Year"] == year].copy()

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(df_year["time"], df_year["load_GW"], linewidth=1)
    ax.set_ylabel("Total Load (GW)")
    ax.set_title(f"NYISO Load in {year}")
    ax.grid(alpha=0.5)
    plt.savefig(f"{data_folder}/combined/load_{year}.png")

    # Calculate percentiles for each day, month, and hour
    percentiles = (
        df_agg.groupby(["Day", "Month", "Hour"])
        .agg(
            Min_Load_GW=("load_GW", "min"),
            q01_Load_GW=("load_GW", lambda x: np.percentile(x, 1)),
            Median_Load_GW=("load_GW", "median"),
            q99_Load_GW=("load_GW", lambda x: np.percentile(x, 99)),
            Max_Load_GW=("load_GW", "max"),
        )
        .reset_index()
    )

    # Add hour of year column based on month, day, hour
    percentiles["HourOfYear"] = (
        (percentiles["Month"] - 1) * 30 * 24
        + (percentiles["Day"] - 1) * 24
        + percentiles["Hour"]
    )

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    percentiles.sort_values("HourOfYear").plot(
        x="HourOfYear",
        y=[
            "Min_Load_GW",
            "q01_Load_GW",
            "Median_Load_GW",
            "q99_Load_GW",
            "Max_Load_GW",
        ],
        title="Percentiles of Load Profile",
        xlabel="Hour of Year",
        ylabel="Load (GW)",
        ax=ax,
    )
    ax.grid(alpha=0.5)
    plt.savefig(f"{data_folder}/combined/load_percentiles.png")


def main():
    # Download and extract data
    download_and_extract_data()

    # Combine the data
    combine_load_data()

    # Plot the historical loads
    plot_historical_loads(2016)


if __name__ == "__main__":
    main()
