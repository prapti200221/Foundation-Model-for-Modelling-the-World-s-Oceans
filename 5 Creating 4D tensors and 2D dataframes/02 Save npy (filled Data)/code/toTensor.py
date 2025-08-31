import pandas as pd  # type: ignore
import sklearn as sk # type: ignore
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras import layers, models, Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten # type: ignore
import math
import os
from datetime import datetime
import time
from tqdm import tqdm  # For progress bar

all_data_df = pd.read_parquet('/Users/imarcolic/Desktop/1 ACADEMIA/2.0 MSc LSE/1 MSc Data Science/ST498 Capstone/1 Data/04 2 years (filled Data)/spatially_filled_data_final.parquet')
print(all_data_df.columns)

######################################## Askig for requested variable and its subvariables ########################################
# ask for input of 4 = (reflectance, optics, plankton, transparency)
user_input = input("Please enter the requested variable: ")

variables_dict = {
    "Reflectance": ["RRS443", "RRS490"],
    "Transparency": ["KD490", "ZSD"],
    "Optics": ["BBP", "CDM"],
    "Plankton": ["CHL", "MICRO"]
}

if user_input in variables_dict:
    x = variables_dict[user_input]
    print(f"Subvariables for {user_input}: {x}")
else:
    print(f"Variable '{user_input}' not found. Available variables are: {', '.join(variables_dict.keys())}")

optics_df = all_data_df.loc[:, ["time", "latitude", "longitude", x[0], x[1], "flags"]]
print(f"Created dataframe with columns: {list(optics_df.columns)}")

#optics_df = all_data_df.iloc[:, [0, 1, 2, 5, 6, 8]]

######################################## Predefining tensor settings ########################################

number_of_unique_timestamps = optics_df["time"].nunique()
number_of_unique_latitudes = optics_df['latitude'].nunique()
number_of_unique_longitudes = optics_df['longitude'].nunique()

print(f"There are {number_of_unique_timestamps} unique timestamp values")
print(f"There are {number_of_unique_latitudes} unique latitude values")
print(f"There are {number_of_unique_longitudes} unique longitude values")

timestamps = optics_df['time'].unique().tolist() # 731
lats = optics_df['latitude'].unique().tolist() # 298 
longs = optics_df['longitude'].unique().tolist() # 827
print(f"Dimensions are ({timestamps}, {lats}, {longs}, 2)")

######################################## Creating new tensor ########################################

# Extract unique values from the current dataframe 
timestamps = sorted(optics_df['time'].unique().tolist())  # 592 unique timestamps
latitudes = sorted(optics_df['latitude'].unique().tolist())  # 298 unique latitudes
longitudes = sorted(optics_df['longitude'].unique().tolist())  # 827 unique longitudes

print(f"Creating a tensor with dimensions:")
print(f"- {len(timestamps)} timestamps")
print(f"- {len(latitudes)} latitudes")
print(f"- {len(longitudes)} longitudes")
print(f"- 2 channels ({x[0]} and {x[1]})")
print(f"Total tensor elements: {len(timestamps) * len(latitudes) * len(longitudes) * 2:,}")

# Create mappings for faster indexing
time_idx = {t: i for i, t in enumerate(timestamps)}
lat_idx = {lat: i for i, lat in enumerate(latitudes)}
lon_idx = {lon: i for i, lon in enumerate(longitudes)}

# Initialize the tensor with -20 values
print("Initializing tensor with -20 values...")
start_time = time.time()
optics_tensor = np.full((len(timestamps), len(latitudes), len(longitudes), 2), -20.0)
print(f"Initialized tensor in {time.time() - start_time:.2f} seconds")

# Fill the tensor with actual values from the original dataframe
print("Filling tensor with actual values...")
start_time = time.time()
filled_count = 0

# Process in batches to show progress
batch_size = 1000000  # Process 1 million rows at a time
total_batches = (len(optics_df) + batch_size - 1) // batch_size

for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, len(optics_df))
    batch = optics_df.iloc[start_idx:end_idx]
    
    for _, row in batch.iterrows():
        t_idx = time_idx[row['time']]
        lat_idx_val = lat_idx[row['latitude']]
        lon_idx_val = lon_idx[row['longitude']]
        
        # Set BBP value (channel 0)
        optics_tensor[t_idx, lat_idx_val, lon_idx_val, 0] = row[x[0]]
        
        # Set CDM value (channel 1)
        optics_tensor[t_idx, lat_idx_val, lon_idx_val, 1] = row[x[1]]
        
        filled_count += 1

print(f"Filled tensor in {time.time() - start_time:.2f} seconds")
print(f"Tensor shape: {optics_tensor.shape}")
print(f"Filled tensor entries: {filled_count:,} ({filled_count/(optics_tensor.size/2)*100:.2f}%)")

######################################## Check tensor for missing values ########################################
missing_count = np.sum(optics_tensor == -20.0) / 2  # Divide by 2 because we count per location, not per channel
total_locations = len(timestamps) * len(latitudes) * len(longitudes)
print(f"Missing values: {missing_count:,} ({missing_count/total_locations*100:.2f}%)")

# Inspect a small slice of the tensor to verify
print("\nInspecting a small slice of the tensor:")
slice_t = 0  # First timestamp
slice_lat = 0  # First latitude
slice_lon_range = slice(0, 5)  # First 5 longitudes
print(f"Values at timestamp {timestamps[slice_t]}, latitude {latitudes[slice_lat]}, first 5 longitudes:")
print(optics_tensor[slice_t, slice_lat, slice_lon_range, :])

######################################## Save the tensor ########################################

# Save the tensor
np.save(f'complete_{user_input}_tensor.npy', optics_tensor)