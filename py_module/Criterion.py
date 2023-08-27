import numpy as np
import pandas as pd
import plotly.graph_objects as go
from Visualization import refined_data, subset_0, subset_1, subset_2, subset_3, subset_4, subset_5 



# Extract the necessary columns
hotel_energy_area_data = refined_data[['name', 'energy_consumption', 'usable_area','latitude','longitude']]

# Display the first few rows of the hotel_energy_area_data
hotel_energy_area_data.head()

# ---------------------------------------------------

# Step 2: Compute area_to_energy_ratio and carbon_emission for each subset
subsets = [subset_0, subset_1, subset_2, subset_3, subset_4, subset_5]
for subset in subsets:
    subset['area_to_energy_ratio'] = subset['usable_area'] / subset['energy_consumption']
    subset['carbon_emission'] = subset['energy_consumption'] * subset['usable_area']

# Step 3: Define the baseline values for area_to_energy_ratio and carbon_emission for each subset
area_to_energy_ratio_baselines = [
    subset['usable_area'].mean() / subset['energy_consumption'].mean() for subset in subsets
]
carbon_emission_baselines = [
    subset['energy_consumption'].mean() * subset['usable_area'].mean() for subset in subsets
]

# Step 4: Define the lambda functions (conditions) for each subset

# Efficiency conditions
E_conditions = [
    lambda df: df['area_to_energy_ratio'] > baseline for baseline in area_to_energy_ratio_baselines
]

# Carbon emission conditions
CE_conditions = [
    lambda df: df['carbon_emission'] < baseline for baseline in carbon_emission_baselines
]

# Step 5: Apply conditions to add a new column in the refined_data dataframe
# We'll initialize the new column with "Not Set" and then update it based on conditions

refined_data['color'] = "Low Efficiency"

for i, (E_condition, CE_condition) in enumerate(zip(E_conditions, CE_conditions)):
    condition = E_condition(subsets[i]) | CE_condition(subsets[i])
    indices = subsets[i][condition].index
    refined_data.loc[indices, 'color'] = "High Efficiency"  # Here "Set" is just a placeholder, you can replace with desired value

    
refined_data['area_to_energy_ratio'] = (refined_data['usable_area']/refined_data['energy_consumption']).round(2)
refined_data['carbon_emission'] = ((refined_data['energy_consumption'] * refined_data['usable_area'])/1000).round(2)
refined_data['efficiency'] = refined_data['area_to_energy_ratio']

# Explicitly load the refined data
hotel_energy_area_data = refined_data
refined_data.head()

# ---------------------------------------------------

# Calculate baselines for each metric based on hotel class
baseline_data = refined_data.groupby('hotelClass').mean()[['energy_consumption', 'usable_area', 'area_to_energy_ratio', 'carbon_emission']]
baseline_data.reset_index(inplace=True)


# download csv
file_path = 'C:/Users/User/Documents/GitHub/bangkok_hotel_data/refined_data.csv'
refined_data.to_csv(file_path, index=False)
