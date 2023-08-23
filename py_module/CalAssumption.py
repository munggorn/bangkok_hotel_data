import numpy as np
import pandas as pd
from ExtractInput import relevant_data

# Assumptions for energy consumption

np.random.seed(42)
BASE_ENERGY =0 # 400 + (200 * np.random.rand(relevant_data.shape[0]))
ENERGY_PER_ROOM = 20+ (5 * np.random.rand(relevant_data.shape[0]))
ENERGY_PER_AMENITY = 5+ (1 * np.random.rand(relevant_data.shape[0]))

# Compute the energy consumption

relevant_data['energy_consumption'] = (BASE_ENERGY + 
                                      relevant_data['numberOfRooms'] * ENERGY_PER_ROOM + 
                                      relevant_data['num_amenities'] * ENERGY_PER_AMENITY)

# Assumptions for usable area based on hotel class

area_per_room = {
    0.0: 20,
    1.0: 24,
    2.0: 28,
    3.0: 32,
    4.0: 36,
    5.0: 40
}
UNOCCUPIED =  relevant_data['numberOfRooms']*(0.1*np.random.rand(relevant_data.shape[0]))

# Compute the usable area
relevant_data['usable_area'] = (relevant_data['numberOfRooms'] -  
                                UNOCCUPIED) *  ((relevant_data['ceilhotelClass']).map(area_per_room)+
                                                5*np.random.rand(relevant_data.shape[0]))


# ---------------------------------------------------

# Calculate baseline energy consumption and usable area
average_energy_consumption = relevant_data['energy_consumption'].mean()
average_usable_area = relevant_data['usable_area'].mean()

# Calculate baseline energy consumption and usable area by hotel class
grouped_data = relevant_data.groupby('ceilhotelClass').agg({
    'energy_consumption': 'mean',
    'usable_area': 'mean'
})
grouped_data.head()