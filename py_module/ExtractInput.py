import numpy as np
import pandas as pd

# Load the data from the CSV file
hotel_data = pd.read_csv('bangkok_hotel_data.csv')

# Display the first few rows to inspect the data
hotel_data.head()


# ---------------------------------------------------

# Extract columns that start with 'amenities/'
amenities_cols = [col for col in hotel_data.columns if col.startswith('amenities/')]

# Calculate the number of amenities for each hotel
hotel_data['num_amenities'] = hotel_data[amenities_cols].notna().sum(axis=1)

# Extract the relevant columns
relevant_data = hotel_data[['name','latitude','longitude','subcategories/0',  'numberOfRooms', 'hotelClass', 'num_amenities']]

# Round up Hotel Class
relevant_data['ceilhotelClass'] = np.ceil(relevant_data['hotelClass'])

# Display the first few rows of the relevant data
relevant_data.head()