Bangkok Hotel Analysis
This repository contains an in-depth analysis of hotels in Bangkok, focusing on the relationship between a hotel's usable area and its energy consumption. The analysis is further segmented by hotel class.

# Table of Contents
1. Datasets
2. Jupyter Notebook Analysis
3. Interactive Dashboard
4. Getting Started
5. Contributing
6. License
7. Acknowledgements

# 1. Datasets

- bangkok_hotel_data.csv: Contains detailed information about hotels in Bangkok, including amenities, ratings, reviews, and more.
- relevant_data.csv: Dataset containing relevant columns extracted for the analysis.
- refined_data.csv: Preprocessed dataset derived from relevant_data.csv 

# 2. Jupyter Notebook Analysis

bangkok_hotel_data_analyze.ipynb: The primary analysis notebook that provides insights into the relationship between usable area and energy consumption of hotels. The analysis uses scatter plots, regression lines, and other visualization techniques to uncover trends.
Interactive Dashboard
The repository also contains code for an interactive Dash application, allowing users to explore the data in a more hands-on manner. Users can select specific hotels and see how they compare against established benchmarks.

# 3.Interactive Dashboard
App.py: The repository also contains code for an interactive Dash application, allowing users to explore the data in a more hands-on manner. Users can select specific hotels and see how they compare against established benchmarks.
https://bangkok-hotel-data-app.onrender.com/

# 4. Getting Started

1. Clone the repository:
   git clone https://github.com/munggorn/bangkok_hotel_data/tree/main

2. Install necessary libraries (preferably in a virtual environment):
   pip install -r requirements.txt

3. Navigate to the repository and launch the Python Module:
   - python ExtractInput.py
   - python CalAssumption.py
   - python Visualization.py
   - python Criterion.py
   - python App.py

4. To run the Dash app:
   python App.py

# 5. Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. Ensure you update tests as appropriate.

# 6. License
MIT

# 7. Acknowledgements
- Data sourced from Altotech
- Thanks to Pamekitti who provided valuable feedback.



