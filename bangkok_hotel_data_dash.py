#!/usr/bin/env python
# coding: utf-8

# # Import relevant libraries and inspect data

# In[1]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
mapbox_token = os.environ.get("MAPBOX_TOKEN")

# Load the data from the CSV file
hotel_data = pd.read_csv('bangkok_hotel_data.csv')

# Display the first few rows to inspect the data
hotel_data.head()


# # Extract the relevant columns

# In[2]:


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


# # Determine the Energy Consumption and the Usable Area

# In[3]:


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


# # Calculate Energy Consumption Baseline and Usable Area Baseline for each class

# In[4]:


# Calculate baseline energy consumption and usable area
average_energy_consumption = relevant_data['energy_consumption'].mean()
average_usable_area = relevant_data['usable_area'].mean()

# Calculate baseline energy consumption and usable area by hotel class
grouped_data = relevant_data.groupby('ceilhotelClass').agg({
    'energy_consumption': 'mean',
    'usable_area': 'mean'
})
grouped_data.head()


# # Plot between Energy Consumption and Usable Area

# In[5]:


relevant_data = relevant_data.dropna()

x=relevant_data['usable_area'].values.reshape(-1,1)
y=relevant_data['energy_consumption'].values.reshape(-1,1)

reg = LinearRegression().fit(x, y)

# Get the predicted values for the linear regression line
y_pred = reg.predict(x)

# Create scatter plot using plotly
fig1 = go.Scatter(x=relevant_data['usable_area'], 
                  y=relevant_data['energy_consumption'], 
                  mode='markers',
                  marker=dict(size=4, colorscale='Viridis'),
                  text=relevant_data['name'],
                  name='Hotels'
                 )
fig2 = go.Scatter(x=relevant_data['usable_area'], 
                  y=y_pred.flatten(), 
                  mode='lines', 
                  name='Benchmark', 
                  line=dict(color='grey'
                           )
                 )


# Combine the traces and create the layout for the plot
layout = go.Layout(
    title='Hotel Energy Consumption vs. Usable Area (By Hotel Class)',
    xaxis=dict(title='Usable Area'),
    yaxis=dict(title='Energy Consumption'),
    legend_title_text='Legend'
)

fig = go.Figure(data=[fig1, fig2], layout=layout)

# Display the plot
fig.show()


# # Remove Outliers

# In[6]:


# Function to remove outliers
def remove_outliers(df, column_name):
    Q3 = df[column_name].quantile(0.75)
    Q1 = df[column_name].quantile(0.25)
    IQR = Q3-Q1
    upper_bound = Q3 + 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR
    
    # Filter the data to remove outliers
    return df[(df[column_name] <= upper_bound) & (df[column_name] >= lower_bound)]

# Remove outliers for energy consumption and usable area
refined_data = remove_outliers(relevant_data, 'energy_consumption')
refined_data = remove_outliers(refined_data, 'usable_area')
refined_data['energy_consumption'] = (refined_data['energy_consumption']/10).map(int)
refined_data['usable_area'] = refined_data['usable_area'].map(int)
refined_data.head()


# In[7]:


# Extract mean values of energy_consumption and usable_area
energy_consumption_mean_all = refined_data['energy_consumption'].mean()
usable_area_mean_all = refined_data['usable_area'].mean()


# In[8]:


x_r=refined_data['usable_area'].values.reshape(-1,1)
y_r=refined_data['energy_consumption'].values.reshape(-1,1)

reg = LinearRegression().fit(x_r, y_r)

# Get the predicted values for the linear regression line
y_r_pred = reg.predict(x_r)

# Create the rectangular hyperbola curve
x_hyperbola = np.linspace(300, 4800, 400)
y_hyperbola = usable_area_mean_all * energy_consumption_mean_all / x_hyperbola

# Create scatter plot using plotly
fig3 = go.Scatter(x=refined_data['usable_area'], 
                  y=refined_data['energy_consumption'], 
                  mode='markers',
                  marker=dict(size=4, colorscale='Viridis'),
                  text=relevant_data['name'],
                  name='Hotels'
                 )
fig4 = go.Scatter(x=refined_data['usable_area'], 
                  y=y_r_pred.flatten(), 
                  mode='lines', 
                  name='Benchmark', 
                  line=dict(color='grey'
                           )
                 )

fig_hyperbola = go.Scatter(x=x_hyperbola, 
                           y=y_hyperbola, 
                           mode='lines', 
                           name='Carbon Emission', 
                           line=dict(color='grey', dash='dot'
                                    )
                          )



# Combine the traces and create the layout for the plot
layout = go.Layout(
    title='Hotel Energy Consumption vs. Usable Area',
    xaxis=dict(title='Usable Area'),
    yaxis=dict(title='Energy Consumption'),
    legend_title_text='Legend'
)


fig = go.Figure(data=[fig3, fig4, fig_hyperbola], layout=layout)

# Display the plot
fig.show()


# # Group by Hotel Size

# In[9]:


from sklearn.cluster import KMeans

# Data for clustering
X =np.sort( refined_data[['usable_area', 'energy_consumption']])

# Apply KMeans clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
refined_data['cluster'] = kmeans.fit_predict(X)

# Fit the linear regression model to get the regression line
reg = LinearRegression().fit(
    refined_data['usable_area'].values.reshape(-1, 1), 
    refined_data['energy_consumption'].values.reshape(-1, 1)
)
y_pred = reg.predict(refined_data['usable_area'].values.reshape(-1, 1))

# Create a scatter plot for each cluster
traces = []
colors = ['red', 'blue', 'green']
labels = ['Small Size Hotels', 'Medium Size Hotels', 'Large Size Hotels']
order = [0, 2, 1]  # Order for 'Small Size Hotels', 'Medium Size Hotels', 'Large Size Hotels'

for cluster, color, label in zip(order, colors, labels):
    subset = refined_data[refined_data['cluster'] == cluster]
    trace = go.Scatter(
        x=subset['usable_area'],
        y=subset['energy_consumption'],
        mode='markers',
        text=relevant_data['name'],
        marker=dict(color=color, size=4, 
                    line=dict(color='white', 
                              width=0.5)
                   ),
        name=label
    )
    traces.append(trace)

# Create a scatter plot for the regression line
line_trace = go.Scatter(
    x=refined_data['usable_area'],
    y=y_pred.flatten(),
    mode='lines',
    line=dict(color='gray'),
    name='Baseline'
)
traces.append(line_trace)

# Create the rectangular hyperbola curve
x_hyperbola = np.linspace(300, 4800, 400)
y_hyperbola = usable_area_mean_all * energy_consumption_mean_all / x_hyperbola

fig_hyperbola = go.Scatter(x=x_hyperbola, 
                           y=y_hyperbola, 
                           mode='lines', 
                           name='Carbon Emission', 
                           line=dict(color='grey' ,
                                     dash='dot'
                                    )
                          )

traces.append(fig_hyperbola)

# Create the layout for the plot
layout = go.Layout(
    title='Hotel Energy Consumption vs. Usable Area (KMeans Clustering)',
    xaxis=dict(title='Usable Area (sq.m.)'),
    yaxis=dict(title='Energy Consumption (MWh/month)'),
    legend_title_text='Hotel Size'
)

# Create the figure and add the traces
fig = go.Figure(data=traces, layout=layout)

# Display the plot
fig.show()


# # Group by Hotel Class

# In[10]:




# Extract unique hotel classes
unique_classes = np.sort(refined_data['ceilhotelClass'].unique())
# Fit the linear regression model to get the regression line
reg = LinearRegression().fit(
    refined_data['usable_area'].values.reshape(-1, 1),
    refined_data['energy_consumption'].values.reshape(-1, 1)
)
y_pred = reg.predict(refined_data['usable_area'].values.reshape(-1, 1))

# Create a scatter plot for each hotel class
traces = []
colors = ['red', 'blue', 'green', 'orange', 'purple', 'black']
for hotel_class, color in zip(unique_classes, colors):
    subset = refined_data[refined_data['ceilhotelClass'] == hotel_class]
    trace = go.Scatter(
        x=subset['usable_area'],
        y=subset['energy_consumption'],
        mode='markers',
        text=relevant_data['name'],
        marker=dict(color=color,
                    size=4,
                    line=dict(color='white',
                              width=0.5)),
        name=f'{int(hotel_class)}-Star'
    )
    traces.append(trace)

# Create a scatter plot for the regression line
line_trace = go.Scatter(
    x=refined_data['usable_area'],
    y=y_pred.flatten(),
    mode='lines',
    line=dict(color='gray'),
    name='Baseline'
)
traces.append(line_trace)

# Create the rectangular hyperbola curve
x_hyperbola = np.linspace(300, 4800, 400)
y_hyperbola = usable_area_mean_all * energy_consumption_mean_all / x_hyperbola

fig_hyperbola = go.Scatter(x=x_hyperbola, 
                           y=y_hyperbola, 
                           mode='lines', 
                           name='Carbon Emission', 
                           line=dict(color='grey',
                                     dash='dot'
                                    )
                          )

traces.append(fig_hyperbola)

# Create the layout for the plot
layout = go.Layout(
    title='Hotel Energy Consumption vs. Usable Area (By Hotel Class)',
    xaxis=dict(title='Usable Area (sq.m.)'),
    yaxis=dict(title='Energy Consumption (MWh/month)'),
    legend_title_text='Hotel Class'
)

# Create the figure and add the traces
fig = go.Figure(data=traces,
                layout=layout)

# Display the plot
fig.show()


# # Compare Each Class with its Baseline

# In[11]:


# Fit the linear regression model to get the regression line for the subset
subset_0 = refined_data[refined_data['ceilhotelClass'] == 0.0]
reg = LinearRegression().fit(
    subset_0['usable_area'].values.reshape(-1, 1), 
    subset_0['energy_consumption'].values.reshape(-1, 1)
)
y_pred = reg.predict(subset_0['usable_area'].values.reshape(-1, 1))

# Extract mean values of energy_consumption and usable_area
energy_consumption_mean = subset_0['energy_consumption'].mean()
usable_area_mean = subset_0['usable_area'].mean()


# Create a scatter plot for the subset
scatter_trace = go.Scatter(
    x=subset_0['usable_area'],
    y=subset_0['energy_consumption'],
    mode='markers',
    text=relevant_data['name'],
    marker=dict(color='red', 
                size=4),
    name='Hotels (0.0-Star)'
)

# Create a scatter plot for the regression line
line_trace = go.Scatter(
    x=subset_0['usable_area'],
    y=y_pred.flatten(),
    mode='lines',
    line=dict(color='grey'),
    name='Baseline'
)

# Create the rectangular hyperbola curve
x_hyperbola = np.linspace(120, 4400, 400)
y_hyperbola = usable_area_mean * energy_consumption_mean / x_hyperbola

fig_hyperbola = go.Scatter(x=x_hyperbola, 
                           y=y_hyperbola, 
                           mode='lines', 
                           name='Carbon Emission', 
                           line=dict(color='grey', 
                                     dash='dot'
                                    )
                          )



# Create the layout for the plot
layout = go.Layout(
    title='Hotel Energy Consumption vs. Usable Area (By Hotel Class)',
    xaxis=dict(title='Usable Area (sq.m.)'),
    yaxis=dict(title='Energy Consumption (MWh/month)'),
    legend_title_text='Legend')

# Create the figure and add the traces
fig = go.Figure(
    data=[scatter_trace,
          line_trace,
          fig_hyperbola],
    layout=layout
)

# Display the plot
fig.show()


# In[12]:


# Fit the linear regression model to get the regression line for the subset_1
subset_1 = refined_data[refined_data['ceilhotelClass'] == 1.0]
reg = LinearRegression().fit(
    subset_1['usable_area'].values.reshape(-1, 1),
    subset_1['energy_consumption'].values.reshape(-1, 1)
)
y_pred = reg.predict(subset_1['usable_area'].values.reshape(-1, 1))

# Extract mean values of energy_consumption and usable_area
energy_consumption_mean = subset_1['energy_consumption'].mean()
usable_area_mean = subset_1['usable_area'].mean()

# Create a scatter plot for the subset_1
scatter_trace = go.Scatter(
    x=subset_1['usable_area'],
    y=subset_1['energy_consumption'],
    mode='markers',
    text=relevant_data['name'],
    marker=dict(color='blue',
                size=4),
    name='Hotels (1.0-Star)'
)

# Create a scatter plot for the regression line
line_trace = go.Scatter(
    x=subset_1['usable_area'],
    y=y_pred.flatten(),
    mode='lines',
    line=dict(color='grey'),
    name='Baseline'
)

# Create the rectangular hyperbola curve
x_hyperbola = np.linspace(200, 2500, 400)
y_hyperbola = usable_area_mean * energy_consumption_mean / x_hyperbola

fig_hyperbola = go.Scatter(x=x_hyperbola, 
                           y=y_hyperbola, 
                           mode='lines', 
                           name='Carbon Emission', 
                           line=dict(color='grey',
                                     dash='dot')
)

# Create the layout for the plot
layout = go.Layout(
    title='Hotel Energy Consumption vs. Usable Area (By Hotel Class)',
    xaxis=dict(title='Usable Area (sq.m.)'),
    yaxis=dict(title='Energy Consumption (MWh/month)'),
    legend_title_text='Legend'
)

# Create the figure and add the traces
fig = go.Figure(data=[scatter_trace,
                      line_trace,
                      fig_hyperbola], 
                layout=layout)

# Display the plot
fig.show()


# In[13]:


subset_2 = refined_data[refined_data['ceilhotelClass'] == 2.0]
# Fit the linear regression model to get the regression line for the subset_2
reg = LinearRegression().fit(
    subset_2['usable_area'].values.reshape(-1, 1),
    subset_2['energy_consumption'].values.reshape(-1, 1)
)
y_pred = reg.predict(subset_2['usable_area'].values.reshape(-1, 1))

# Extract mean values of energy_consumption and usable_area
energy_consumption_mean = subset_2['energy_consumption'].mean()
usable_area_mean = subset_2['usable_area'].mean()

# Create a scatter plot for the subset_2
scatter_trace = go.Scatter(
    x=subset_2['usable_area'],
    y=subset_2['energy_consumption'],
    mode='markers',
    text=relevant_data['name'],
    marker=dict(color='green', 
                size=4),
    name='Hotels (2.0-Star)'
)

# Create a scatter plot for the regression line
line_trace = go.Scatter(
    x=subset_2['usable_area'],
    y=y_pred.flatten(),
    mode='lines',
    text=relevant_data['name'],
    line=dict(color='grey'),
    name='Baseline'
)

# Create the rectangular hyperbola curve
x_hyperbola = np.linspace(150, 4500, 400)
y_hyperbola = usable_area_mean * energy_consumption_mean / x_hyperbola

fig_hyperbola = go.Scatter(x=x_hyperbola, 
                           y=y_hyperbola, 
                           mode='lines', 
                           name='Carbon Emission', 
                           line=dict(color='grey', 
                                     dash='dot')
)


# Create the layout for the plot
layout = go.Layout(
    title='Hotel Energy Consumption vs. Usable Area (By Hotel Class)',
    xaxis=dict(title='Usable Area (sq.m.)'),
    yaxis=dict(title='Energy Consumption (MWh/month)'),
    legend_title_text='Legend'
)

# Create the figure and add the traces
fig = go.Figure(data=[scatter_trace, line_trace,
                      fig_hyperbola], 
                layout=layout)

# Display the plot
fig.show()


# In[14]:


subset_3 = refined_data[refined_data['ceilhotelClass'] == 3.0]
# Fit the linear regression model to get the regression line for the subset_3
reg = LinearRegression().fit(
    subset_3['usable_area'].values.reshape(-1, 1), 
    subset_3['energy_consumption'].values.reshape(-1, 1)
)
y_pred = reg.predict(subset_3['usable_area'].values.reshape(-1, 1))

# Extract mean values of energy_consumption and usable_area
energy_consumption_mean = subset_3['energy_consumption'].mean()
usable_area_mean = subset_3['usable_area'].mean()

# Create a scatter plot for the subset_3
scatter_trace = go.Scatter(
    x=subset_3['usable_area'],
    y=subset_3['energy_consumption'],
    mode='markers',
    text=relevant_data['name'],
    marker=dict(color='orange',
                size=4),
    name='Hotels (3.0-Star)'
)

# Create a scatter plot for the regression line
line_trace = go.Scatter(
    x=subset_3['usable_area'],
    y=y_pred.flatten(),
    mode='lines',
    line=dict(color='grey'),
    name='Baseline'
)

# Create the rectangular hyperbola curve
x_hyperbola = np.linspace(500, 4600, 400)
y_hyperbola = usable_area_mean * energy_consumption_mean / x_hyperbola

fig_hyperbola = go.Scatter(x=x_hyperbola, 
                           y=y_hyperbola, 
                           mode='lines', 
                           name='Carbon Emission', 
                           line=dict(color='grey', 
                                     dash='dot')
)

# Create the layout for the plot
layout = go.Layout(
    title='Hotel Energy Consumption vs. Usable Area (By Hotel Class)',
    xaxis=dict(title='Usable Area (sq.m.)'),
    yaxis=dict(title='Energy Consumption (MWh/month)'),
    legend_title_text='Legend'
)

# Create the figure and add the traces
fig = go.Figure(data=[scatter_trace,
                      line_trace,
                      fig_hyperbola],
                layout=layout)

# Display the plot
fig.show()


# In[15]:


subset_4 = refined_data[refined_data['ceilhotelClass'] == 4.0]
# Fit the linear regression model to get the regression line for the subset_4
reg = LinearRegression().fit(
    subset_4['usable_area'].values.reshape(-1, 1),
    subset_4['energy_consumption'].values.reshape(-1, 1)
)
y_pred = reg.predict(subset_4['usable_area'].values.reshape(-1, 1))

# Extract mean values of energy_consumption and usable_area
energy_consumption_mean = subset_4['energy_consumption'].mean()
usable_area_mean = subset_4['usable_area'].mean()

# Create a scatter plot for the subset_4
scatter_trace = go.Scatter(
    x=subset_4['usable_area'],
    y=subset_4['energy_consumption'],
    mode='markers',
    text=relevant_data['name'],
    marker=dict(color='purple',
                size=4),
    name='Hotels (4.0-Star)'
)

# Create a scatter plot for the regression line
line_trace = go.Scatter(
    x=subset_4['usable_area'],
    y=y_pred.flatten(),
    mode='lines',
    line=dict(color='grey'),
    name='Baseline'
)

# Create the rectangular hyperbola curve
x_hyperbola = np.linspace(1100, 4800, 400)
y_hyperbola = usable_area_mean * energy_consumption_mean / x_hyperbola

fig_hyperbola = go.Scatter(x=x_hyperbola, 
                           y=y_hyperbola, 
                           mode='lines', 
                           name='Carbon Emission', 
                           line=dict(color='grey',
                                     dash='dot')
)

# Create the layout for the plot
layout = go.Layout(
    title='Hotel Energy Consumption vs. Usable Area (By Hotel Class)',
    xaxis=dict(title='Usable Area (sq.m.)'),
    yaxis=dict(title='Energy Consumption (MWh/month)'),
    legend_title_text='Legend'
)

# Create the figure and add the traces
fig = go.Figure(data=[scatter_trace, 
                      line_trace,
                      fig_hyperbola],
                layout=layout)

# Display the plot
fig.show()


# In[16]:


subset_5 = refined_data[refined_data['ceilhotelClass'] == 5.0]
# Fit the linear regression model to get the regression line for the subset_5
reg = LinearRegression().fit(
    subset_5['usable_area'].values.reshape(-1, 1), 
    subset_5['energy_consumption'].values.reshape(-1, 1)
)
y_pred = reg.predict(subset_5['usable_area'].values.reshape(-1, 1))

# Extract mean values of energy_consumption and usable_area
energy_consumption_mean = subset_5['energy_consumption'].mean()
usable_area_mean = subset_5['usable_area'].mean()

# Create a scatter plot for the subset_5
scatter_trace = go.Scatter(
    x=subset_5['usable_area'],
    y=subset_5['energy_consumption'],
    mode='markers',
    text=relevant_data['name'],
    marker=dict(color='black', 
                size=4),
    name='Hotels (5.0-Star)'
)

# Create a scatter plot for the regression line
line_trace = go.Scatter(
    x=subset_5['usable_area'],
    y=y_pred.flatten(),
    mode='lines',
    line=dict(color='grey'),
    name='Baseline'
)

# Create the rectangular hyperbola curve
x_hyperbola = np.linspace(1500, 4800, 400)
y_hyperbola = usable_area_mean * energy_consumption_mean / x_hyperbola

fig_hyperbola = go.Scatter(x=x_hyperbola, 
                           y=y_hyperbola, 
                           mode='lines', 
                           name='Carbon Emission', 
                           line=dict(color='grey', 
                                     dash='dot')
)

# Create the layout for the plot
layout = go.Layout(
    title='Hotel Energy Consumption vs. Usable Area (By Hotel Class)',
    xaxis=dict(title='Usable Area (sq.m.)'),
    yaxis=dict(title='Energy Consumption (MWh/month)'),
    legend_title_text='Legend'
)

# Create the figure and add the traces
fig = go.Figure(data=[scatter_trace,
                      line_trace,
                      fig_hyperbola], 
                layout=layout)

# Display the plot
fig.show()


# # Extract necessary column for visualization on DASH

# In[17]:


# Extract the necessary columns
hotel_energy_area_data = refined_data[['name', 'energy_consumption', 'usable_area','latitude','longitude']]

# Display the first few rows of the hotel_energy_area_data
hotel_energy_area_data.head()


# In[18]:


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


# In[19]:


# Calculate baselines for each metric based on hotel class
baseline_data = refined_data.groupby('hotelClass').mean()[['energy_consumption', 'usable_area', 'area_to_energy_ratio', 'carbon_emission']]
baseline_data.reset_index(inplace=True)
baseline_data


# # Create Interactive platform with DASH

# In[21]:


import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots



# Calculate baselines for each metric based on hotel class
baseline_data = hotel_energy_area_data.groupby('hotelClass').mean()[['energy_consumption', 'usable_area', 'area_to_energy_ratio', 'carbon_emission']]
baseline_data.reset_index(inplace=True)

# Initialize the Dash app with bootstrap components
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MATERIA])  # Using CYBORG theme for a different aesthetic
server = app.server
# Define the app layout
app.layout = dbc.Container([
    # Navbar
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Hotel Energy and Area Details", href="#"))
        ],
        brand="Interactive Hotel Carbon Benchmarking Platform",
        brand_href="#",
        color="dark",
        dark=True,
        className="mb-4"
    ),

    # Search Bar
    dbc.Row([
        dbc.Col([
            dbc.Label("Select a Hotel:", className="mb-2", style={"font-weight": "bold"}),
            dcc.Dropdown(
                id='hotel-name-input',
                options=[{'label': name, 'value': name} for name in hotel_energy_area_data['name'].unique()],
                value='',
                searchable=True,
                clearable=True,
                placeholder="Search for a hotel...",
                style={"border-radius": "15px"}
            )
        ], width=8, className="offset-2 mb-4")
    ]),

    # Hotel Details
    dbc.Row(id='output-container', className="mb-4"),

    # Map and Bar Graphs
    dbc.Row([
        dbc.Col(dcc.Loading(
            id="loading",
            type="cube",
            children=dcc.Graph(id='hotel-map')
        ), width=6, className="mb-4"),
        
        # Bar Graphs Subplot
        dbc.Col(dcc.Graph(id='bar-subplot'), width=6, className="mb-4")
    ])
], fluid=True)

@app.callback(
    [Output('output-container', 'children'),
     Output('hotel-map', 'figure'),
     Output('bar-subplot', 'figure')],
    [Input('hotel-name-input', 'value')]
)
def update_output(hotel_name):
    matching_hotels = hotel_energy_area_data[hotel_energy_area_data['name'].str.contains(hotel_name, case=False, na=False)]
    
    if matching_hotels.empty:
        return dbc.Alert("No matching hotel found.", color="warning"), {}, {}
    else:
        hotel = matching_hotels.iloc[0]

        # Hotel Details
        details_content = [
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Energy Consumption", className="card-title"),
                        html.P(f"{hotel['energy_consumption']} MWh/month", className="card-text")
                    ])
                ], color="info", inverse=True),
                width=3
            ),

            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Usable Area", className="card-title"),
                        html.P(f"{hotel['usable_area']} sq.m.", className="card-text")
                    ])
                ], color="success", inverse=True),
                width=3
            ),

            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Efficiency", className="card-title"),
                        html.P(f"{hotel['area_to_energy_ratio']} sq.m./MWh/month", className="card-text")
                    ])
                ], color="warning", inverse=True),
                width=3
            ),

            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Carbon Emission", className="card-title"),
                        html.P(f"{hotel['carbon_emission']} kg CO₂/month", className="card-text")
                    ])
                ], color="danger", inverse=True),
                width=3
            )
        ]

        details_row = dbc.Row(details_content, className="mb-4")
        
        fig = px.scatter_mapbox(matching_hotels, lat='latitude', lon='longitude', hover_name='name',
                                color='color', color_discrete_map={'High Efficiency': 'green', 'Low Efficiency': 'red'},
                                hover_data=['hotelClass','energy_consumption', 'usable_area', 'efficiency', 'carbon_emission'], 
                                zoom=10, center={"lat": hotel['latitude'], "lon": hotel['longitude']})
        fig.update_layout(mapbox_style="carto-positron", mapbox_accesstoken=mapbox_token)
        
        # Bar Graphs Subplot
        hotel_class = hotel['hotelClass']
        baseline = baseline_data[baseline_data['hotelClass'] == hotel_class].iloc[0]
        
        metrics = ['energy_consumption', 'usable_area', 'area_to_energy_ratio', 'carbon_emission']
        subplot_titles = ['Energy Consumption (MWh/month)', 'Usable Area (sq.m.)', 'Efficiency (sq.m./MWh/month)', 'Carbon Emission (kg CO₂/month)']
        
        subplot_fig = make_subplots(rows=2, cols=2, subplot_titles=subplot_titles)
        
        row_col = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for metric, title, (row, col) in zip(metrics, subplot_titles, row_col):
            subplot_fig.add_trace(go.Bar(x=["Hotel", "Baseline"], y=[hotel[metric], baseline[metric]], name=title, text=[f"{hotel[metric]:.2f}", f"{baseline[metric]:.2f}"], textposition='auto'), row=row, col=col)
        
        subplot_fig.update_layout(
    showlegend=False, 
    barmode='group', 
    title_text="Hotel Metrics vs Baseline",
    title_x=0.5  # This centers the title
)
    
    return details_row, fig, subplot_fig

# Run the app
if __name__ == '__main__':
    app.run_server(port=8051)


# In[ ]:


pip freeze > requirements.txt

