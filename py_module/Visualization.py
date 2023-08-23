import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from CalAssumption import relevant_data

# Clean the data
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

# ---------------------------------------------------

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

# ---------------------------------------------------

# Extract mean values of energy_consumption and usable_area
energy_consumption_mean_all = refined_data['energy_consumption'].mean()
usable_area_mean_all = refined_data['usable_area'].mean()

# ---------------------------------------------------

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


# ---------------------------------------------------

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


# ---------------------------------------------------



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


# ---------------------------------------------------

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


# ---------------------------------------------------

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


# ---------------------------------------------------

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


# ---------------------------------------------------

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


# ---------------------------------------------------

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


# ---------------------------------------------------

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