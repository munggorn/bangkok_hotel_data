import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#from Criterion import refined_data
import os
mapbox_token = os.environ.get("MAPBOX_TOKEN")


# Explicitly load the refined data
hotel_energy_area_data = pd.read_csv('refined_data.csv')
hotel_energy_area_data['hotelRanking'] = hotel_energy_area_data['rankingString']

# Calculate baselines for each metric based on hotel class
baseline_data = hotel_energy_area_data.groupby('hotelClass').mean()[['energy_consumption', 'usable_area', 'area_to_energy_ratio', 'carbon_emission']]
baseline_data.reset_index(inplace=True)

# Initialize the Dash app with bootstrap components
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MATERIA])  
server = app.server
# Define the app layout
app.layout = dbc.Container([
    dcc.Location(id='url', refresh=True),
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
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id='hotel-name-input',
                    options=[{'label': name, 'value': name} for name in hotel_energy_area_data['name'].unique()],
                    value='',
                    searchable=True,
                    clearable=True,
                    placeholder="Search for a hotel...",
                    style={"border-radius": "15px", "width": "100%"}
                ),
            ], width=8),
            dbc.Col([
                html.A("Home", id="home-button", className="btn btn-primary ml-1", href="/", style={'height': '40px', 'line-height': '15px'})
            ], width=1)
        ])
    ], width=11, className="offset-2 mb-4")
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
                                hover_data=['hotelClass','hotelRanking','energy_consumption', 'usable_area', 'efficiency', 'carbon_emission'], 
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
    app.run_server(debug = True)
