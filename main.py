import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

# Sample data
df = pd.DataFrame({
    "X": [1, 2, 3, 4, 5],
    "Y": [2, 3, 5, 7, 11]
})

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    dcc.Graph(
        id='scatter-plot',
        figure=px.scatter(df, x='X', y='Y', title='Scatter Plot').update_traces(mode='markers')
    )
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
