from flask import Flask
from faker import Faker
import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px

app = Flask(__name__)
fake = Faker()

@app.route('/')
def hello():
    return "Hi, this is elong."

@app.route('/data')
def data():
    # Generate some fake data
    data = {'x': range(10), 'y': np.random.randn(10)}
    df = pd.DataFrame(data)
    
    # Plot the data
    fig = px.scatter(df, x='x', y='y', title='Random Data')
    graph = dcc.Graph(figure=fig)
    
    return html.Div(children=[html.H1('Random Data'), graph])

if __name__ == '__main__':
    app.run(debug=True)
