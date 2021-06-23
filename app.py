from flask import Flask,render_template
import pandas as pd
from Stress_detection import dataset_handling
import os
import json
import plotly

app = Flask(__name__)
d = dataset_handling()

def dataset():
    dataframe_hrv = d.creating_dataframe()
    dataframe_hrv = d.fix_stress_labels(df=dataframe_hrv)
    dataframe_hrv = d.missing_values(dataframe_hrv)
    selected_x_columns,exported_pipeline=d.train_and_test(dataframe_hrv)
    return selected_x_columns,exported_pipeline

def plot():
    selected_x_columns,exported_pipeline = dataset()
    fig = []
    for _ in os.listdir(path="datasets"):
        files = "datasets//"  + _
        input_df = pd.read_csv(files)
        figure = d.plotFitBitReading(input_df,exported_pipeline,selected_x_columns)
        fig.append(figure)
    figureJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    ids = ['figure-{}'.format(i) for i, _ in enumerate(fig)]
    return figureJSON,ids

@app.route('/')
def index():
    figureJSON,ids = plot()
    return render_template('index.html',ids=ids,figuresJSON=figureJSON)
    

if __name__ == '__main__':
    app.run(debug=True)
