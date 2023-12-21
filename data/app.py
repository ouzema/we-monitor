import pickle
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Load the model
with open('pred.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the dataset (ensure 'jumia2.csv' is in the same directory)
df = pd.read_csv('jumia2.csv')

app = Flask(__name__)

@app.route('/')
def index():
    # Prepare data for the chart
    chart_data = df.to_dict('records')
    return render_template('index.html', chart_data=chart_data)

@app.route('/data', methods=['GET'])  # Corrected route for predictions
def get_predictions():
    # Generate predictions using your model
    predictions = model.predict(df)  # Replace with model-specific input
    return jsonify({'predictions': predictions.tolist()})  # Return as a JSON list
