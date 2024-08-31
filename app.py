import pickle
from flask import Flask, jsonify, request
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def predict_with_custom_values(custom_values):
    # Load the model and scaler
    with open('E:/ooof/Breast-Cancer-model/model/model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('E:/ooof/Breast-Cancer-model/model/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Expected feature names in the same order as during training
    expected_features = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 
        'compactness_mean', 'concavity_mean', 'points_mean', 'symmetry_mean', 'dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 
        'compactness_se', 'concavity_se', 'points_se', 'symmetry_se', 'dimension_se', 
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 
        'compactness_worst', 'concavity_worst', 'points_worst', 'symmetry_worst', 
        'dimension_worst'
    ]
    
    # Ensure custom_values contains all expected features, filling missing ones with 0
    custom_values_complete = {key: custom_values.get(key, 0) for key in expected_features}
    
    # Convert custom values to DataFrame
    custom_values_df = pd.DataFrame([custom_values_complete])
    
    # Preprocess the custom values
    custom_values_scaled = scaler.transform(custom_values_df)
    
    # Make predictions
    predictions = model.predict(custom_values_scaled)
    probabilities = model.predict_proba(custom_values_scaled)
    scores = model.decision_function(custom_values_scaled)
    coefficients = model.coef_
    intercept = model.intercept_
    
    # Convert arrays to lists
    return {
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist(),
        "scores": scores.tolist(),
        "coefficients": coefficients.tolist(),
        "intercept": intercept.tolist()
    }

@app.route("/api/graph", methods=['POST'])
def get_graphs():
    data = request.get_json()

    # Extract the values corresponding to the expected keys
    custom_values = {key: data.get(key) for key in data.keys()}
    
    predictions = predict_with_custom_values(custom_values)

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
