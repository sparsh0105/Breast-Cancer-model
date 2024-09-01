import io
import pickle
from flask import Flask, Response, jsonify, request
import pandas as pd
from flask_cors import CORS
import plotly.io as pio
import gzip
import plotly.graph_objects as go
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  

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
    
    
    # Convert arrays to lists
    return {
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist(),
    }

def get_graphs(custom_values):
    features = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
        'smoothness_mean', 'compactness_mean', 'concavity_mean',
        'points_mean', 'symmetry_mean', 'dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se',
        'smoothness_se', 'compactness_se', 'concavity_se',
        'points_se', 'symmetry_se', 'dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
        'smoothness_worst', 'compactness_worst', 'concavity_worst',
        'points_worst', 'symmetry_worst', 'dimension_worst'
    ]
    
    # Convert custom values to DataFrame
    df = pd.DataFrame([custom_values])

    # Ensure df has only the required features (drop any extras like 'diagnosis')
    df = df[features]

    # Load the scaler
    with open('E:\ooof\Breast-Cancer-model\model\scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Apply the scaler
    normalized_user_input = scaler.transform(df)
    
    # Convert normalized data back to dictionary
    normalized_user_input_dict = pd.DataFrame(normalized_user_input, columns=features).iloc[0].to_dict()

    print(normalized_user_input_dict)

    categories = [
        'radius ', 'texture ', 'perimeter ', 'area ', 
        'smoothness ', 'compactness ', 'concavity ', 
        'points ', 'symmetry ', 'dimension '
    ]
    meanData = [
        normalized_user_input_dict['radius_mean'], normalized_user_input_dict['texture_mean'], normalized_user_input_dict['perimeter_mean'],
        normalized_user_input_dict['area_mean'], normalized_user_input_dict['smoothness_mean'], normalized_user_input_dict['compactness_mean'],
        normalized_user_input_dict['concavity_mean'], normalized_user_input_dict['points_mean'], normalized_user_input_dict['symmetry_mean'], normalized_user_input_dict['dimension_mean']
    ]
    seData = [
        normalized_user_input_dict['radius_se'], normalized_user_input_dict['texture_se'], normalized_user_input_dict['perimeter_se'],
        normalized_user_input_dict['area_se'], normalized_user_input_dict['smoothness_se'], normalized_user_input_dict['compactness_se'],
        normalized_user_input_dict['concavity_se'], normalized_user_input_dict['points_se'], normalized_user_input_dict['symmetry_se'], normalized_user_input_dict['dimension_se']
    ]
    worstData = [
        normalized_user_input_dict['radius_worst'], normalized_user_input_dict['texture_worst'], normalized_user_input_dict['perimeter_worst'],
        normalized_user_input_dict['area_worst'], normalized_user_input_dict['smoothness_worst'], normalized_user_input_dict['compactness_worst'],
        normalized_user_input_dict['concavity_worst'], normalized_user_input_dict['points_worst'], normalized_user_input_dict['symmetry_worst'], normalized_user_input_dict['dimension_worst']
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=meanData,
        theta=categories,
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.5)',  # Blue color with 50% transparency
        name="Mean Values"
    ))
    fig.add_trace(go.Scatterpolar(
        r=seData,
        theta=categories,
        fill='toself',
        fillcolor='rgba(255, 127, 14, 0.5)',  
        name="SE Values"
    ))
    fig.add_trace(go.Scatterpolar(
        r=worstData,
        theta=categories,
        fill='toself',
        fillcolor='rgba(44, 160, 44, 0.5)',  
        name="Worst Values"
    ))
    
    # Updating layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]  # Ensure your scaled data falls within this range
            ),
        ),
        showlegend=True,
        title=""
    )
    
    graph_json = pio.to_json(fig)

    # Compress JSON data with gzip
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode='wb') as file:
        file.write(graph_json.encode('utf-8'))

    compressed_json = buffer.getvalue()

    # Create the Flask response with correct headers
    response = Response(compressed_json, content_type='application/json')
    response.headers['Content-Encoding'] = 'gzip'
    response.headers['Content-Length'] = len(compressed_json)

    return response

@app.route("/api/prediction", methods=['POST'])
def get_prediction():
    data = request.get_json()

    # Extract the values corresponding to the expected keys
    custom_values = {key: data.get(key) for key in data.keys()}
    
    predictions = predict_with_custom_values(custom_values)

    return jsonify(predictions)

@app.route("/api/graph", methods=["POST"])
def get_graph():
    data = request.get_json()
    custom_values = {key: data.get(key) for key in data.keys()}
    
    return get_graphs(custom_values)

if __name__ == '__main__':
    app.run(debug=True)
