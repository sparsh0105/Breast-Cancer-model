import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

def create_model(data):
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=1)

    model = LogisticRegression(solver="liblinear")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Classification Report: ", classification_report(y_test, y_pred))

    return model, scaler

def get_clean_data():
    data = pd.read_csv('E:/ooof/Breast-Cancer-model/data/data.csv')
    data = data.drop(['id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    print(data.describe().T)
    return data

def make_model():
    data = get_clean_data()
    model, scaler = create_model(data)

    with open('E:/ooof/Breast-Cancer-model/model/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('E:/ooof/Breast-Cancer-model/model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return data

def predict_with_custom_values(custom_values):
    # Load the model and scaler
    with open('E:/ooof/Breast-Cancer-model/model/model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('E:/ooof/Breast-Cancer-model/model/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Convert custom values to DataFrame
    custom_values_df = pd.DataFrame([custom_values])
    
    # Preprocess the custom values
    custom_values_scaled = scaler.transform(custom_values_df)
    
    # Make predictions
    predictions = model.predict(custom_values_scaled)
    probabilities = model.predict_proba(custom_values_scaled)
    scores = model.decision_function(custom_values_scaled)
    coefficients = model.coef_
    intercept = model.intercept_
    return {
        "predictions":predictions,"probabilities":probabilities,"scores":scores,"coefficients":coefficients,"intercept":intercept
    }

if __name__ == "__main__":
    # Uncomment this line if you need to re-train and save the model
    # make_model()

    # Example custom values for prediction
   # Example custom values likely to predict malignant
    custom_values = {
    'radius_mean': 20.0,         # Malignant tumors often have larger radii
    'texture_mean': 30.0,        # Higher texture values can be indicative of malignancy
    'perimeter_mean': 130.0,     # Larger perimeter values
    'area_mean': 1000.0,         # Larger area values
    'smoothness_mean': 0.15,     # Higher smoothness values might be associated with malignancy
    'compactness_mean': 0.15,    # Higher compactness can be indicative of malignant tumors
    'concavity_mean': 0.2,       # Higher concavity is often associated with malignant tumors
    'points_mean': 0.1,          # Higher points_mean values can be associated with malignancy
    'symmetry_mean': 0.2,        # Lower symmetry values can indicate malignancy
    'dimension_mean': 0.1,       # Larger dimension values
    'radius_se': 1.5,            # Larger radius_se values
    'texture_se': 2.0,           # Larger texture_se values
    'perimeter_se': 5.0,         # Larger perimeter_se values
    'area_se': 80.0,             # Larger area_se values
    'smoothness_se': 0.015,      # Larger smoothness_se values
    'compactness_se': 0.05,      # Higher compactness_se values
    'concavity_se': 0.05,        # Higher concavity_se values
    'points_se': 0.03,           # Larger points_se values
    'symmetry_se': 0.03,         # Larger symmetry_se values
    'dimension_se': 0.02,        # Larger dimension_se values
    'radius_worst': 25.0,        # Largest radius_worst values
    'texture_worst': 35.0,       # Higher texture_worst values
    'perimeter_worst': 150.0,    # Larger perimeter_worst values
    'area_worst': 1500.0,        # Larger area_worst values
    'smoothness_worst': 0.2,     # Higher smoothness_worst values
    'compactness_worst': 0.2,    # Higher compactness_worst values
    'concavity_worst': 0.2,      # Higher concavity_worst values
    'points_worst': 0.05,        # Larger points_worst values
    'symmetry_worst': 0.25,      # Lower symmetry_worst values
    'dimension_worst': 0.1       # Larger dimension_worst values
}

    
    predictions = predict_with_custom_values(custom_values)
    get_clean_data()

    print("Predictions:", predictions)
