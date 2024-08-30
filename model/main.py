import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

#function to test model

def create_model(data):

    X = data.drop(['diagnosis'], axis = 1)
    y = data['diagnosis']


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    X_train , X_test , y_train , y_test = train_test_split(X_scaled , y , test_size = 0.3 , random_state = 1)

    model = LogisticRegression(solver="liblinear")
    model.fit(X_train , y_train)

    y_pred = model.predict(X_test)
    print("Classification Report: ", classification_report(y_test, y_pred))

    return model , scaler

def get_clean_data():
    data = pd.read_csv('data/data.csv')
    data = data.drop(['id'],axis = 1)
    data['diagnosis'] = data['diagnosis'].map({'M':1 , 'B':0})
    return data

def main():

    data = get_clean_data()
    model , scaler = create_model(data)

    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model,f)

    with open('model/scaler.pkl' , 'wb') as f:
        pickle.dump(scaler , f)

    print(data.head())
    
    


if __name__ == "__main__":
    main()