# -*- coding: utf-8 -*-
"""V_DiseaseToDrug.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lasSLjKWEUxDtnUCZvu6NOr7VVrJUgoa
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

data = pd.read_csv('New_Medicine_Prediction_Dataset.csv')  # Replace with the actual path to your dataset

user_condition = input("Enter your condition: ")

filtered_drugs = data[data['condition'] == user_condition]

if not filtered_drugs.empty:
    X = filtered_drugs[['rating', 'usefulCount']]
    y = filtered_drugs['rating']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
    random_forest.fit(X_train, y_train)

    predictions = random_forest.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print(f"Random Forest Model")
    print(f"R-squared (R2) Score: {r2:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")

    test_set = filtered_drugs.loc[y_test.index].copy()
    test_set['predicted_rating'] = predictions

    recommended_drugs = test_set.sort_values(by='predicted_rating', ascending=False)
    recommended_drugs_extended = recommended_drugs[['drugName', 'rating', 'usefulCount', 'predicted_rating', 'price', 'sideEffect0']].head()

    print(f"Top recommended drugs for {user_condition} using Random Forest:")
    print(recommended_drugs_extended)

else:
    print("No drugs found for your condition.")