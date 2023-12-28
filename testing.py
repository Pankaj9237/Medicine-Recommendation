import pickle
import pandas as pd 
import numpy as np

with open('knn_model.pkl', 'rb') as file:
    knn_model = pickle.load(file)


data = pd.read_csv('New_Medicine_Prediction_Dataset.csv')
def recommend_drugs_ml(user_input_drug):

    drug_scores = {}

    user_data = data[data['drugName'] == user_input_drug]
    user_condition = user_data['condition'].values[0]

      # Using only KNNBasic

    drug_scores['KNNBasic'] = []
    for drug in data['drugName'].unique():
        condition = data[data['drugName'] == drug]['condition'].values[0]

        if drug != user_input_drug and condition == user_condition:
            prediction = knn_model.predict(user_input_drug, drug)
            rating = prediction.est
            useful_count = user_data['usefulCount'].values[0]
            price = data[data['drugName'] == drug]['price'].values[0]  # Get the price
            side_effect = data[data['drugName'] == drug]['sideEffect0'].values[0]  # Get the side effect

            score = (0.5 * rating) + (0.5 * useful_count)

            drug_scores['KNNBasic'].append({
                'Drug': drug,
                'Condition': condition,  # Include condition in the output
                'Score': score,
                'Price': price,  # Include price in the output
                'Side Effect': side_effect  # Include side effect in the output
            })

    return drug_scores


def reco(user_input_drug):

  recommendations = recommend_drugs_ml(user_input_drug)

  for model_name, scores in recommendations.items():
    sorted_scores = sorted(scores, key=lambda x: x['Score'], reverse=True)[:5]
  print(sorted_scores)
      


reco("Lybrel")