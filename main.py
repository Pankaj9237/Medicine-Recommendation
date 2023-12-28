from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


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
  return sorted_scores

# Your recommendation function (reco function) and other necessary imports
# ... (assuming the reco function and necessary imports are defined)

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Alternate.html')
def alternate_page():
    # Your logic to render the Alternate.html file
    return render_template('Alternate.html')

@app.route('/Medicine.html')
def Medicine_page():
    # Your logic to render the Alternate.html file
    return render_template('Medicine.html')

@app.route('/first.html')
def first_page():
    # Your logic to render the Alternate.html file
    return render_template('first.html')

@app.route('/Heart.html')
def Heart_page():
    # Your logic to render the Alternate.html file
    return render_template('Heart.html')

@app.route('/Diabetices.html')
def Diabetices_page():
    # Your logic to render the Alternate.html file
    return render_template('Diabetices.html')

@app.route('/Parkinson.html')
def Parkinson_page():
    # Your logic to render the Alternate.html file
    return render_template('Parkinson.html')


@app.route('/drug', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input_drug = request.form.get("user_input_drug")
        recommendations = reco(user_input_drug)
        if recommendations:
            return jsonify({"recommendations": recommendations})
        else:
            return jsonify({"message": "No recommendations found."}), 404
    else:
        return jsonify({"message": "Error processing request."}), 400


@app.route('/diseasetodrug', methods=['GET', 'POST'])
def DisToDrug():
    print("Runing1")
    if request.method == 'POST':
        user_condition = request.form.get("user_input_drug")
        print(user_condition)
        filtered_drugs = data[data['condition'] == user_condition]

        if not filtered_drugs.empty:
            X = filtered_drugs[['rating', 'usefulCount']]
            y = filtered_drugs['rating']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
            random_forest.fit(X_train, y_train)

            predictions = random_forest.predict(X_test)

            test_set = filtered_drugs.loc[y_test.index].copy()
            test_set['predicted_rating'] = predictions

            recommended_drugs = test_set.sort_values(by='predicted_rating', ascending=False)
            recommended_drugs_extended = recommended_drugs[['drugName', 'price','sideEffect0']].head()
            print(type(recommended_drugs_extended))
            df_dict = recommended_drugs_extended.to_dict(orient='records')
            print(df_dict)
            return jsonify({"recommendations": df_dict})
        else:
            return jsonify({"message": "No recommendations found."}), 404
        
        # return render_template('DisToDrug.html', result=result)

    return jsonify({"message": "Error processing request."}), 400

# if recommendations:
#             return jsonify({"recommendations": recommendations})
#         else:
#             return jsonify({"message": "No recommendations found."}), 404
#     else:
#         return jsonify({"message": "Error processing request."}), 400


if __name__ == '__main__':
    app.run(debug=True)
