import numpy as np
import sklearn

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

from flask import Flask, request, jsonify, render_template
import pickle
import joblib
import numpy as np

app = Flask(__name__)
model = pickle.load(open('//Users//daviddela//Flask//my_flask_project//model1.pkl', 'rb'))
print(type(model))


# # model = joblib.load('model.pkl')
# # joblib.dump(model, 'model_update.pkl')
# # model = pickle.load(open('model_update.pkl', 'rb'))
# model = None

# def load_model():
#     global model
#     model = joblib.load('model.pkl')
#     print(type(model))
#
# load_model()
#
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    player_value = data.get('value')
    player_age = data.get('age')
    release_clause = data.get('releaseClause')
    movement_reaction = data.get('movementReaction')
    player_potential = data.get('potential')

    prediction = model.predict([[player_value, release_clause, player_age, player_potential, movement_reaction]])

    response = {'player_rating': prediction}
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
    # app.run(host="0.0.0.0")            # deploy


