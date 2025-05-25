import os
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

logistic_model = joblib.load(os.path.join(MODEL_DIR, 'logistic_regression.pkl'))
tfidf_vectorizer = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
random_forest_model = joblib.load(os.path.join(MODEL_DIR, 'random_forest_regressor.pkl'))
label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))

def predict_category(description):
    vec = tfidf_vectorizer.transform([description])
    pred_encoded = logistic_model.predict(vec)[0]
    category = label_encoder.inverse_transform([pred_encoded])[0]
    return category

def predict_future_expense(expenses):
    # expenses is expected to be a list of Expense objects

    if not expenses:
        return 0.0

    categories = [e.category for e in expenses]
    try:
        most_common_cat = max(set(categories), key=categories.count)
    except ValueError:
        most_common_cat = 'Others'

    cat_encoded = label_encoder.transform([most_common_cat])[0]

    # Model expects a 2D array with 1 feature
    X_pred = np.array([[cat_encoded]])
    prediction = random_forest_model.predict(X_pred)
    return round(prediction[0], 2)

