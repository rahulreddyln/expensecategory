import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

def predict_category(description):
    return "Miscellaneous"  # Default for now

def predict_next_month_by_category(expenses):
    if not expenses:
        return {}

    data = []
    for e in expenses:
        expense_date = e.date if isinstance(e.date, datetime) else datetime.strptime(e.date, "%Y-%m-%d")
        data.append({
            'date': expense_date,
            'category': e.category,
            'amount': e.amount
        })
    df = pd.DataFrame(data)

    df['month'] = df['date'].dt.month
    df['category_encoded'] = LabelEncoder().fit_transform(df['category'])

    X = df[['category_encoded', 'month']]
    y = df['amount']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    le = LabelEncoder()
    le.fit(df['category'])
    categories = df['category'].unique()

    next_month = datetime.today() + relativedelta(months=1)
    predictions = {}
    for cat in categories:
        cat_encoded = le.transform([cat])[0]
        X_pred = np.array([[cat_encoded, next_month.month]])
        pred = model.predict(X_pred)[0]
        predictions[cat] = {next_month.strftime('%Y-%m'): round(pred, 2)}

    return predictions

def aggregate_expenses_by_month(expenses):
    if not expenses:
        return pd.DataFrame()

    data = []
    for e in expenses:
        expense_date = e.date if isinstance(e.date, datetime) else datetime.strptime(e.date, "%Y-%m-%d")
        data.append({
            'date': expense_date,
            'category': e.category,
            'amount': e.amount
        })

    df = pd.DataFrame(data)
    df['year_month'] = df['date'].dt.strftime('%Y-%m')
    summary = df.groupby(['year_month', 'category'])['amount'].sum().reset_index()
    summary.rename(columns={'amount': 'total'}, inplace=True)
    return summary

