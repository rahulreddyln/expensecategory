import os
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

nltk.download('stopwords')

DATA_PATH = './dataset/expense_labels2.csv'
MODEL_DIR = '/opt/expensetrackergpt/project/app/expenses/models'

def clean_text(text):
    """Basic text cleaning."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    print("Initial category distribution:")
    print(df['category'].value_counts())

    # Drop rows with missing values
    df = df.dropna(subset=['category', 'description'])

    # Filter out underrepresented classes
    counts = df['category'].value_counts()
    valid_categories = counts[counts >= 3].index
    df = df[df['category'].isin(valid_categories)].reset_index(drop=True)

    print("\nFiltered category distribution:")
    print(df['category'].value_counts())

    # Clean text
    df['cleaned_description'] = df['description'].astype(str).apply(clean_text)

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df['category'])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_description'], y, test_size=0.2, stratify=y, random_state=42
    )

    # TF-IDF vectorizer
    tfidf = TfidfVectorizer(
        stop_words=stopwords.words('english'),
        ngram_range=(1, 2),
        max_features=5000
    )
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    # Logistic Regression classifier
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train_vec, y_train)

    # Evaluate classifier
    y_pred = clf.predict(X_test_vec)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    # Save classifier components
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, os.path.join(MODEL_DIR, 'logistic_regression.pkl'))
    joblib.dump(tfidf, os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
    joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))

    print("\n✔ Saved: classification model, vectorizer, label encoder")

    # Regression model (amount prediction)
    if 'amount' in df.columns and 'date' in df.columns:
        df_reg = df.dropna(subset=['amount', 'category', 'date']).copy()

        # Flexible date parsing
        df_reg['date_parsed'] = pd.to_datetime(df_reg['date'], errors='coerce')
        df_reg = df_reg.dropna(subset=['date_parsed'])

        # Extract features
        df_reg['month'] = df_reg['date_parsed'].dt.month
        df_reg['day'] = df_reg['date_parsed'].dt.day
        df_reg['year'] = df_reg['date_parsed'].dt.year

        print("✔ Parsed dates preview:")
        print(df_reg[['date', 'date_parsed']].head())
        print("✔ Regression dataset shape:", df_reg.shape)

        # Filter unknown categories
        df_reg = df_reg[df_reg['category'].isin(le.classes_)]

        if not df_reg.empty:
            category_encoded = le.transform(df_reg['category'])
            X_reg = pd.DataFrame({
                'category': category_encoded,
                'month': df_reg['month'].values,
                'day': df_reg['day'].values,
                'year': df_reg['year'].values
            })
            y_reg = df_reg['amount'].values

            rf = RandomForestRegressor(n_estimators=300, random_state=42)
            rf.fit(X_reg, y_reg)

            joblib.dump(rf, os.path.join(MODEL_DIR, 'random_forest_regressor.pkl'))
            print("✔ Regression model saved as random_forest_regressor.pkl.")
        else:
            print("⚠ No valid data left for regression after filtering.")
    else:
        print("⚠ 'amount' or 'date' column not found. Skipping regression.")

if __name__ == "__main__":
    main()

