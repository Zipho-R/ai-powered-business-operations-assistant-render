import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "training_tickets.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
CATEGORY_MODEL_PATH = os.path.join(MODEL_DIR, "category_model.pkl")
PRIORITY_MODEL_PATH = os.path.join(MODEL_DIR, "priority_model.pkl")
URGENCY_MODEL_PATH = os.path.join(MODEL_DIR, "urgency_model.pkl")


def clean_dataframe(df):
    required_columns = ["request", "category", "priority", "urgency"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing column in CSV: {col}")

    df = df.dropna(subset=required_columns)

    for col in required_columns:
        df[col] = df[col].astype(str).str.strip()

    valid_categories = ["IT", "HR", "Finance", "Operations"]
    valid_priorities = ["High", "Medium", "Low"]
    valid_urgencies = ["Immediate", "Not Immediate"]

    df = df[df["category"].isin(valid_categories)]
    df = df[df["priority"].isin(valid_priorities)]
    df = df[df["urgency"].isin(valid_urgencies)]

    return df


def train_single_model(X_vectorized, y, model_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(f"\n--- {model_name} Model ---")
    print("Accuracy:", round(accuracy_score(y_test, predictions), 2))
    print(classification_report(y_test, predictions))

    return model


def train_models():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df = clean_dataframe(df)

    print("Training records:", len(df))

    print("\nCategory counts:")
    print(df["category"].value_counts())

    print("\nPriority counts:")
    print(df["priority"].value_counts())

    print("\nUrgency counts:")
    print(df["urgency"].value_counts())

    X = df["request"]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )

    X_vectorized = vectorizer.fit_transform(X)

    category_model = train_single_model(X_vectorized, df["category"], "Category")
    priority_model = train_single_model(X_vectorized, df["priority"], "Priority")
    urgency_model = train_single_model(X_vectorized, df["urgency"], "Urgency")

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(category_model, CATEGORY_MODEL_PATH)
    joblib.dump(priority_model, PRIORITY_MODEL_PATH)
    joblib.dump(urgency_model, URGENCY_MODEL_PATH)

    print("\nAll models saved successfully.")
    print("Vectorizer:", VECTORIZER_PATH)
    print("Category model:", CATEGORY_MODEL_PATH)
    print("Priority model:", PRIORITY_MODEL_PATH)
    print("Urgency model:", URGENCY_MODEL_PATH)


if __name__ == "__main__":
    train_models()