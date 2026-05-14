import os
import joblib


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
CATEGORY_MODEL_PATH = os.path.join(MODEL_DIR, "category_model.pkl")
PRIORITY_MODEL_PATH = os.path.join(MODEL_DIR, "priority_model.pkl")
URGENCY_MODEL_PATH = os.path.join(MODEL_DIR, "urgency_model.pkl")


def load_models():
    required_files = [
        VECTORIZER_PATH,
        CATEGORY_MODEL_PATH,
        PRIORITY_MODEL_PATH,
        URGENCY_MODEL_PATH
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                "Model files not found. Run: python AITicketClassifier/train_model.py"
            )

    vectorizer = joblib.load(VECTORIZER_PATH)
    category_model = joblib.load(CATEGORY_MODEL_PATH)
    priority_model = joblib.load(PRIORITY_MODEL_PATH)
    urgency_model = joblib.load(URGENCY_MODEL_PATH)

    return vectorizer, category_model, priority_model, urgency_model


def predict_with_confidence(model, X):
    prediction = model.predict(X)[0]
    confidence = 0

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        confidence = round(max(probabilities) * 100, 1)

    return prediction, confidence


def predict_ticket(ticket_text):
    vectorizer, category_model, priority_model, urgency_model = load_models()

    X = vectorizer.transform([ticket_text])

    category, category_confidence = predict_with_confidence(category_model, X)
    priority, priority_confidence = predict_with_confidence(priority_model, X)
    urgency, urgency_confidence = predict_with_confidence(urgency_model, X)

    return {
        "category": category,
        "category_confidence": category_confidence,
        "priority": priority,
        "priority_confidence": priority_confidence,
        "urgency": urgency,
        "urgency_confidence": urgency_confidence
    }


# Keeps your old import working:
# from AITicketClassifier.predict import predict_category
def predict_category(ticket_text):
    result = predict_ticket(ticket_text)
    return result["category"], result["category_confidence"]


if __name__ == "__main__":
    while True:
        text = input("Enter ticket or type quit: ").strip()

        if text.lower() == "quit":
            break

        result = predict_ticket(text)

        print("\nCategory:", result["category"], f"({result['category_confidence']}%)")
        print("Priority:", result["priority"], f"({result['priority_confidence']}%)")
        print("Urgency:", result["urgency"], f"({result['urgency_confidence']}%)")
        print()