# fraud_detection.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest

def detect_fraudulent_clauses(clauses):
    """
    Detects risky/inconsistent clauses using anomaly detection (Isolation Forest).
    Returns a list of dictionaries: [{'text': clause, 'is_fraud': True/False}]
    """
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(clauses)

    model = IsolationForest(contamination=0.1, random_state=42)
    labels = model.fit_predict(X.toarray())  # -1 = anomaly

    results = []
    for clause, label in zip(clauses, labels):
        results.append({
            'text': clause,
            'is_fraud': label == -1
        })

    return results
