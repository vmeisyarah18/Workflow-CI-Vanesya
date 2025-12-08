import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Parser untuk MLflow Project
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

# Load dataset
df = pd.read_csv(args.data_path)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

with mlflow.start_run():

    # Model training
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Prediksi
    preds = model.predict(X_test)
    preds_proba = model.predict_proba(X_test)[:, 1]

    # Metric
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, preds_proba)

    # Manual logging
    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("model", "LogisticRegression")

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", auc)

    # Log model
    mlflow.sklearn.log_model(model, "model")

print("Training selesai — Accuracy:", acc)
