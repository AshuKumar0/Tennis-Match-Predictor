import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/ml_dataset_elo.csv")
X = df.drop("target", axis=1)

X = X.astype({"surface_Clay": int, "surface_Grass": int, "surface_Hard": int})
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Accuracy: {acc_lr:.2%}")

xgb_model = XGBClassifier(eval_metric="logloss")
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {acc_xgb:.2%}")

import pickle

pickle.dump(lr_model, open("src/lr_model.pkl", "wb"))
print("Logistic Regression model saved!")