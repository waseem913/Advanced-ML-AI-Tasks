

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib


# 1. Load dataset
df = pd.read_csv(
    r"E:\python programs\Ai_Ml internship tasks\2nd tasks\Task2_Churn_Pipeline\Telco-Customer-Churn.csv"
)


# Replace with your CSV path

# Drop customerID (not useful)
df = df.drop(columns=["customerID"])

# Target column
target = "Churn"
X = df.drop(columns=[target])
y = df[target].apply(lambda x: 1 if x == "Yes" else 0)  # Encode target as 0/1

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 3. Identify numerical & categorical columns

numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()


# 4. Preprocessing transformers

num_transformer = StandardScaler()
cat_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, numerical_features),
        ("cat", cat_transformer, categorical_features)
    ]
)


# 5. ML Pipeline

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))  # default classifier
])


# 6. Hyperparameter grid (separate for each classifier)

param_grid = [
    {
        "classifier": [LogisticRegression(max_iter=1000)],
        "classifier__C": [0.1, 1, 10]
    },
    {
        "classifier": [RandomForestClassifier(random_state=42)],
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [5, 10, None]
    }
]


# 7. GridSearchCV

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)


# 8. Evaluate on test set

y_pred = grid_search.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# 9. Export complete pipeline

joblib.dump(grid_search.best_estimator_, "churn_pipeline.pkl")
print("Pipeline saved as 'churn_pipeline.pkl'")
