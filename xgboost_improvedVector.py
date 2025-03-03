import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import joblib

# Load Data
df = pd.read_csv('synthetic_defect_data.csv')
print(df['Outage Risk (1/0)'].value_counts(normalize=True))  # Check class distribution

# Fill missing values
df.fillna({'Description': ''}, inplace=True)

# **Feature Engineering**
# Interaction feature (Severity × Component)
df['Severity_Component'] = df['Severity'] * df['Component']

# **TF-IDF Features for Description**
vectorizer = TfidfVectorizer(max_features=50)  # Extract up to 50 most important words
tfidf_features = vectorizer.fit_transform(df['Description']).toarray()

# **Fix: Adjust Column Names Dynamically**
num_tfidf_features = tfidf_features.shape[1]  # Get actual feature count
tfidf_df = pd.DataFrame(tfidf_features, columns=[f"word_{i}" for i in range(num_tfidf_features)])

# **Fix: Ensure Same Index for Concatenation**
df = df.reset_index(drop=True)
tfidf_df = tfidf_df.reset_index(drop=True)

# Concatenate new features
df = pd.concat([df, tfidf_df], axis=1)

# Define features and target
X = df.drop(columns=['Outage Risk (1/0)', 'Description'])
y = df['Outage Risk (1/0)']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardizing numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# **XGBoost Model with Hyperparameter Tuning**
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Hyperparameter tuning for XGBoost using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

grid_search = GridSearchCV(xgb_model, param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get best hyperparameters
best_xgb_model = grid_search.best_estimator_
print(f"Best XGBoost parameters: {grid_search.best_params_}")

# **RandomForest with Hyperparameter Tuning**
rf_model = RandomForestClassifier(random_state=42)

# Hyperparameter tuning for RandomForest using GridSearchCV
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search_rf.fit(X_train, y_train)

# Get best hyperparameters
best_rf_model = grid_search_rf.best_estimator_
print(f"Best RandomForest parameters: {grid_search_rf.best_params_}")

# **Stacking Classifier**
# Using XGBoost, RandomForest, and Logistic Regression as base learners
base_learners = [
    ('xgb', best_xgb_model),
    ('rf', best_rf_model),
    ('lr', LogisticRegression(random_state=42)),
]

# Using Logistic Regression as the final estimator
stacking_model = StackingClassifier(estimators=base_learners, final_estimator=LogisticRegression())

# Train the stacking model
stacking_model.fit(X_train, y_train)

# **Model Evaluation**
y_pred_stack = stacking_model.predict(X_test)

print("\nStacking Classifier Classification Report:\n", classification_report(y_test, y_pred_stack))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_stack))

# **Save the stacking model, vectorizer, and scaler**
joblib.dump(stacking_model, 'outage_risk_stacking_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')  # Save the vectorizer for later use
joblib.dump(scaler, 'scaler.joblib')  # Save the scaler for later use
print("Stacking model, vectorizer, and scaler trained and saved.")
