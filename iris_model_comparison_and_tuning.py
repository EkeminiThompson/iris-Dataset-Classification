# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib  # To save the best model

# Function to load and split the data
def load_and_split_data(test_size=0.3, random_state=42):
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# Function to scale the features
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

# Function to train and evaluate models using cross-validation
def evaluate_models(models, X_train_scaled, y_train):
    results = {}
    for model_name, model in models.items():
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        results[model_name] = {
            'Mean CV Score': np.mean(cv_scores),
            'Std CV Score': np.std(cv_scores)
        }
    return results

# Function to perform hyperparameter tuning with GridSearchCV
def tune_random_forest(X_train_scaled, y_train):
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train_scaled, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

# Function to train the best model and evaluate on test data
def train_and_evaluate_best_model(best_model, X_train_scaled, X_test_scaled, y_train, y_test):
    best_model.fit(X_train_scaled, y_train)
    y_pred = best_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy: .4f}")
    print("Confusion Matrix:")
    print(confusion_mat)
    print("Classification Report:")
    print(class_report)
    
    return best_model

# Function to save the trained model
def save_model(model, model_filename='best_model.pkl'):
    joblib.dump(model, model_filename)
    print(f"Model saved as {model_filename}")

# Main execution
if __name__ == "__main__":
    # Load and split data
    X_train, X_test, y_train, y_test = load_and_split_data()

    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=200),
        'Decision Tree': DecisionTreeClassifier(),
        'SVM': SVC(kernel='linear'),
        'Random Forest': RandomForestClassifier(),
        'KNN': KNeighborsClassifier()
    }

    # Evaluate models using cross-validation
    results = evaluate_models(models, X_train_scaled, y_train)

    # Print cross-validation results for each model
    print("Cross-validation results:")
    for model_name, result in results.items():
        print(f"{model_name} - Mean CV Score: {result['Mean CV Score']: .4f}, Std CV Score: {result['Std CV Score']: .4f}")

    # Tune Random Forest model
    best_rf_model, best_params = tune_random_forest(X_train_scaled, y_train)
    print(f"\nBest hyperparameters for Random Forest: {best_params}")

    # Train and evaluate the best model on test data
    print("\nEvaluating Random Forest on test data:")
    best_rf_model = train_and_evaluate_best_model(best_rf_model, X_train_scaled, X_test_scaled, y_train, y_test)

    # Save the best model
    save_model(best_rf_model)
