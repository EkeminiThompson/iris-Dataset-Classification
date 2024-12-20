# Iris Dataset Classification Models Comparison

This project compares several classification models on the Iris dataset. The models evaluated include:

- Logistic Regression
- Decision Tree
- Support Vector Machine (SVM)
- Random Forest
- K-Nearest Neighbors (KNN)

## Steps in the Project:
1. **Data Preprocessing**: 
   - The Iris dataset is loaded and split into training and testing sets.
   - Features are scaled using StandardScaler for models that require feature scaling (e.g., Logistic Regression, SVM, KNN).
   
2. **Model Training and Evaluation**: 
   - Multiple classification models are trained and evaluated using cross-validation and test data.
   - The Random Forest model undergoes hyperparameter tuning with GridSearchCV.
   
3. **Performance Metrics**: 
   - Models are evaluated based on accuracy, confusion matrix, and classification report.

4. **Model Saving**: 
   - The best model (Random Forest) is saved using `joblib` for future use.

## Dependencies:
- numpy
- pandas
- scikit-learn
- joblib

## How to Run:
1. Clone this repository to your local machine.
2. Install dependencies (using `pip` or `conda`).
3. Run the Python script to see model comparisons and performance metrics.

## License:
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
