# Heart Disease Predictor Model

## Project Overview
This project aims to identify the most influential risk factors associated with heart disease or heart attacks using the dataset titled "Heart_Disease_Health_Indicators." The research endeavor aims to construct an accurate predictive model for the early detection of individuals susceptible to heart disease. The primary motivation behind this investigation is to enhance healthcare interventions, reduce the incidence of cardiovascular ailments, and ultimately promote better cardiovascular health outcomes.

**Dataset Source**: [Kaggle Heart Disease Health Indicators](https://www.kaggle.com/datasets/bhaveshmisra/heart-disease-indicators/)

## Installation
Ensure you have the following dependencies installed:
- Python 3.x
- Pandas
- NumPy
- Scikit-Learn
- Seaborn
- Matplotlib

Install the dependencies using pip:
```sh
pip install pandas numpy scikit-learn seaborn matplotlib
```

### Data Preparation
1. **Import Required Libraries**:
   Import libraries such as Pandas, NumPy, Scikit-Learn, Seaborn, and Matplotlib.

2. **Load Data**:
   Load the dataset from a CSV file using Pandas.

3. **Visualize the Dataset**:
   Display the first few rows of the dataset to understand its structure.

4. **Clean the Data**:
   - Check for null values and data types.
   - Perform data cleaning by handling missing values and correcting data types.

### Exploratory Data Analysis
1. **Summary Statistics**:
   Calculate and display summary statistics for numerical features.

2. **Box Plots**:
   Create box plots for numerical features (e.g., BMI and Age) to identify outliers.

3. **Correlation Matrix**:
   Create a heatmap to visualize the correlation matrix.

### Feature Engineering
1. **Winsorization**:
   Apply Winsorization to handle outliers in BMI and Age.

2. **One-Hot Encoding**:
   Encode categorical features using one-hot encoding.

### Data Splitting and Scaling
1. **Train-Test Split**:
   Split the data into training (70%) and test (30%) sets.

2. **Standardization**:
   Standardize the features using StandardScaler.

### Machine Learning Models
#### Logistic Regression
1. **Hyperparameter Tuning**:
   Use GridSearchCV to tune hyperparameters for the logistic regression model.

2. **Model Training**:
   Train the logistic regression model with the best hyperparameters.

3. **Model Evaluation**:
   Evaluate the model using accuracy, precision, recall, F1-score, and confusion matrix.
   Adjust the threshold to optimize precision and recall.

4. **Feature Importance**:
   Identify the most significant features using model coefficients and feature selection techniques.

#### Random Forest Classifier
1. **Hyperparameter Tuning**:
   Use GridSearchCV to tune hyperparameters for the random forest classifier.

2. **Model Training**:
   Train the random forest classifier with the best hyperparameters.

3. **Model Evaluation**:
   Evaluate the model using accuracy, precision, recall, F1-score, and confusion matrix.
   Adjust the threshold to optimize precision and recall.

4. **Resampling**:
   Address class imbalance using resampling techniques.

5. **Calibrated Classifier**:
   Wrap the random forest classifier in a CalibratedClassifierCV for better probability estimates.

### Visualization
1. **ROC Curve**:
   Plot the ROC curve and calculate the ROC-AUC score.

2. **Precision-Recall Curve**:
   Plot the precision-recall curve and identify the optimal threshold.

3. **Class Distribution**:
   Visualize the class distribution before and after resampling.

4. **Calibration Curve**:
   Plot the calibration curve and calculate the Brier score.
   
## Results

### Logistic Regression
- **Best Hyperparameters**: {'C': 0.1, 'class_weight': 'balanced', 'max_iter': 1000, 'penalty': 'l2', 'random_state': 42, 'solver': 'lbfgs'}
- **Test Accuracy**: 75.1%
- **Precision**: 24.5%
- **Recall**: 79.3%
- **F1-Score**: 37.5%

### Random Forest Classifier
- **Best Hyperparameters**: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'random_state': 42}
- **Test Accuracy**: 89.8%
- **Precision**: 41.3%
- **Recall**: 14.3%
- **F1-Score**: 21.3%

### Random Forest Classifier (After Resampling)
- **Resampled Data Accuracy**: 77.9%
- **Resampled Data Precision**: 76.3%
- **Resampled Data Recall**: 81.1%
- **Resampled Data F1-Score**: 78.6%

### Key Findings
- The logistic regression model achieved a balanced recall and precision, making it suitable for identifying potential heart disease cases effectively.
- The random forest classifier, while highly accurate, initially had lower recall due to class imbalance.
- After resampling, the random forest classifier showed significantly improved recall and precision, making it a robust model for predicting heart disease.

## Conclusion

Predicting heart disease can significantly enhance healthcare interventions, reduce the incidence of cardiovascular ailments, and ultimately promote better cardiovascular health outcomes. The combination of logistic regression and random forest classifiers provides a robust approach to understand and predict heart disease risk.