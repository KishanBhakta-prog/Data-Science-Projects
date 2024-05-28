# E-commerce Order Status Prediction

## Project Overview
Predicting the order status in an e-commerce dataset. Understanding and predicting order statuses in e-commerce is crucial for efficient supply chain management and customer satisfaction. 

**Dataset Source**: [Kaggle](https://www.kaggle.com/datasets/thedevastator/unlock-profits-with-e-commerce-sales-data/code)

## Installation
Ensure you have the following dependencies installed:
- Python 3.x
- Pandas
- NumPy
- Scikit-Learn
- Seaborn
- Matplotlib
- Imbalanced-learn

Install the dependencies using pip:
```sh
pip install pandas numpy scikit-learn seaborn matplotlib imbalanced-learn
```

### Data Preparation
1. **Import Required Libraries**:
   Import libraries such as Pandas, NumPy, Scikit-Learn, Seaborn, and Matplotlib.

2. **Load Data**:
   Load the dataset from a CSV file using Pandas.

3. **Visualize the Dataset**:
   Display the first few rows of the dataset to understand its structure.

4. **Perform Data Cleaning**:
   - Get a summary of the dataset and check for missing values.
   - Impute missing values using the most frequent strategy.
   - Drop rows with missing values in specific columns.
   - Create new features such as 'TotalAmount' by multiplying 'Qty' and 'Amount'.
   - One-hot encode categorical variables.

5. **Handle Missing Values and Outliers**:
   - Calculate Z-scores for numeric columns and handle outliers by removing rows with Z-scores above a threshold.

### Data Visualization
1. **Visualize Sales Over Time**:
   Create a line plot to visualize total sales over time.

### Machine Learning Models
#### Clustering Analysis
1. **Perform K-means Clustering**:
   Apply K-means clustering to segment customers based on features like 'Qty' and 'Amount'.

2. **Visualize Clusters**:
   Use scatter plots to visualize the customer segments.

3. **Evaluate Clustering**:
   Calculate metrics such as the Silhouette Score and Davies-Bouldin Index to evaluate the clustering results.

#### Supervised Learning
1. **Random Forest Classifier**:
   - Train a Random Forest classifier to predict order status.
   - Evaluate the model using accuracy, precision, recall, and F1-score.
   - Visualize the confusion matrix.

2. **Neural Network**:
   - Train a neural network to predict order status.
   - Evaluate the model using accuracy, precision, recall, and F1-score.
   - Visualize the confusion matrix and plot the learning curve.

### Model Evaluation
1. **Compare Models**:
   Compare the performance of the Random Forest classifier and the neural network using evaluation metrics.

2. **Feature Importance**:
   Identify and rank the most important features for each model.
   
## Results

### Clustering
- Achieved a silhouette score of 0.94.
- Visualized customer segmentation based on quantity and amount.

### Random Forest Classifier
- Achieved an accuracy of 64.3%.
- Evaluated using confusion matrix and classification report.

### Neural Network Classifier
- Achieved an accuracy of 78%.
- Evaluated using confusion matrix and classification report.

## Conclusion

Predicting order statuses can significantly enhance e-commerce operations, contributing to efficient supply chain management and improved customer satisfaction. The combination of clustering and supervised learning methods provides a robust approach to understand and predict order statuses.