## Project Overview
This project employs an ensemble of advanced regression models to predict county-level solar capacity in Megawatts of Alternating Current (MW-AC) across the United States. The aim is to identify regions where solar energy resources are underutilized, thereby supporting informed decision-making in both public and private sectors to promote a more efficient and responsible energy landscape.

**Dataset Source**: [National Renewable Energy Laboratory](https://www.nrel.gov/gis/data-solar.html)

## Installation
Ensure you have the following dependencies installed:
- Python 3.x
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- SciPy
- Statsmodels

Install the dependencies using pip:
```sh
pip install pandas numpy scikit-learn matplotlib scipy statsmodels
```

## Data Preparation

The data preparation for this project involved several key steps across multiple datasets, ensuring a comprehensive and clean dataset for analysis.

1. **Data Collection**:
   Data was gathered from various Excel sheets and combined into a single dataset. Each dataset underwent individual preprocessing before merging.

2. **FIPS Codes**:
   FIPS codes were prepared and standardized to ensure consistency across datasets. This allowed for accurate merging and analysis.

3. **Merging Additional Datasets**:
   Additional datasets were merged based on common keys such as FIPS codes, enriching the main dataset with more features and information.

4. **Standardization**:
   Features were standardized to ensure they are on the same scale, which is crucial for the performance of machine learning algorithms.

5. **Train-Test Split**:
   The data was split into training and testing sets to evaluate the model's performance accurately.

6. **Final Dataset**:
   The final dataset included multiple standardized features prepared from the initial merging and cleaning process, ready for modeling.
   
   ## Results

### Model Performance

#### Random Forest Regressor
- **R² Score**: 0.7974
- The Random Forest model demonstrated a strong performance in predicting solar capacity.

#### Gradient Boosting Regressor
- **R² Score**: 0.8813
- The Gradient Boosting model outperformed other models, showing the highest accuracy in predictions.

#### Support Vector Machine (SVM)
- **R² Score**: -0.0191
- The SVM model did not perform well on this dataset.

#### Decision Tree Regressor
- **R² Score**: 0.7355
- The Decision Tree model showed reasonable performance but was less accurate than the Random Forest and Gradient Boosting models.

### Ensemble Model
- **R² Score**: 0.8176
- Combining predictions from all models, the ensemble model provided robust and balanced predictions.

### Top Performing Counties
- Identified the counties with the highest discrepancy in predictions versus actual solar capacity.
- Notably, San Bernardino, CA, and Otero, NM showed the largest differences, indicating potential areas for model improvement.

### Overestimation Analysis
- **398 Counties**: Over-estimated by more than 5 MW-AC.
- Focus on addressing these discrepancies to refine model accuracy.

### Time Series Analysis
- Conducted Fourier Transform and Seasonal Decomposition to understand the GHI data's periodicity and trends.
- The Adfuller test indicated non-stationarity in the GHI data, which was handled appropriately.

## Conclusion

The solar capacity prediction project successfully identified and utilized key factors influencing solar energy generation across various counties. The Gradient Boosting model showed the highest accuracy, indicating its suitability for this type of regression problem. However, the ensemble model provided a balanced and robust prediction by combining multiple algorithms.

Future work should focus on refining the model to address overestimation in specific counties and exploring additional features that could enhance prediction accuracy. Time series analysis of GHI data provided valuable insights into the seasonal trends and helped improve the overall model performance.

This project demonstrates the potential of machine learning models to predict solar capacity accurately, aiding in strategic planning and decision-making for solar energy projects.