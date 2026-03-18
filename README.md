# Diabetes Prediction Analysis

Exploratory data analysis, statistical testing, and machine learning prediction on the Pima Indians Diabetes Dataset.

## Dataset
- 768 patients, 9 features
- Source: Kaggle - Pima Indians Diabetes Dataset
- Target variable: Diabetes diagnosis (1 = Yes, 0 = No)
- 34.9% of patients diagnosed with diabetes

## Analysis

### Key Findings
- Glucose is the strongest predictor of diabetes (correlation: 0.47, p < 0.0001)
- BMI and Age are also highly significant (p < 0.0001)
- BloodPressure was the only non-significant feature (p = 0.0715)
- PCA explains 49.5% of variance with partial separation between groups along PC1

### Visualizations
- Glucose distribution by diabetes outcome
- BMI and age comparison using box plots
- Correlation heatmap of all features
- Statistical significance plot (-log10 p-values)
- PCA of clinical features

### Machine Learning Model
- Algorithm: Logistic Regression
- Train/Test split: 80/20
- Model Accuracy: 74.68%

## Tools
- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn, scipy

## Author
Emi Rivera | Data Science Student | George Washington University
