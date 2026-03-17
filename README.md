# Diabetes Prediction Analysis

Exploratory data analysis and machine learning prediction on the Pima Indians Diabetes Dataset.

## Dataset
- 768 patients, 9 features
- Source: Kaggle - Pima Indians Diabetes Dataset
- Target variable: Diabetes diagnosis (1 = Yes, 0 = No)

## Analysis

### Key Findings
- Glucose is the strongest predictor of diabetes (correlation: 0.47)
- Diabetic patients have significantly higher BMI (median ~35 vs ~30)
- Diabetic patients tend to be older (median age ~36 vs ~27)
- Higher number of pregnancies is associated with increased diabetes risk

### Visualizations
- Glucose distribution by diabetes outcome
- BMI and age comparison using box plots
- Correlation heatmap of all features

### Machine Learning Model
- Algorithm: Logistic Regression
- Train/Test split: 80/20
- Model Accuracy: 74.68%

## Tools
- Python
- pandas
- matplotlib
- seaborn
- scikit-learn

## Author
Emi Rivera | Data Science Student | George Washington University
