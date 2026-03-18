import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ── Load Data ──
df = pd.read_csv('diabetes.csv')

# Preview the first 5 rows
print(df.head())

# Check the shape of the dataset (rows, columns)
print(df.shape)

# Basic statistics of the dataset
print(df.describe())

# ── Plot 1: Glucose Distribution ──
df.groupby('Outcome')['Glucose'].hist(alpha=0.5, bins=20)
plt.xlabel('Glucose Level')
plt.ylabel('Count')
plt.title('Glucose Distribution by Diabetes Outcome')
plt.legend(['No Diabetes', 'Diabetes'])
plt.savefig('glucose_distribution.png')
print("Glucose plot saved!")

# ── Plot 2: BMI and Age Analysis ──
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.boxplot(x='Outcome', y='BMI', data=df)
plt.xticks([0, 1], ['No Diabetes', 'Diabetes'])
plt.title('BMI by Diabetes Outcome')

plt.subplot(1, 2, 2)
sns.boxplot(x='Outcome', y='Age', data=df)
plt.xticks([0, 1], ['No Diabetes', 'Diabetes'])
plt.title('Age by Diabetes Outcome')

plt.tight_layout()
plt.savefig('bmi_age_analysis.png')
print("BMI and Age plot saved!")

# ── Plot 3: Correlation Heatmap ──
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap of All Features')
plt.savefig('correlation_heatmap.png')
print("Heatmap saved!")

# ── Machine Learning Model ──
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2%}")

# ── Statistical Tests ──
features_test = ['Glucose', 'BMI', 'Age', 'Insulin', 'BloodPressure']

print("\n=== Statistical Tests ===")
for feature in features_test:
    group1 = df[df['Outcome'] == 1][feature].dropna()
    group2 = df[df['Outcome'] == 0][feature].dropna()
    t_stat, p_value = stats.ttest_ind(group1, group2)
    significance = "SIGNIFICANT" if p_value < 0.05 else "not significant"
    print(f"{feature}: p-value = {p_value:.4f} → {significance}")

# ── Plot 4: P-value Visualization ──
results = []
for feature in features_test:
    group1 = df[df['Outcome'] == 1][feature].dropna()
    group2 = df[df['Outcome'] == 0][feature].dropna()
    t_stat, p_value = stats.ttest_ind(group1, group2)
    results.append({'Feature': feature, 'p-value': p_value})

results_df = pd.DataFrame(results)

plt.figure(figsize=(10, 5))
colors = ['red' if p < 0.05 else 'gray' for p in results_df['p-value']]
plt.barh(results_df['Feature'], -np.log10(results_df['p-value']), color=colors)
plt.axvline(x=-np.log10(0.05), color='black', linestyle='--', label='p = 0.05')
plt.xlabel('-log10(p-value)')
plt.title('Statistical Significance of Diabetes Features')
plt.legend()
plt.tight_layout()
plt.savefig('pvalue_plot.png')
print("P-value plot saved!")

# ── Plot 5: PCA Analysis ──
pca_features = ['Glucose', 'BMI', 'Age', 'Insulin',
                'BloodPressure', 'DiabetesPedigreeFunction']

X_pca = df[pca_features].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

pca = PCA(n_components=2)
X_pca_result = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
colors = df.loc[X_pca.index, 'Outcome'].map({0: 'blue', 1: 'red'})
plt.scatter(X_pca_result[:, 0], X_pca_result[:, 1], c=colors, alpha=0.6)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('PCA of Diabetes Clinical Features')
plt.legend(handles=[
    plt.scatter([], [], color='blue', label='No Diabetes'),
    plt.scatter([], [], color='red', label='Diabetes')
], labels=['No Diabetes', 'Diabetes'])
plt.tight_layout()
plt.savefig('pca_plot.png')
print(f"PCA plot saved!")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.1%}")