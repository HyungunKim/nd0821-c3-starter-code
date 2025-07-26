import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Try to import seaborn, but continue if not available
try:
    import seaborn as sns
    has_seaborn = True
except ImportError:
    has_seaborn = False
    print("Seaborn not found. Using matplotlib for all visualizations.")

# Create output directory for EDA results if it doesn't exist
output_dir = os.path.join("eda_results")
os.makedirs(output_dir, exist_ok=True)

# Load data
data_path = os.path.join("data", "census.csv")
data = pd.read_csv(data_path)

# Print column names to identify the correct target variable
print("Column names in the dataset:", data.columns.tolist())

# Check if there are any spaces in column names and strip them
data.columns = data.columns.str.strip()

# Print unique values in the salary column
print("Unique values in the salary column:", data['salary'].unique())

# Strip spaces from all string columns
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = data[col].str.strip()

# Print unique values in the salary column after stripping
print("Unique values in the salary column after stripping:", data['salary'].unique())

# save the clean data csv
data.to_csv(os.path.join(output_dir, "clean_data.csv"), index=False)

# Define categorical features (from train_model.py)
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Save basic dataset information to text file
with open(os.path.join(output_dir, "data_info.txt"), "w") as f:
    f.write(f"Dataset Shape: {data.shape}\n\n")
    f.write("Data Types:\n")
    f.write(str(data.dtypes) + "\n\n")
    f.write("Summary Statistics for Numerical Features:\n")
    f.write(str(data.describe()) + "\n\n")
    f.write("Missing Values:\n")
    f.write(str(data.isnull().sum()) + "\n\n")

    # Check for '?' values which often represent missing data in this dataset
    for col in data.columns:
        question_marks = (data[col] == '?').sum()
        if question_marks > 0:
            f.write(f"Column {col} has {question_marks} '?' values\n")

# Analyze target variable distribution
plt.figure(figsize=(10, 6))
salary_counts = data['salary'].value_counts()
plt.bar(salary_counts.index, salary_counts.values)
plt.title('Distribution of Salary')
plt.xlabel('Salary')
plt.ylabel('Count')
plt.savefig(os.path.join(output_dir, "salary_distribution.png"))
plt.close()

# Save target distribution to text file
with open(os.path.join(output_dir, "target_distribution.txt"), "w") as f:
    f.write("Salary Distribution:\n")
    f.write(str(salary_counts) + "\n\n")
    f.write(f"Percentage >50K: {salary_counts['>50K'] / len(data) * 100:.2f}%\n")
    f.write(f"Percentage <=50K: {salary_counts['<=50K'] / len(data) * 100:.2f}%\n")

# Analyze numerical features
numerical_features = [col for col in data.columns if col not in cat_features and col != 'salary']

# Histograms for numerical features
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features):
    plt.subplot(2, 3, i+1)
    if has_seaborn:
        sns.histplot(data[feature], kde=True)
    else:
        plt.hist(data[feature], bins=30)
        # Add a density curve
        if len(data[feature].unique()) > 10:  # Only add KDE for continuous-like variables
            from scipy import stats
            density = stats.gaussian_kde(data[feature].dropna())
            x_vals = np.linspace(data[feature].min(), data[feature].max(), 100)
            plt.plot(x_vals, density(x_vals), 'r-')
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "numerical_distributions.png"))
plt.close()

# Box plots for numerical features by salary
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features):
    plt.subplot(2, 3, i+1)
    if has_seaborn:
        sns.boxplot(x='salary', y=feature, data=data)
    else:
        # Create boxplot with matplotlib
        salary_categories = data['salary'].unique()
        box_data = [data[data['salary'] == cat][feature] for cat in salary_categories]
        plt.boxplot(box_data, labels=salary_categories)
        plt.ylabel(feature)
    plt.title(f'{feature} by Salary')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "numerical_by_salary.png"))
plt.close()

# Correlation matrix for numerical features
plt.figure(figsize=(10, 8))
numerical_data = data[numerical_features]
correlation_matrix = numerical_data.corr()
if has_seaborn:
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
else:
    # Create heatmap with matplotlib
    plt.imshow(correlation_matrix, cmap='coolwarm')
    plt.colorbar()
    # Add correlation values as text
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            plt.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}",
                     ha="center", va="center", color="black")
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title('Correlation Matrix of Numerical Features')
plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
plt.close()

# Analyze categorical features
for feature in cat_features:
    plt.figure(figsize=(12, 6))

    # Replace '?' with 'Unknown' for better visualization
    feature_data = data[feature].replace('?', 'Unknown')

    # Get value counts and sort by frequency
    value_counts = feature_data.value_counts().sort_values(ascending=False)

    # If there are too many categories, only show top 10
    if len(value_counts) > 10:
        plt.subplot(1, 2, 1)
        top_categories = value_counts.head(10)
        if has_seaborn:
            sns.barplot(x=top_categories.index, y=top_categories.values)
        else:
            plt.bar(range(len(top_categories)), top_categories.values)
            plt.xticks(range(len(top_categories)), top_categories.index)
        plt.title(f'Top 10 Categories in {feature}')
        plt.xticks(rotation=45, ha='right')
    else:
        plt.subplot(1, 2, 1)
        if has_seaborn:
            sns.barplot(x=value_counts.index, y=value_counts.values)
        else:
            plt.bar(range(len(value_counts)), value_counts.values)
            plt.xticks(range(len(value_counts)), value_counts.index)
        plt.title(f'Categories in {feature}')
        plt.xticks(rotation=45, ha='right')

    # Stacked bar chart showing relationship with salary
    plt.subplot(1, 2, 2)

    # For features with many categories, only show top 10
    if len(value_counts) > 10:
        top_categories = value_counts.head(10).index
        filtered_data = data[data[feature].isin(top_categories)]
    else:
        filtered_data = data

    # Calculate percentage of >50K for each category
    salary_by_category = pd.crosstab(filtered_data[feature], filtered_data['salary'], normalize='index') * 100
    salary_by_category.plot(kind='bar', stacked=True)
    plt.title(f'Salary Distribution by {feature}')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{feature}_analysis.png"))
    plt.close()

    # Save categorical feature statistics to text file
    with open(os.path.join(output_dir, f"{feature}_stats.txt"), "w") as f:
        f.write(f"{feature} Value Counts:\n")
        f.write(str(data[feature].value_counts()) + "\n\n")
        f.write(f"{feature} Value Counts (Percentage):\n")
        f.write(str(data[feature].value_counts(normalize=True) * 100) + "\n\n")

        # Cross-tabulation with salary
        f.write(f"{feature} vs Salary (Counts):\n")
        f.write(str(pd.crosstab(data[feature], data['salary'])) + "\n\n")
        f.write(f"{feature} vs Salary (Row Percentages):\n")
        f.write(str(pd.crosstab(data[feature], data['salary'], normalize='index') * 100) + "\n\n")

# Feature importance for predicting salary (using a simple model)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Prepare data for feature importance
X = data.drop('salary', axis=1)
y = (data['salary'] == '>50K').astype(int)  # Convert to binary

# Encode categorical features
for feature in cat_features:
    le = LabelEncoder()
    X[feature] = le.fit_transform(X[feature].replace('?', 'Unknown'))

# Train a simple model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Plot feature importance
plt.figure(figsize=(12, 8))
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

if has_seaborn:
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
else:
    # Create horizontal bar plot with matplotlib
    plt.barh(range(len(feature_importance)), feature_importance['Importance'])
    plt.yticks(range(len(feature_importance)), feature_importance['Feature'])
plt.title('Feature Importance for Predicting Salary')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_importance.png"))
plt.close()

# Save feature importance to text file
with open(os.path.join(output_dir, "feature_importance.txt"), "w") as f:
    f.write("Feature Importance for Predicting Salary:\n")
    f.write(str(feature_importance) + "\n")

print("EDA completed. Results saved to:", output_dir)
