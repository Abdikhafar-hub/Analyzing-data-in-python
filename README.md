# Analyzing-data-in-python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset

# Load the Iris dataset from sklearn
try:
    iris_data = load_iris(as_frame=True)
    df = iris_data.frame
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Display the first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Check data types and missing values
print("\nDataset Information:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# Clean the dataset (no missing values in this dataset, but demonstrating filling)
df.fillna(df.mean(), inplace=True)  # Replace NaNs with mean of each column

# Task 2: Basic Data Analysis

# Compute basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Group by species and compute mean of numerical columns
grouped_means = df.groupby('target').mean()
print("\nMean values grouped by target (species):")
print(grouped_means)

# Patterns or interesting findings
print("\nPatterns:")
print("Sepal length and width vary significantly between species.")

# Task 3: Data Visualization

# Line Chart: Simulating a trend for illustration (use first column as a dummy time-series)
plt.figure(figsize=(10, 6))
plt.plot(df.index[:50], df["sepal length (cm)"][:50], label="Sepal Length")
plt.plot(df.index[:50], df["petal length (cm)"][:50], label="Petal Length")
plt.title("Trend of Sepal and Petal Length (First 50 Samples)")
plt.xlabel("Index")
plt.ylabel("Length (cm)")
plt.legend()
plt.show()

# Bar Chart: Average petal length by species
plt.figure(figsize=(10, 6))
sns.barplot(x=grouped_means.index, y=grouped_means['petal length (cm)'], palette='viridis')
plt.title("Average Petal Length by Species")
plt.xlabel("Species (Target)")
plt.ylabel("Petal Length (cm)")
plt.show()

# Histogram: Distribution of Sepal Length
plt.figure(figsize=(10, 6))
plt.hist(df['sepal length (cm)'], bins=15, color='skyblue', edgecolor='black')
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# Scatter Plot: Sepal length vs. Petal length
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="sepal length (cm)", y="petal length (cm)", hue="target", palette="deep")
plt.title("Relationship Between Sepal Length and Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species (Target)")
plt.show()

