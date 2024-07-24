import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create DataFrame
data = {
    'Year': [2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'Sales': [150, 200, 180, 220, 250, 300, 222],
    'Expenses': [120, 140, 160, 180, 190, 230, 200]  # Assumed value for 2024
}
df = pd.DataFrame(data)

# 1. Distribution of variables
plt.figure(figsize=(12, 4))

plt.subplot(121)
df['Sales'].hist(bins=10)
plt.title('Distribution of Sales')
plt.xlabel('Sales')
plt.ylabel('Frequency')

plt.subplot(122)
df['Expenses'].hist(bins=10)
plt.title('Distribution of Expenses')
plt.xlabel('Expenses')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# 2. Box plots to show distribution and potential outliers
plt.figure(figsize=(12, 5))
sns.boxplot(data=df[['Sales', 'Expenses']])
plt.title('Box Plot of Sales and Expenses')
plt.show()

# 3. Line plot to show trends over years
plt.figure(figsize=(10, 6))
plt.plot(df['Year'], df['Sales'], marker='o', label='Sales')
plt.plot(df['Year'], df['Expenses'], marker='o', label='Expenses')
plt.title('Sales and Expenses Over Years')
plt.xlabel('Year')
plt.ylabel('Amount')
plt.legend()
plt.grid(True)
plt.show()

# 4. Scatter plot to check for correlation
plt.figure(figsize=(8, 6))
plt.scatter(df['Sales'], df['Expenses'])
plt.title('Sales vs Expenses')
plt.xlabel('Sales')
plt.ylabel('Expenses')
plt.grid(True)
plt.show()

# 5. Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[['Sales', 'Expenses']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
