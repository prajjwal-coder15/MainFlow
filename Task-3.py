import pandas as pd
import matplotlib.pyplot as plt

# Create a sample DataFrame
data = {
    'Year': [2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'Sales': [150, 200, 180, 220, 250, 300, 222],
    'Expenses': [120, 140, 160, 180, 190, 230, ]
}
df = pd.DataFrame(data)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))


# Bar Chart
df.plot(x='Year', y=['Sales', 'Expenses'], kind='bar', ax=ax1)
ax1.set_title('Sales and Expenses by Year')
ax1.set_xlabel('Year')
ax1.set_ylabel('Amount (in thousands)')
ax1.legend(['Sales', 'Expenses'])

# Line Chart
df.plot(x='Year', y=['Sales', 'Expenses'], kind='line', ax=ax2, marker='o')
ax2.set_title('Sales and Expenses Trend')
ax2.set_xlabel('Year')
ax2.set_ylabel('Amount (in thousands)')
ax2.legend(['Sales', 'Expenses'])

# Adjust layout and display the charts
plt.tight_layout()
plt.show()
