import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

raw_table = np.array(pd.read_csv("dataset/berkeley_processed.csv", header=None))
corr_matrix = np.corrcoef(raw_table, rowvar=False)
# Print the DataFrame
# print(corr_matrix)

# Create a heatmap
plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
plt.title('Correlation Matrix')
plt.show()