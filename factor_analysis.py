#FACTOR ANALYSIS:

import pandas as pd# for csv and data manipulation
import numpy as np#Supports numerical computations.
import matplotlib.pyplot as plt
import seaborn as sns#Used for visualizing the scree plot and factor loadings.
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
#Provides tools for Factor Analysis, including Bartlett’s test and KMO test.    

# Load data
df = pd.read_csv(r"C:\Users\arist\Downloads\ICT_Subdimension_Dataset new.csv")

# Display the first few rows
df.head()

# Check for missing values
#print(df.isnull().sum()) to check if data has any missing values if yes then we should do preprocessing


# If missing values exist, consider imputing or dropping them
# For example, to drop rows with missing values:
df.dropna(inplace=True)

# List the columns to understand the dataset
print(df.columns)

features = [
    'Household Internet Access (%)', 
    'Fixed Broadband Subscriptions (%)', 
    'Availability of WIFI in Public Areas (count)', 
    'e-Government (%)',  
    'Electricity Supply ICT Monitoring (%)'
]

df_selected = df[features]

# Bartlett's Test
bartlett_test = calculate_bartlett_sphericity(df_selected)
print(f"Bartlett's Test: {bartlett_test}")


# Check Optimal Number of Factors
fa = FactorAnalyzer(n_factors=len(features), rotation=None)
fa.fit(df_selected)
ev, _ = fa.get_eigenvalues()

# Factor Analysis with Optimal Factors
optimal_factors = sum(ev > 1)
fa = FactorAnalyzer(n_factors=optimal_factors, rotation="varimax")
fa.fit(df_selected)

# Factor Loadings
loadings = pd.DataFrame(fa.loadings_, index=features, columns=[f"Factor {i+1}" for i in range(optimal_factors)])
print("\nFactor Loadings:\n", loadings)


# Visualization of Factor Loadings
plt.figure(figsize=(10, 6))
sns.heatmap(loadings, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Factor Loadings Heatmap")
plt.show()


# Scree Plot
plt.figure(figsize=(8, 5))
plt.scatter(range(1, len(features) + 1), ev, color='red', label="Eigenvalues")
plt.plot(range(1, len(features) + 1), ev, linestyle="dashed")
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.title('Scree Plot')
plt.axhline(y=1, color='blue', linestyle='dashed')
plt.legend()    
plt.show()



