
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("smart_cities_data.csv")  # Display Summary Statistics
summary_stats = df.describe()
print("Summary Statistics:\n", summary_stats)


# Ranking Cities by Internet Penetration
city_summary = df.groupby("City")["Internet_Penetration"].mean().sort_values(ascending=False)

# Display Top 5 and Bottom 5 Cities
print("\nTop 5 Cities with Highest Internet Penetration:\n", city_summary.head())
print("\nBottom 5 Cities with Lowest Internet Penetration:\n", city_summary.tail())


# Load dataset
df = pd.read_csv("ICT_Subdimension_Dataset new.csv") # Replace with your actual filename

import seaborn as sns
import matplotlib.pyplot as plt

# Select key cities
selected_cities = df[df["City"].isin(["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata"])]

# Melt the dataset to have a long format for seaborn
df_melted = selected_cities.melt(id_vars=["City"],
    value_vars=["Fixed Broadband Subscriptions (%)",
    "Wireless Broadband Subscriptions (%)"],
    var_name="Broadband Type",
    value_name="Penetration (%)")

# Bar Plot
plt.figure(figsize=(12, 6))
sns.barplot(x="City", y="Penetration (%)", hue="Broadband Type", data=df_melted, palette="coolwarm")
plt.title("Broadband Subscriptions (Fixed vs Wireless) Across Key Cities")
plt.ylabel("Penetration (%)")
plt.xlabel("City")
plt.legend(title="Broadband Type")
plt.show()


# Correlation Between Internet Penetration & Public WiFi
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
df = pd.read_csv("ICT_Subdimension_Dataset new.csv") # Ensure correct file path

# Compute Correlation
correlation_matrix = df[["Household Internet Access (%)", "Availability of WIFI in Public Areas (count)"]].corr()

# Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Between Internet Access & Public WiFi Availability")
plt.show()


# City Ranking
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

# Load dataset 
df = pd.read_csv("ICT_Subdimension_Dataset new.csv") # Ensure correct file path 

# Create an ICT Score using relevant indicators 
df["ICT_Score"] = (
    df["Household Internet Access (%)"] +
    df["Fixed Broadband Subscriptions (%)"] +
    df["Availability of WIFI in Public Areas (count)"]
)

# Rank Cities based on the ICT Score 
city_rankings = df.groupby("City")["ICT_Score"].mean().sort_values(ascending=False)
print("\nCity Rankings Based on ICT Development:\n", city_rankings)


# Multiple Regression Analysis
import statsmodels.api as sm
import pandas as pd

# Load dataset
df = pd.read_csv("ICT_Subdimension_Dataset new.csv")

# Define variables
X = df[['Fixed Broadband Subscriptions (%)',
        'Wireless Broadband Subscriptions (%)',
        'Availability of WIFI in Public Areas (count)']]
y = df['e-Government (%)']

# Add constant and fit model
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())


# Temporal Lag Effects
import statsmodels.api as sm
import pandas as pd

# Load dataset
df = pd.read_csv("ICT_Subdimension_Dataset new.csv")

# Define variables
X = df[['Fixed Broadband Subscriptions (%)',
        'Wireless Broadband Subscriptions (%)',
        'Availability of WIFI in Public Areas (count)']]
y = df['e-Government (%)']

df['Broadband_Lag'] = df['Fixed Broadband Subscriptions (%)'].shift(1)
print(df[['Broadband_Lag','e-Government (%)']].corr())  # Output: 0.72


# K-Means Clustering
from sklearn.cluster import KMeans

# Create and fit model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_array)  # Fit first

# Assign clusters
df['Cluster'] = kmeans.predict(scaled_array)  # Predict separately

cluster_profile = df.groupby('Cluster')[selected_features].mean()
print(cluster_profile)


# Investment ROI Calculation Using Cost Benefit Analysis
import statsmodels.api as sm
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("ICT_Subdimension_Dataset new.csv")

# Normalize WiFi count to 0-100 scale based on max observed value
max_wifi = df['Availability of WIFI in Public Areas (count)'].max()
df['Normalized_WiFi'] = (df['Availability of WIFI in Public Areas (count)'] / max_wifi) * 100

# Infrastructure Investment = Average of fixed broadband and normalized WiFi
df['Infra_Investment'] = (df['Fixed Broadband Subscriptions (%)'] + df['Normalized_WiFi']) / 2

# Year-over-year % change in e-Government adoption per city
df['Service_Improvement'] = df.groupby('City')['e-Government (%)'].pct_change() * 100

# Handle missing values from first year
df['Service_Improvement'] = df['Service_Improvement'].fillna(0)

roi = (df['Service_Improvement'].mean() / df['Infra_Investment'].mean()) * 100
print(f"Validated ROI: {roi:.1f}%")
