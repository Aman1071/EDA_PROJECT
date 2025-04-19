#BENCHMARKING

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"C:\Users\arist\Downloads\ICT_Subdimension_Dataset new.csv")

# Select key columns
benchmark_cols = [
    'Household Internet Access (%)',
    'Fixed Broadband Subscriptions (%)',
    'Wireless Broadband Subscriptions (%)',
    'Smart Water Meters (%)',
    'Smart Electricity Meters (%)',
    'Availability of WIFI in Public Areas (count)',
    'e-Government (%)',
    'Public Sector e-procurement (%)'
]

latest_df = df[df['Year'] == df['Year'].max()].copy()

# Normalize
scaler = MinMaxScaler()
latest_df[benchmark_cols] = scaler.fit_transform(latest_df[benchmark_cols])

# Benchmark Score (Average of all indicators)
latest_df['Benchmark Score'] = latest_df[benchmark_cols].mean(axis=1)

# Quartiles for classification
q1 = latest_df['Benchmark Score'].quantile(0.25)
q2 = latest_df['Benchmark Score'].quantile(0.50)
q3 = latest_df['Benchmark Score'].quantile(0.75)

def classify(score):
    if score >= q3:
        return ' Benchmark City'
    elif score >= q2:
        return ' Emerging Leader'
    elif score >= q1:
        return ' Mid-Level City'
    else:
        return ' Lagging City'

# Apply classification
latest_df['Category'] = latest_df['Benchmark Score'].apply(classify)

# Sort for visualization
sorted_df = latest_df.sort_values(by='Benchmark Score', ascending=True)

# Plot
plt.figure(figsize=(12, 8))
sns.barplot(
    data=sorted_df,
    y='City',
    x='Benchmark Score',
    hue='Category',
    dodge=False,
    palette='viridis'
)
plt.title('ICT Benchmark Score by City (Using Quartile Classification)', fontsize=14)
plt.xlabel('Benchmark Score')
plt.ylabel('City')
plt.legend(title='Category')
plt.tight_layout()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()
