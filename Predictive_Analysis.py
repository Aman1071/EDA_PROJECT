import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_excel("/content/ICT_Subdimension_Dataset.xlsx")

# Define key indicators
key_indicators = [
    'Household Internet Access (%)',
    'Fixed Broadband Subscriptions (%)',
    'Wireless Broadband Subscriptions (%)',
    'Wireless Broadband Coverage 4G (%)',
    'Smart Electricity Meters (%)'
]

# Compute ICT Adoption Score
df['ICT_Adoption_Score'] = df[key_indicators].mean(axis=1)

# Predict for 2023, 2024, 2025
multi_year_predictions = []
for city in df['City'].unique():
    city_data = df[df['City'] == city]
    X = city_data[['Year']]
    y = city_data['ICT_Adoption_Score']

    model = LinearRegression()
    model.fit(X, y)

    for year in [2023, 2024, 2025]:
        pred = model.predict([[year]])[0]
        multi_year_predictions.append({
            'City': city,
            'Year': year,
            'Predicted_Adoption_Score': pred
        })

multi_year_df = pd.DataFrame(multi_year_predictions)

# Pivot and compute mean
mean_scores = multi_year_df.pivot(index='City', columns='Year', values='Predicted_Adoption_Score').reset_index()
mean_scores['Mean_Adoption_Score'] = mean_scores[[2023, 2024, 2025]].mean(axis=1)

# Classify cities
def classify(score):
    if score >= 85:
        return 'High'
    elif score >= 60:
        return 'Medium'
    else:
        return 'Low'

mean_scores['Adoption_Category'] = mean_scores['Mean_Adoption_Score'].apply(classify)

# Display results
print(mean_scores[['City', 'Mean_Adoption_Score', 'Adoption_Category']].round(2))
# Prepare data for heatmap
heatmap_data = sample_df.set_index('City')[['Mean_Adoption_Score']].sort_values('Mean_Adoption_Score', ascending=False)

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='YlGnBu', linewidths=0.5, linecolor='gray')
plt.title("Sample Cities: Mean ICT Adoption Score (2023–2025)", fontsize=14)
plt.xlabel("Mean Score")
plt.ylabel("City")
plt.tight_layout()
plt.show()
