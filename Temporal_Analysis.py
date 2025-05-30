
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load the dataset
file_path = "/content/ICT_Subdimension_Dataset.xlsx"
df = pd.read_excel(file_path, sheet_name='ICT_Subdimension_Dataset')

# List of metric columns (excluding City and Year)
metric_columns = df.columns[2:]

# Compute annual growth rates
def calculate_growth_rates(group):
    group = group.sort_values(by='Year')
    for col in metric_columns:
        group[f'{col} Growth Rate'] = group[col].pct_change() * 100
    return group.reset_index(drop=True)

df_growth = df.groupby('City', group_keys=False).apply(calculate_growth_rates).reset_index(drop=True)

# Forecasting future values (Next 3 years)
def forecast_future_values(group):
    future_years = [2024, 2025, 2026]
    forecast_df = pd.DataFrame({'City': group['City'].iloc[0], 'Year': future_years})

    for col in metric_columns:
        model = ExponentialSmoothing(group[col], trend='add', seasonal=None, damped_trend=True).fit()
        forecast_df[col] = model.forecast(steps=3).values

    return forecast_df

forecasted_data = df.groupby('City', group_keys=False).apply(forecast_future_values).reset_index(drop=True)

# Compute cumulative (average) growth rate for each feature
cumulative_growth = df_growth[[f'{col} Growth Rate' for col in metric_columns]].mean()

# Compute average forecasted values for all cities
average_forecasted_values = forecasted_data[metric_columns].mean()

# Finding top 3 cities based on highest growth rates
best_growth_cities = {}
for col in metric_columns:
    growth_col = f'{col} Growth Rate'
    if growth_col in df_growth.columns:
        best_growth_cities[col] = df_growth.groupby('City', group_keys=False)[growth_col].mean().nlargest(3).index.tolist()

# Finding cities with highest predicted values
best_forecasted_cities = {}
for col in metric_columns:
    best_forecasted_cities[col] = forecasted_data.groupby('City', group_keys=False)[col].mean().idxmax()

# Output results
growth_rates_summary = df_growth[['City', 'Year'] + [f'{col} Growth Rate' for col in metric_columns]]
forecast_summary = forecasted_data

print("Top 3 Cities Based on Growth Rates:")
print(pd.DataFrame(best_growth_cities))
print("\nCities with Highest Predicted Values:")
print(pd.DataFrame.from_dict(best_forecasted_cities, orient='index', columns=['Best City']))

# Save results to Excel
with pd.ExcelWriter("/content/City_Growth_Analysis.xlsx") as writer:
    growth_rates_summary.to_excel(writer, sheet_name="Growth Rates", index=False)
    forecast_summary.to_excel(writer, sheet_name="Forecast", index=False)
    cumulative_growth.to_excel(writer, sheet_name="Cumulative Growth")
    average_forecasted_values.to_excel(writer, sheet_name="Avg Forecast")

# Choose a few important features for visualization
important_features = [
    'Household Internet Access (%)',
    'Fixed Broadband Subscriptions (%)',
    'Wireless Broadband Subscriptions (%)',
    'Wireless Broadband Coverage 4G (%)',
    'Availability of WIFI in Public Areas (count)'
]

# Select 4-5 main cities based on highest average internet access (custom logic can be adjusted)
top_cities = df.groupby('City')['Household Internet Access (%)'].mean().nlargest(5).index.tolist()

# Filter data for only top cities
df_main = df[df['City'].isin(top_cities)]
df_growth_main = df_growth[df_growth['City'].isin(top_cities)]
forecasted_data_main = forecasted_data[forecasted_data['City'].isin(top_cities)]

sns.set(style="whitegrid")

# Trend Analysis
for col in metric_columns:
    plt.figure(figsize=(12, 6))

    if 'Coverage' in col or 'Access' in col or 'Penetration' in col:
        for city in top_cities:
            city_data = df_main[df_main['City'] == city]
            plt.plot(city_data['Year'], city_data[col], label=city, marker='o')
            plt.fill_between(city_data['Year'], city_data[col], alpha=0.3)
        plt.title(f"Trend Line Plot with Shaded Area of {col}")
        plt.ylabel(col)
        plt.xlabel("Year")

    elif 'Subscription' in col or 'Subscribers' in col:
        sns.barplot(data=df_main, x='Year', y=col, hue='City', errorbar=None)
        plt.title(f"Bar Plot of {col}")
        plt.ylabel(col)
        plt.xlabel("Year")

    elif 'WIFI' in col or 'WiFi' in col:
        sns.boxplot(data=df_main, x='Year', y=col)
        plt.title(f"Box Plot of {col}")
        plt.ylabel(col)
        plt.xlabel("Year")

    elif 'Internet' in col or 'Broadband' in col:
        sns.lineplot(data=df_main, x='Year', y=col, hue='City', marker='o')
        plt.title(f"Trend Line Plot of {col}")
        plt.ylabel(col)
        plt.xlabel("Year")

    else:
        sns.scatterplot(data=df_main, x='Year', y=col, hue='City', s=100)
        plt.title(f"Scatter Plot of {col}")
        plt.ylabel(col)
        plt.xlabel("Year")

    plt.legend(title='City', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Growth Rate Visualization
for col in important_features:
    growth_col = f'{col} Growth Rate'
    if growth_col in df_growth_main.columns:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_growth_main, x='City', y=growth_col, errorbar=None)
        plt.title(f'Annual Growth Rate for {col}')
        plt.xticks(rotation=45)
        plt.xlabel('City')
        plt.ylabel('Growth Rate (%)')
        plt.grid(axis='y')
        plt.show()

# Forecasted Values Visualization
for col in important_features:
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=forecasted_data_main, x='Year', y=col, hue='City', marker='o', linestyle='--')
    plt.title(f'Forecasted Values for {col}')
    plt.xlabel('Year')
    plt.ylabel(col)
    plt.legend(title='City', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.show()
