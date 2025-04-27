import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams.update({'font.family': 'DejaVu Sans'})
sns.set(style="whitegrid")
# Load the dataset
df = pd.read_csv("air_quality.csv")
# Convert 'Start_Date' to datetime format, coercing errors to NaT
df["Start_Date"] = pd.to_datetime(df["Start_Date"], errors='coerce')
# Remove duplicate rows
df.drop_duplicates(inplace=True)
#count null values
print(df.isnull().sum())
#replace null values
df['Message']=df['Message'].fillna('No Message')
# Create a copy of the cleaned DataFrame
df_clean = df.copy()
print(df_clean.isnull().sum())
# Save the cleaned DataFrame to a new CSV file
df_clean.to_csv('df_clean.csv', index=False)
# Display summary statistics of the cleaned DataFrame
print("\nSummary statistics of the cleaned DataFrame:")
print(df_clean.describe())


# Objective 1: Seasonal Trends Over Time
no2_df = df_clean[df_clean["Name"] == "Nitrogen dioxide (NO2)"].copy()
no2_df["Start_Date"] = pd.to_datetime(no2_df["Start_Date"], errors='coerce')
no2_df[["Season", "Year"]] = no2_df["Time Period"].str.extract(r"(\w+)\s+(\d{4})")
no2_df["Year"] = no2_df["Year"].astype(float)
seasonal_trend = no2_df.groupby(["Year", "Season"])["Data Value"].median().reset_index()
# Sort seasons in logical order
season_order = ["Winter", "Spring", "Summer", "Fall"]
seasonal_trend["Season"] = pd.Categorical(seasonal_trend["Season"], categories=season_order, ordered=True)
# Plot
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=seasonal_trend,
    x="Year",
    y="Data Value",
    hue="Season",
    palette="Set2",
    marker="o")
plt.title("Seasonal Trends in NO$_2$ Levels Over the Years", fontsize=15, weight='bold')
plt.xlabel("Year", fontsize=12)
plt.ylabel("Median NO$_2$ Level (ppb)", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(title="Season")
sns.despine()
plt.tight_layout()
plt.show()

# Objective 2: Top 10 Neighborhoods by Avg NO2 with gradient coloring
top_places = no2_df.groupby("Geo Place Name")["Data Value"].mean().sort_values(ascending=False).head(10)
top_df = top_places.reset_index().rename(columns={"Data Value": "Average NO2"})
norm = plt.Normalize(top_df["Average NO2"].min(), top_df["Average NO2"].max())
color_list = [plt.cm.viridis(norm(val)) for val in top_df["Average NO2"]]
plt.figure(figsize=(12, 6))
ax = sns.barplot(
    data=top_df,
    x="Average NO2",
    y="Geo Place Name",
    color= 'blue'
)
for i, value in enumerate(top_df["Average NO2"]):
    ax.text(value + 0.5, i, f"{value:.1f}", va='center', fontsize=10)

plt.title("Top 10 Polluted Neighborhoods by Average NO$_2$ Levels", fontsize=14, weight='bold')
plt.xlabel("Average NO$_2$ Level (ppb)", fontsize=12)
plt.ylabel("Neighborhood", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

# Objective 3: Distribution of PM2.5 (Histogram)
pm25_df = df_clean[df_clean["Name"] == "Fine particles (PM 2.5)"].copy()

plt.figure(figsize=(12, 6))
sns.histplot(
    data=pm25_df,
    x="Data Value",
    bins=40,
    kde=True,
    color=plt.cm.plasma(0.6),
    edgecolor="black",
    alpha=0.8
)
plt.title("Distribution of PM$_{2.5}$ Levels", fontsize=16, weight='bold')
plt.xlabel("PM$_{2.5}$ Level (\u00b5g/m$^3$)", fontsize=13)
plt.ylabel("Frequency", fontsize=13)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# Objective 4: Correlation Heatmap Between Pollutants
pivot_df = df_clean.pivot_table(
    index=["Geo Join ID", "Start_Date"],
    columns="Name",
    values="Data Value",
    aggfunc="mean"
)

plt.figure(figsize=(16, 10))
sns.heatmap(pivot_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Between Different Pollutants", fontsize=16, weight='bold')
plt.tight_layout()
plt.show()

# Objective 5: NO2 Levels Over Time in Top 15 Polluted Neighborhoods
top15_places = no2_df.groupby("Geo Place Name")["Data Value"].mean().sort_values(ascending=False).head(15).index
no2_top_df = no2_df[no2_df["Geo Place Name"].isin(top15_places)].copy()
no2_top_df["Year-Month"] = no2_top_df["Start_Date"].dt.to_period("M").astype(str)

pivot_df = no2_top_df.pivot_table(
    index="Geo Place Name",
    columns="Year-Month",
    values="Data Value",
    aggfunc="mean"
)
pivot_df["Total"] = pivot_df.sum(axis=1)
pivot_df = pivot_df.sort_values(by="Total", ascending=False).drop(columns="Total")

plt.figure(figsize=(16, 8))
sns.heatmap(pivot_df, cmap="YlOrRd", linewidths=0.5, linecolor='white',
            cbar_kws={"label": "NO$_2$ Level (ppb)"})
plt.title("NO$_2$ Levels Over Time Across Top 15 Polluted Neighborhoods", fontsize=16, weight='bold')
plt.xlabel("Time (Year-Month)")
plt.ylabel("Neighborhood")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Objective 6: Monthly Trend of PM₂.₅ Levels Over Time (Line Chart)
df_clean["Month"] = df_clean["Start_Date"].dt.month
df_clean["Month_Name"] = df_clean["Start_Date"].dt.strftime("%b")
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
df_clean["Month_Name"] = pd.Categorical(df_clean["Month_Name"], categories=month_order, ordered=True)
# Scatter plot
plt.figure(figsize=(14, 6))
sns.scatterplot(
    data=df_clean,
    x="Month_Name",
    y="Data Value",
    hue="Name",
    alpha=0.6,
    palette="Set2"
)

plt.title("Monthly Pollutant Levels (Scatter Plot)", fontsize=15, weight='bold')
plt.xlabel("Month", fontsize=12)
plt.ylabel("Pollutant Level (ppb / µg/m³)", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(title="Pollutant", bbox_to_anchor=(1.05, 1), loc='upper left')
sns.despine()
plt.tight_layout()
plt.show()


# Objective 7: Improved Pie Chart
pollutant_counts = df_clean["Name"].value_counts()
plt.figure(figsize=(10, 8))
colors = plt.cm.tab20.colors
wedges, texts, autotexts = plt.pie(
    pollutant_counts,
    labels=None,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    pctdistance=0.85
)
# donut Chart
centre_circle = plt.Circle((0, 0), 0.60, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.legend(wedges, pollutant_counts.index, title="Pollutants", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
plt.title("Proportion of Pollutant Types Monitored", fontsize=15, weight='bold')
plt.tight_layout()
plt.show()
