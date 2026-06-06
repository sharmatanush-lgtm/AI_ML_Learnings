# ==========================================
# AI STYLE NETFLIX INSIGHT GENERATOR
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------
# LOAD DATA
# ------------------------------------------

df = pd.read_csv("netflix_titles.csv")

print("\nDataset Shape:", df.shape)

# ------------------------------------------
# BASIC CLEANING
# ------------------------------------------

df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')

df['year_added'] = df['date_added'].dt.year

# ------------------------------------------
# STORE INSIGHTS
# ------------------------------------------

insights = []

# ==========================================
# 1. DATASET OVERVIEW
# ==========================================

rows, cols = df.shape

insights.append(
    f"The dataset contains {rows} rows and {cols} columns."
)

# ==========================================
# 2. MISSING VALUE ANALYSIS
# ==========================================

missing = df.isnull().sum()

for col, val in missing.items():

    percent = (val / len(df)) * 100

    if percent > 20:
        insights.append(
            f"Column '{col}' has very high missing values ({percent:.1f}%)."
        )

    elif percent > 0:
        insights.append(
            f"Column '{col}' has {percent:.1f}% missing values."
        )

# ==========================================
# 3. AUTOMATIC CATEGORICAL INSIGHTS
# ==========================================

categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:

    if df[col].nunique() < 50:

        top = df[col].value_counts().idxmax()

        top_count = df[col].value_counts().max()

        percent = (top_count / len(df)) * 100

        if percent > 50:
            insights.append(
                f"In column '{col}', majority of values are '{top}' ({percent:.1f}%)."
            )

        elif percent > 25:
            insights.append(
                f"'{top}' is the most common value in '{col}' ({percent:.1f}%)."
            )

# ==========================================
# 4. CONTENT TYPE ANALYSIS
# ==========================================

if 'type' in df.columns:

    counts = df['type'].value_counts()

    top_type = counts.idxmax()

    percent = (counts.max() / counts.sum()) * 100

    insights.append(
        f"Netflix mainly focuses on '{top_type}' content ({percent:.1f}% of catalog)."
    )

# ==========================================
# 5. COUNTRY ANALYSIS
# ==========================================

if 'country' in df.columns:

    countries = df['country'].dropna().str.split(', ').explode()

    top_country = countries.value_counts().idxmax()

    percent = (
        countries.value_counts().max() /
        len(countries)
    ) * 100

    insights.append(
        f"'{top_country}' produces the highest Netflix content ({percent:.1f}% contribution)."
    )

# ==========================================
# 6. YEAR TREND ANALYSIS
# ==========================================

if 'release_year' in df.columns:

    yearly = df['release_year'].value_counts().sort_index()

    growth = yearly.pct_change() * 100

    high_growth_years = growth[growth > 50]

    if not high_growth_years.empty:

        year = high_growth_years.idxmax()

        value = high_growth_years.max()

        insights.append(
            f"Netflix content saw explosive growth around {year} ({value:.1f}% increase)."
        )

    peak_year = yearly.idxmax()

    insights.append(
        f"Maximum content was released in {peak_year}."
    )

# ==========================================
# 7. RATING ANALYSIS
# ==========================================

if 'rating' in df.columns:

    top_rating = df['rating'].value_counts().idxmax()

    percent = (
        df['rating'].value_counts().max()
        / len(df)
    ) * 100

    insights.append(
        f"Most Netflix content is rated '{top_rating}' ({percent:.1f}%)."
    )

# ==========================================
# 8. GENRE ANALYSIS
# ==========================================

if 'listed_in' in df.columns:

    genres = df['listed_in'].dropna().str.split(', ').explode()

    top_genre = genres.value_counts().idxmax()

    insights.append(
        f"The most popular genre on Netflix is '{top_genre}'."
    )

# ==========================================
# 9. OUTLIER DETECTION
# ==========================================

numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:

    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = df[
        (df[col] < lower) |
        (df[col] > upper)
    ]

    percent = (len(outliers) / len(df)) * 100

    if percent > 5:

        insights.append(
            f"Column '{col}' contains significant outliers ({percent:.1f}%)."
        )

# ==========================================
# 10. DIVERSITY ANALYSIS
# ==========================================

for col in categorical_cols:

    unique_ratio = df[col].nunique() / len(df)

    if unique_ratio > 0.5:

        insights.append(
            f"Column '{col}' has very high diversity with many unique values."
        )

# ==========================================
# 11. CORRELATION ANALYSIS
# ==========================================

if len(numeric_cols) > 1:

    corr = df[numeric_cols].corr()

    for i in range(len(corr.columns)):
        for j in range(i):

            val = corr.iloc[i, j]

            if abs(val) > 0.7:

                insights.append(
                    f"Strong correlation detected between "
                    f"'{corr.columns[i]}' and "
                    f"'{corr.columns[j]}' "
                    f"(correlation={val:.2f})."
                )

# ==========================================
# PRINT ALL INSIGHTS
# ==========================================

print("\n")
print("="*60)
print("AI GENERATED INSIGHTS")
print("="*60)

for i, insight in enumerate(insights, 1):
    print(f"{i}. {insight}")

# ==========================================
# OPTIONAL VISUALIZATION
# ==========================================

plt.figure(figsize=(8,5))

sns.countplot(
    x='type',
    data=df
)

plt.title("Netflix Content Type Distribution")

plt.show()