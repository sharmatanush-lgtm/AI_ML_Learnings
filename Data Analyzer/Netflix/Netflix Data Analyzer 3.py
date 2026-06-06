# ==========================================
# UNIVERSAL AI DATASET INSIGHT ENGINE
# Works on ANY CSV Dataset
# ==========================================

import pandas as pd
import numpy as np

# ------------------------------------------
# LOAD DATASET
# ------------------------------------------

file_path = "your_dataset.csv"

df = pd.read_csv(file_path)

# ------------------------------------------
# BASIC INFO
# ------------------------------------------

rows, cols = df.shape

print("\n==============================")
print("DATASET OVERVIEW")
print("==============================")

print(f"Rows    : {rows}")
print(f"Columns : {cols}")

# ------------------------------------------
# STORE INSIGHTS
# ------------------------------------------

insights = []

# ==========================================
# 1. MISSING VALUE ANALYSIS
# ==========================================

missing = df.isnull().sum()

for col in df.columns:

    missing_percent = (
        missing[col] / rows
    ) * 100

    if missing_percent > 0:

        insights.append(
            f"Column '{col}' has "
            f"{missing_percent:.1f}% missing values."
        )

# ==========================================
# 2. COLUMN TYPE ANALYSIS
# ==========================================

for col in df.columns:

    dtype = df[col].dtype

    insights.append(
        f"Column '{col}' is of type '{dtype}'."
    )

# ==========================================
# 3. CATEGORICAL COLUMN INSIGHTS
# ==========================================

categorical_cols = df.select_dtypes(
    include=['object']
).columns

for col in categorical_cols:

    unique_count = df[col].nunique()

    # Skip extremely unique columns
    if unique_count < rows * 0.5:

        top_value = df[col].mode()[0]

        top_count = (
            df[col] == top_value
        ).sum()

        percent = (
            top_count / rows
        ) * 100

        insights.append(
            f"In column '{col}', "
            f"'{top_value}' appears most frequently "
            f"({percent:.1f}% of rows)."
        )

        # Diversity analysis
        diversity = unique_count / rows

        if diversity < 0.01:

            insights.append(
                f"Column '{col}' has low diversity "
                f"with only {unique_count} unique values."
            )

# ==========================================
# 4. NUMERIC COLUMN ANALYSIS
# ==========================================

numeric_cols = df.select_dtypes(
    include=np.number
).columns

for col in numeric_cols:

    mean = df[col].mean()
    median = df[col].median()
    std = df[col].std()

    insights.append(
        f"Column '{col}' has average "
        f"value {mean:.2f}."
    )

    # Skew detection
    if mean > median * 1.5:

        insights.append(
            f"Column '{col}' appears positively skewed."
        )

    elif median > mean * 1.5:

        insights.append(
            f"Column '{col}' appears negatively skewed."
        )

    # Outlier detection
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = df[
        (df[col] < lower) |
        (df[col] > upper)
    ]

    outlier_percent = (
        len(outliers) / rows
    ) * 100

    if outlier_percent > 5:

        insights.append(
            f"Column '{col}' contains "
            f"{outlier_percent:.1f}% outliers."
        )

# ==========================================
# 5. CORRELATION ANALYSIS
# ==========================================

if len(numeric_cols) > 1:

    corr_matrix = df[numeric_cols].corr()

    for i in range(len(corr_matrix.columns)):

        for j in range(i):

            corr = corr_matrix.iloc[i, j]

            if abs(corr) > 0.7:

                insights.append(
                    f"Strong correlation detected between "
                    f"'{corr_matrix.columns[i]}' and "
                    f"'{corr_matrix.columns[j]}' "
                    f"(correlation={corr:.2f})."
                )

# ==========================================
# 6. TIME TREND ANALYSIS
# ==========================================

for col in df.columns:

    # Detect datetime columns automatically
    try:

        converted = pd.to_datetime(
            df[col],
            errors='coerce'
        )

        valid_dates = converted.notnull().sum()

        # If enough dates exist
        if valid_dates > rows * 0.5:

            yearly = converted.dt.year.value_counts()

            if len(yearly) > 2:

                peak_year = yearly.idxmax()

                insights.append(
                    f"Most records in '{col}' belong "
                    f"to year {peak_year}."
                )

    except:
        pass

# ==========================================
# PRINT ALL INSIGHTS
# ==========================================

print("\n")
print("="*60)
print("AUTO-GENERATED AI INSIGHTS")
print("="*60)

for i, insight in enumerate(insights, 1):

    print(f"{i}. {insight}")