import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

df = pd.read_csv("ipl.csv")


print(df.shape)
print(df.columns)
print(df.info())
print(df.isnull().sum())
if "umpire3" in df.columns:
    df.drop("umpire3", axis=1, inplace=True)

df.fillna(df.mean(numeric_only=True), inplace=True)

for col in df.select_dtypes(include="object").columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nStatistical Summary:")
print(df.describe())

print("\nCategorical Summary:")
print(df.describe(include="object"))

sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Values Heatmap")
plt.show()

if "winner" in df.columns:
    plt.figure(figsize=(12,6))
    df["winner"].value_counts().plot(kind="bar")
    plt.title("Matches Won by Teams")
    plt.ylabel("Wins")
    plt.show()

if "toss_winner" in df.columns:
    plt.figure(figsize=(12,6))
    df["toss_winner"].value_counts().plot(kind="bar")
    plt.title("Toss Wins by Teams")
    plt.ylabel("Toss Wins")
    plt.show()

if "player_of_match" in df.columns:
    plt.figure(figsize=(12,6))
    df["player_of_match"].value_counts().head(10).plot(kind="bar")
    plt.title("Top 10 Player of the Match Awards")
    plt.show()

if "city" in df.columns:
    plt.figure(figsize=(12,6))
    df["city"].value_counts().head(10).plot(kind="bar")
    plt.title("Top Match Cities")
    plt.show()

if "winner" in df.columns:
    text = " ".join(df["winner"].astype(str))
    wc = WordCloud(width=1000, height=500, background_color="white").generate(text)

    plt.figure(figsize=(12,6))
    plt.imshow(wc)
    plt.axis("off")
    plt.title("Winning Teams WordCloud")
    plt.show()

numeric_cols = df.select_dtypes(include=np.number)

if len(numeric_cols.columns) > 1:
    plt.figure(figsize=(10,8))
    sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()

for col in numeric_cols.columns:
    plt.figure(figsize=(8,4))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

for col in numeric_cols.columns:
    plt.figure(figsize=(8,4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()