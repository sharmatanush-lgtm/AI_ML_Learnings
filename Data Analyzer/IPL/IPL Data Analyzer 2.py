import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("ipl.csv")
print("\nDataset Shape:", df.shape)
if "umpire3" in df.columns:
    df = df.drop("umpire3", axis=1)
df.fillna(df.mean(numeric_only=True), inplace=True)
df.dropna(inplace=True)
insights = []
rows, cols = df.shape
insights.append(
    f"The IPL dataset contains {rows} matches and {cols} features."
)
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

categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    if df[col].nunique() < 50:
        top = df[col].value_counts().idxmax()
        top_count = df[col].value_counts().max()
        percent = (top_count / len(df)) * 100
        if percent > 50:
            insights.append(
                f"Majority of records in '{col}' are '{top}' ({percent:.1f}%)."
            )
if 'winner' in df.columns:
    top_team = df['winner'].value_counts().idxmax()
    wins = df['winner'].value_counts().max()
    insights.append(
        f"'{top_team}' is the most successful team with {wins} wins."
    )
if 'player_of_match' in df.columns:
    top_player = df['player_of_match'].value_counts().idxmax()
    awards = df['player_of_match'].value_counts().max()
    insights.append(
        f"'{top_player}' has won the most Player of the Match awards ({awards})."
    )
if 'toss_winner' in df.columns:
    toss_team = df['toss_winner'].value_counts().idxmax()
    insights.append(
        f"'{toss_team}' has won the most tosses."
    )
if {'toss_winner', 'winner'}.issubset(df.columns):
    toss_match = (df['toss_winner'] == df['winner']).mean() * 100
    insights.append(
        f"In {toss_match:.1f}% of matches, the toss winner also won the match."
    )
if 'city' in df.columns:
    city = df['city'].value_counts().idxmax()
    matches = df['city'].value_counts().max()
    insights.append(
        f"'{city}' hosted the highest number of IPL matches ({matches})."
    )
if 'season' in df.columns:
    season_counts = df['season'].value_counts()
    peak_season = season_counts.idxmax()
    insights.append(
        f"Season '{peak_season}' recorded the highest number of matches."
    )
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

    percent = len(outliers) / len(df) * 100

    if percent > 5:

        insights.append(
            f"'{col}' contains significant outliers ({percent:.1f}% of records)."
        )
for i, insight in enumerate(insights, 1):
    print(f"{i}. {insight}")
if 'winner' in df.columns:
    plt.figure(figsize=(10,5))

    sns.countplot(
        y=df['winner'],
        order=df['winner'].value_counts().index
    )
    plt.title("Match Wins by Team")
    plt.tight_layout()
