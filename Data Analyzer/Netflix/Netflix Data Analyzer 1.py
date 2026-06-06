# ==========================================
# NETFLIX DATA ANALYSIS PROJECT
# ==========================================
# Install required libraries first:
# pip install pandas matplotlib seaborn wordcloud

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


# ------------------------------------------
# LOAD DATA
# ------------------------------------------

# Replace with your file name
df = pd.read_csv("netflix_titles.csv")

print("\n==============================")
print("FIRST 5 ROWS")
print("==============================")
print(df.head())

print("\n==============================")
print("DATA INFO")
print("==============================")
print(df.info())

print("\n==============================")
print("MISSING VALUES")
print("==============================")
print(df.isnull().sum())

# ------------------------------------------
# DATA CLEANING
# ------------------------------------------

# Convert date_added into datetime
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')

# Extract year and month
df['year_added'] = df['date_added'].dt.year
df['month_added'] = df['date_added'].dt.month_name()

# Fill missing country values
df['country'] = df['country'].fillna("Unknown")

# ------------------------------------------
# INSIGHT 1
# Netflix has more Movies than TV Shows
# ------------------------------------------

type_counts = df['type'].value_counts()

print("\n==============================")
print("MOVIES VS TV SHOWS")
print("==============================")
print(type_counts)

plt.figure(figsize=(6,4))
sns.countplot(x='type', data=df)
plt.title("Movies vs TV Shows on Netflix")
plt.show()

# ------------------------------------------
# INSIGHT 2
# Top Countries Producing Content
# ------------------------------------------

country_counts = df['country'].str.split(', ').explode().value_counts().head(10)

print("\n==============================")
print("TOP 10 COUNTRIES")
print("==============================")
print(country_counts)

plt.figure(figsize=(10,5))
country_counts.plot(kind='bar')
plt.title("Top 10 Countries Producing Netflix Content")
plt.ylabel("Number of Titles")
plt.show()

# ------------------------------------------
# INSIGHT 3
# Content Growth Over Years
# ------------------------------------------

year_counts = df['release_year'].value_counts().sort_index()

plt.figure(figsize=(12,5))
year_counts.plot()
plt.title("Netflix Content Growth by Release Year")
plt.xlabel("Year")
plt.ylabel("Number of Titles")
plt.show()

print("\nObservation:")
print("Content increased sharply after 2010.")

# ------------------------------------------
# INSIGHT 4
# Which Rating is Most Common?
# ------------------------------------------

rating_counts = df['rating'].value_counts()

print("\n==============================")
print("CONTENT RATINGS")
print("==============================")
print(rating_counts)

plt.figure(figsize=(10,5))
sns.countplot(y='rating', data=df,
              order=df['rating'].value_counts().index)
plt.title("Most Common Ratings")
plt.show()

# ------------------------------------------
# INSIGHT 5
# Most Common Genres
# ------------------------------------------

genres = df['listed_in'].str.split(', ').explode()

genre_counts = genres.value_counts().head(10)

print("\n==============================")
print("TOP GENRES")
print("==============================")
print(genre_counts)

plt.figure(figsize=(10,5))
genre_counts.plot(kind='bar')
plt.title("Top Genres on Netflix")
plt.show()

# ------------------------------------------
# INSIGHT 6
# Which Year Netflix Added Most Content?
# ------------------------------------------

added_year_counts = df['year_added'].value_counts().sort_index()

plt.figure(figsize=(12,5))
added_year_counts.plot(kind='line', marker='o')
plt.title("Content Added to Netflix Each Year")
plt.xlabel("Year")
plt.ylabel("Titles Added")
plt.show()

# ------------------------------------------
# INSIGHT 7
# Monthly Trend
# ------------------------------------------

month_order = [
    'January', 'February', 'March', 'April',
    'May', 'June', 'July', 'August',
    'September', 'October', 'November', 'December'
]

month_counts = df['month_added'].value_counts().reindex(month_order)

plt.figure(figsize=(12,5))
month_counts.plot(kind='bar')
plt.title("Monthly Content Additions")
plt.ylabel("Titles Added")
plt.show()

# ------------------------------------------
# INSIGHT 8
# Top Directors
# ------------------------------------------

directors = df['director'].dropna().str.split(', ').explode()

director_counts = directors.value_counts().head(10)

print("\n==============================")
print("TOP DIRECTORS")
print("==============================")
print(director_counts)

plt.figure(figsize=(10,5))
director_counts.plot(kind='bar')
plt.title("Top Directors on Netflix")
plt.show()

# ------------------------------------------
# INSIGHT 9
# Duration Analysis
# ------------------------------------------

movies = df[df['type'] == 'Movie']

movies['duration_int'] = movies['duration'].str.extract('(\d+)').astype(float)

plt.figure(figsize=(10,5))
sns.histplot(movies['duration_int'], bins=30)
plt.title("Movie Duration Distribution")
plt.xlabel("Minutes")
plt.show()

# ------------------------------------------
# INSIGHT 10
# WordCloud of Genres
# ------------------------------------------

text = " ".join(genres.dropna())

wordcloud = WordCloud(width=1000,
                      height=500,
                      background_color='white').generate(text)

plt.figure(figsize=(15,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Popular Genres WordCloud")
plt.show()

# ------------------------------------------
# FINAL AUTOMATIC INSIGHTS
# ------------------------------------------

print("\n====================================")
print("KEY INSIGHTS FROM NETFLIX DATA")
print("====================================")

# Insight 1
if type_counts['Movie'] > type_counts['TV Show']:
    print("1. Netflix has more Movies than TV Shows.")

# Insight 2
top_country = country_counts.index[0]
print(f"2. {top_country} produces the highest Netflix content.")

# Insight 3
print("3. Netflix content increased rapidly after 2010.")

# Insight 4
top_genre = genre_counts.index[0]
print(f"4. Most common genre is '{top_genre}'.")

# Insight 5
top_rating = rating_counts.index[0]
print(f"5. Most common rating is '{top_rating}'.")

# Insight 6
top_year = added_year_counts.idxmax()
print(f"6. Netflix added most content in {int(top_year)}.")

# Insight 7
top_month = month_counts.idxmax()
print(f"7. Netflix adds most content in {top_month}.")

# Insight 8
top_director = director_counts.index[0]
print(f"8. Most featured director is {top_director}.")

print("\nAnalysis Completed Successfully!")