import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Wczytywanie danych
df = pd.read_csv(r"C:\Users\wmusi\OneDrive\Pulpit\global_music_data\Global_Music_Streaming_Listener_Preferences.csv")

# Zmiana nazw kolumn
df.columns = ["user_ID", "age", "country", "streaming_platform", "top_genre",
              "minutes_streamed_per_day", "number_of_songs_liked", "most_played_artist",
              "subscription_type", "listening_time", "discover_weekly_engagement_%",
              "repeat_song_rate_%"]

# Zmiana object na category
columns_to_convert = ["user_ID", "country", "streaming_platform", "top_genre",
                      "most_played_artist", "subscription_type", "listening_time"]
for column in columns_to_convert:
    df[column] = df[column].astype("category")
pd.set_option("display.width", None)
print(df.head(3))
df.info()
missing_values = df.isnull().sum()
print(f"missing_values: \n{missing_values}")

# Test Spearmana-nieliniowe zależności
spearman_corr= df[["age", "number_of_songs_liked"]].corr(method="spearman")
plt.figure(figsize=(10, 6))
sns.heatmap(spearman_corr, annot= True, cmap="coolwarm", fmt=".2f")
plt.show()

# Klasteryzacja K-Means
X = df[["age", "repeat_song_rate_%"]].values
km = KMeans(n_clusters=3,
            init="random",
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)

plt.scatter(X[y_km == 0, 0],
            X[y_km == 0, 1],
            s=50, c="lightgreen",
            marker="s", edgecolor="black",
            label="cluster 1")
plt.scatter(X[y_km == 1, 0],
            X[y_km == 1, 1],
            s=50, c="orange",
            marker="o", edgecolor="black",
            label="cluster 2")
plt.scatter(X[y_km == 2, 0],
            X[y_km == 2, 1],
            s=50, c="lightblue",
            marker="v", edgecolor="black",
            label="cluster 3")
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250, marker="*",
            c="red", edgecolor="black",
            label="centroids")
plt.legend(scatterpoints=1)
plt.grid()
plt.xlabel("age")
plt.ylabel("repeat_song_rate_%")
plt.show()

# "Elbow Method" do znalezienia optymalnej liczby klastrów (k), w algorytmie.
distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.title("Elbow Method for Optimal K")
plt.show()

df["cluster"] = y_km + 1
print(df.groupby("cluster")[["age", "repeat_song_rate_%"]].mean())
print(df.groupby("cluster")[["age", "repeat_song_rate_%"]].std())

# Rozkład wieku użytkowników
plt.figure(figsize=(8, 5))
sns.histplot(df["age"], bins=20, kde=True, color="blue")
plt.xlabel("Wiek użytkowników")
plt.ylabel("Liczba użytkowników")
plt.title("Rozkład wieku użytkowników")
plt.show()

# Liczba użytkowników w poszczególnych krajach
top_10_countries=df["country"].value_counts().head(10)
plt.figure(figsize=(19,10))
plt.bar(top_10_countries.index, top_10_countries.values, color="orange", width=0.5)
plt.xlabel("Kraj", fontsize=17)
plt.ylabel("Liczba użytkowników", fontsize=17)
plt.title("Top 10 krajów z największą liczbą użytkowników", fontsize=20)
plt.xticks(rotation=45, fontsize=15)
plt.yticks(fontsize=15)
plt.ylim(0, 700)
plt.tight_layout()
plt.show()

# Histogramy dla różnych zmiennych
columns_to_hist = ["age", "minutes_streamed_per_day", "number_of_songs_liked",
                   "listening_time", "repeat_song_rate_%", "discover_weekly_engagement_%"]
plt.figure(figsize=(12, 8))
for i, col in enumerate(columns_to_hist, 1):
    plt.subplot(2, 3, i)
    plt.hist(df[col], bins=10, edgecolor="black", alpha=0.7, color="royalblue")
    plt.title(col.replace("_", " ").capitalize(), fontweight="bold")
    plt.xlabel(col.replace("_", " ").capitalize())
    plt.ylabel("Liczba użytkowników")
    if col == "listening_time":
        plt.ylim(0,1650)
    else:
        plt.ylim(0, 550)
plt.tight_layout()
plt.show()