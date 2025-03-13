import pandas as pd

# Wczytywanie danych
df = pd.read_csv(r"C:\Users\wmusi\OneDrive\Pulpit\global_music_data\Global_Music_Streaming_Listener_Preferences.csv")

# Zmiana nazw kolumn
df.columns = ["user_ID", "age", "country", "streaming_platform", "top_genre",
              "minutes_streamed_per_day", "number_of_songs_liked", "most_played_artist",
              "subscription_type", "listening_time", "discover_weekly_engagement_%",
              "repear_song_rate_%"]

#Zmiana object na category
columns_to_convert = ["user_ID", "country", "streaming_platform", "top_genre",
                      "most_played_artist", "subscription_type", "listening_time"]
for column in columns_to_convert:
    df[column] = df[column].astype("category")
pd.set_option("display.width", None)
print(df.head(3))
df.info()
missing_values = df.isnull().sum()
print(f"missing_values: \n{missing_values}")
