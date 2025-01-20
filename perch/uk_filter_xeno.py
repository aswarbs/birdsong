import pandas as pd

# Load the metadata
df = pd.read_csv('perch/train_metadata.csv')

# Define UK boundaries
uk_lat_min, uk_lat_max = 49.9, 60.9
uk_lon_min, uk_lon_max = -8.6, 1.8

# Filter for UK bird songs
uk_bird_songs = df[
    (df.latitude >= uk_lat_min) & (df.latitude <= uk_lat_max) &
    (df.longitude >= uk_lon_min) & (df.longitude <= uk_lon_max)
]

# Save the filtered dataset to a new CSV file
uk_bird_songs.to_csv('uk_bird_songs.csv', index=False)

print(f"Filtered {len(uk_bird_songs)} UK bird songs.")
