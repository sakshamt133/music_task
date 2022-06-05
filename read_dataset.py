import pandas as pd


# ============================= Read the dataset using pandas ========================
df = pd.read_csv("music.csv")

# =====================   Convert the different genre of music to int type =========================
values = df['genre'].unique()

for i, val in enumerate(values):
    df.replace(val, i, inplace=True)

