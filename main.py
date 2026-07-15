import pandas as pd

movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

print("Movies:")
print(movies.head())

print("\nRatings:")
print(ratings.head())

#cek ukuran data
print("\nJumlah data movies:", movies.shape)
print("Jumlah data ratings:", ratings.shape)

#cek info data
print("\nInfo Movies:")
print(movies.info())

print("\nInfo Ratings:")
print(ratings.info())

#cek data kosong
print("\nMissing values Movies:")
print(movies.isnull().sum())

print("\nMissing values Ratings:")
print(ratings.isnull().sum())

#cek distribusi rating
print("\nStatistik Rating:")
print(ratings['rating'].describe())