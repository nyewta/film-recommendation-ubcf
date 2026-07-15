import streamlit as st
import pandas as pd
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Pengaturan halaman
st.set_page_config(
    page_title="Sistem Rekomendasi Film",
    page_icon="🎬",
    layout="wide"
)

# Membaca dataset
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Menghitung informasi dataset
jumlah_user = ratings["userId"].nunique()
jumlah_film = movies["movieId"].nunique()
jumlah_rating = len(ratings)
teks_user = f"{jumlah_user:,}".replace(",", ".")
teks_film = f"{jumlah_film:,}".replace(",", ".")
teks_rating = f"{jumlah_rating:,}".replace(",", ".")

# Membaca data training yang digunakan saat modeling
train_data = pd.read_csv("train_data.csv")

# Membuat matriks user-film dari data training
matriks_user_film = train_data.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)

# Nilai kosong diisi 0 untuk perhitungan Cosine Similarity
matriks_angka = matriks_user_film.fillna(0)


def buat_rekomendasi_user_baru(rating_user, k=20, jumlah_rekomendasi=10):
    # Membuat profil user baru dengan kolom film yang sama
    profil_user = pd.Series(
        0.0,
        index=matriks_angka.columns
    )

    # Mengisi rating dari slider ke profil user baru
    for movie_id, rating in rating_user.items():
        if movie_id in profil_user.index:
            profil_user.loc[movie_id] = rating

    # Menghitung kemiripan user baru dengan seluruh user training
    nilai_cosine = cosine_similarity(
        profil_user.to_numpy().reshape(1, -1),
        matriks_angka.to_numpy()
    )[0]

    similaritas_user_baru = pd.Series(
        nilai_cosine,
        index=matriks_angka.index
    )

    # Mengambil 20 neighbor dengan similarity positif tertinggi
    neighbor = similaritas_user_baru[
        similaritas_user_baru > 0
    ].sort_values(
        ascending=False
    ).head(k)

    # Jika tidak ada neighbor yang mirip, kembalikan tabel kosong
    if neighbor.empty:
        return pd.DataFrame(
            columns=[
                "movieId",
                "title",
                "genres",
                "rating_prediksi"
            ]
        )

    # Mengambil data rating dari neighbor terpilih
    data_neighbor = matriks_user_film.loc[neighbor.index]

    # Film yang sudah diberi rating oleh user baru tidak direkomendasikan lagi
    film_sudah_dinilai = set(rating_user.keys())

    hasil_prediksi = []

    for movie_id in data_neighbor.columns:
        if movie_id in film_sudah_dinilai:
            continue

        rating_neighbor = data_neighbor[movie_id].dropna()

        if rating_neighbor.empty:
            continue

        nilai_similaritas = neighbor.loc[rating_neighbor.index]

        if nilai_similaritas.sum() == 0:
            continue

        # Rumus prediksi sama seperti fungsi prediksi_rating di notebook
        prediksi = (
            nilai_similaritas * rating_neighbor
        ).sum() / nilai_similaritas.sum()

        hasil_prediksi.append([
            movie_id,
            float(prediksi)
        ])

    hasil_prediksi = pd.DataFrame(
        hasil_prediksi,
        columns=["movieId", "rating_prediksi"]
    )

    # Menggabungkan hasil prediksi dengan judul dan genre film
    hasil_prediksi = hasil_prediksi.merge(
        movies,
        on="movieId",
        how="left"
    )

    # Mengurutkan dari prediksi tertinggi
    hasil_prediksi = hasil_prediksi.sort_values(
        "rating_prediksi",
        ascending=False
    )

    return hasil_prediksi.head(jumlah_rekomendasi)

film_awal = [
    {
        "movieId": 72998,
        "judul": "Avatar",
        "tahun": "2009",
        "genre": "Action, Adventure, Sci-Fi",
        "poster": "assets/posters/avatar.jpg",
        "rating_awal": 4.5
    },
    {
        "movieId": 114180,
        "judul": "The Maze Runner",
        "tahun": "2014",
        "genre": "Action, Mystery, Sci-Fi",
        "poster": "assets/posters/maze_runner.jpg",
        "rating_awal": 4.5
    },
    {
        "movieId": 8368,
        "judul": "Harry Potter and the Prisoner of Azkaban",
        "tahun": "2004",
        "genre": "Adventure, Fantasy",
        "poster": "assets/posters/harry_azkaban.jpg",
        "rating_awal": 4.5
    },
    {
        "movieId": 1246,
        "judul": "Dead Poets Society",
        "tahun": "1989",
        "genre": "Drama",
        "poster": "assets/posters/dead_poets.jpg",
        "rating_awal": 4.5
    },
    {
        "movieId": 103688,
        "judul": "The Conjuring",
        "tahun": "2013",
        "genre": "Horror, Thriller",
        "poster": "assets/posters/conjuring.jpg",
        "rating_awal": 4.5
    }
]

# Film yang dapat dipilih lewat tombol Ganti Film.
# Semua poster disimpan di folder assets/posters.

film_pilihan = {
    "Avatar (2009)": {
        "judul": "Avatar",
        "poster": "assets/posters/avatar.jpg"
    },
    "Maze Runner, The (2014)": {
        "judul": "The Maze Runner",
        "poster": "assets/posters/maze_runner.jpg"
    },
    "Harry Potter and the Prisoner of Azkaban (2004)": {
        "judul": "Harry Potter and the Prisoner of Azkaban",
        "poster": "assets/posters/harry_azkaban.jpg"
    },
    "Dead Poets Society (1989)": {
        "judul": "Dead Poets Society",
        "poster": "assets/posters/dead_poets.jpg"
    },
    "Conjuring, The (2013)": {
        "judul": "The Conjuring",
        "poster": "assets/posters/conjuring.jpg"
    },
    "Perks of Being a Wallflower, The (2012)": {
        "judul": "The Perks of Being a Wallflower",
        "poster": "assets/posters/perks_wallflower.jpg"
    },
    "The Greatest Showman (2017)": {
        "judul": "The Greatest Showman",
        "poster": "assets/posters/greatest_showman.jpg"
    },
    "Amazing Spider-Man, The (2012)": {
        "judul": "The Amazing Spider-Man",
        "poster": "assets/posters/amazing_spiderman.jpg"
    },
    "10 Things I Hate About You (1999)": {
        "judul": "10 Things I Hate About You",
        "poster": "assets/posters/10_things_i_hate_about_you.jpg"
    },
    "Interstellar (2014)": {
        "judul": "Interstellar",
        "poster": "assets/posters/interstellar.jpg"
    },
    "Lord of the Rings: The Return of the King, The (2003)": {
        "judul": "The Lord of the Rings: The Return of the King",
        "poster": "assets/posters/lotr_return_king.jpg"
    },
    "Before Sunrise (1995)": {
        "judul": "Before Sunrise",
        "poster": "assets/posters/before_sunrise.jpg"
    },
    "La La Land (2016)": {
        "judul": "La La Land",
        "poster": "assets/posters/la_la_land.jpg"
    },
    "A Quiet Place (2018)": {
        "judul": "A Quiet Place",
        "poster": "assets/posters/a_quiet_place.jpg"
    },
    "The Intern (2015)": {
        "judul": "The Intern",
        "poster": "assets/posters/the_intern.jpg"
    },
    "Captain America: Civil War (2016)": {
        "judul": "Captain America: Civil War",
        "poster": "assets/posters/captain_america_civil_war.jpg"
    },
    "Deadpool (2016)": {
        "judul": "Deadpool",
        "poster": "assets/posters/deadpool.jpg"
    },
    "Jumanji: Welcome to the Jungle (2017)": {
        "judul": "Jumanji: Welcome to the Jungle",
        "poster": "assets/posters/jumanji.jpg"
    },
    "Devil Wears Prada, The (2006)": {
        "judul": "The Devil Wears Prada",
        "poster": "assets/posters/devil_wears_prada.jpg"
    },
    "13 Going on 30 (2004)": {
        "judul": "13 Going on 30",
        "poster": "assets/posters/13_going_on_30.jpg"
    },
    "Now You See Me (2013)": {
        "judul": "Now You See Me",
        "poster": "assets/posters/now_you_see_me.jpg"
    },
    "Black Panther (2017)": {
        "judul": "Black Panther",
        "poster": "assets/posters/black_panther.jpg"
    },
    "Sherlock Holmes (2009)": {
        "judul": "Sherlock Holmes",
        "poster": "assets/posters/sherlock_holmes.jpg"
    },
    "Twilight (2008)": {
        "judul": "Twilight",
        "poster": "assets/posters/twilight.jpg"
    },
    "Transformers (2007)": {
        "judul": "Transformers",
        "poster": "assets/posters/transformers.jpg"
    },
    "Notebook, The (2004)": {
        "judul": "The Notebook",
        "poster": "assets/posters/the_notebook.jpg"
    },
    "Eternal Sunshine of the Spotless Mind (2004)": {
        "judul": "Eternal Sunshine of the Spotless Mind",
        "poster": "assets/posters/eternal_sunshine.jpg"
    },
    "Catch Me If You Can (2002)": {
        "judul": "Catch Me If You Can",
        "poster": "assets/posters/catch_me_if_you_can.jpg"
    },
    "Hobbit: An Unexpected Journey, The (2012)": {
        "judul": "The Hobbit: An Unexpected Journey",
        "poster": "assets/posters/hobbit_unexpected_journey.jpg"
    },
    "It (2017)": {
        "judul": "It",
        "poster": "assets/posters/it_2017.jpg"
    },
    "Minions (2015)": {
        "judul": "Minions",
        "poster": "assets/posters/minions.jpg"
    },
    "Kingsman: The Secret Service (2015)": {
        "judul": "Kingsman: The Secret Service",
        "poster": "assets/posters/kingsman_secret_service.jpg"
    },
    "Inside Out (2015)": {
        "judul": "Inside Out",
        "poster": "assets/posters/inside_out.jpg"
    },
    "John Wick (2014)": {
        "judul": "John Wick",
        "poster": "assets/posters/john_wick.jpg"
    },
    "Divergent (2014)": {
        "judul": "Divergent",
        "poster": "assets/posters/divergent.jpg"
    },
    "Thor: The Dark World (2013)": {
        "judul": "Thor: The Dark World",
        "poster": "assets/posters/thor_dark_world.jpg"
    },
    "Avengers, The (2012)": {
        "judul": "The Avengers",
        "poster": "assets/posters/avengers_2012.jpg"
    },
    "Mission: Impossible (1996)": {
        "judul": "Mission: Impossible",
        "poster": "assets/posters/mission_impossible.jpg"
    },
    "Good Will Hunting (1997)": {
        "judul": "Good Will Hunting",
        "poster": "assets/posters/good_will_hunting.jpg"
    },
    "Godfather, The (1972)": {
        "judul": "The Godfather",
        "poster": "assets/posters/the_godfather.jpg"
    },
    "Fight Club (1999)": {
        "judul": "Fight Club",
        "poster": "assets/posters/fight_club.jpg"
    },
    "Zootopia (2016)": {
        "judul": "Zootopia",
        "poster": "assets/posters/zootopia.jpg"
    },
    "Man of Steel (2013)": {
        "judul": "Man of Steel",
        "poster": "assets/posters/man_of_steel.jpg"
    },
    "Doctor Strange (2016)": {
        "judul": "Doctor Strange",
        "poster": "assets/posters/doctor_strange.jpg"
    },
    "Incredibles, The (2004)": {
        "judul": "The Incredibles",
        "poster": "assets/posters/the_incredibles.jpg"
    },
    "Finding Nemo (2003)": {
        "judul": "Finding Nemo",
        "poster": "assets/posters/finding_nemo.jpg"
    }
}

def tampilkan_bintang(rating):
    jumlah_bintang = int(rating)
    return "★" * jumlah_bintang + "☆" * (5 - jumlah_bintang)

# Menyimpan daftar film yang sedang tampil di aplikasi
if "film_tampil" not in st.session_state:
    st.session_state["film_tampil"] = film_awal.copy()

# Menyimpan kondisi form ganti film
if "form_ganti_film" not in st.session_state:
    st.session_state["form_ganti_film"] = False

# Mengatur warna dan tampilan dasar
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            
    .stApp {
        background-color: #0b0c13;
        color: white;
    }
            
    .stApp,
    .stApp *,
    .header-web,
    .header-web *,
    .proses,
    .proses * {
        font-family: "Inter", sans-serif !important;
    }

    .block-container {
        max-width: 1380px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
}

    .header-web {
        padding: 28px;
        border-radius: 18px;
        background: linear-gradient(135deg, #2b1450, #111522);
        border: 1px solid #49326f;
        margin-bottom: 18px;
    }

    .judul-header {
        font-size: 30px;
        font-weight: 700;
        color: #f5f3ff;
        margin-bottom: 6px;
    }

    .ungu {
        color: #a855f7;
    }

    .subjudul-header {
        font-size: 13px;
        color: #b8b2c8;
    }

    .metode {
        margin-top: 16px;
        padding: 12px 15px;
        border-radius: 10px;
        background-color: #191525;
        border: 1px solid #3a2858;
        color: #d8c8ff;
        font-size: 12px;
    }

    .proses {
        display: flex;
        justify-content: space-between;
        gap: 10px;
        padding: 16px 20px;
        border-radius: 14px;
        background-color: #12121c;
        border: 1px solid #29243c;
        margin-bottom: 28px;
    }

    .step {
        width: 32%;
        color: #b8b2c8;
        font-size: 11px;
    }

    .step-aktif {
        color: #d8c8ff;
        font-weight: 600;
    }

    [data-testid="stMetric"] {
        background-color: #181724;
        border: 1px solid #30284c;
        border-radius: 14px;
        padding: 15px;
    }

    [data-testid="stMetricLabel"] {
        color: #aaa5bf;
    }

    [data-testid="stMetricValue"] {
        color: #c4b5fd;
    }
    
        .judul-bagian {
        color: #f5f3ff;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 2px;
    }

    .keterangan-bagian {
        color: #aaa5bf;
        font-size: 11px;
        margin-bottom: 12px;
    }

    .film-atas {
        display: flex;
        justify-content: space-between;
        gap: 12px;
    }

    .judul-film {
        color: #f5f3ff;
        font-size: 13px;
        font-weight: 600;
    }

    .tahun-film {
        color: #aaa5bf;
        font-size: 10px;
        margin-top: 2px;
    }

    .genre-film {
        display: inline-block;
        margin-top: 7px;
        padding: 3px 8px;
        border-radius: 10px;
        border: 1px solid #6d4cd2;
        background: #211b37;
        color: #c4b5fd;
        font-size: 9px;
    }

    .nilai-rating {
        color: #b794f6;
        font-size: 15px;
        font-weight: 700;
        text-align: right;
    }

    .bintang-rating {
        color: #a78bfa;
        font-size: 12px;
        text-align: right;
        margin-top: 4px;
    }

    .poster-kosong {
        width: 68px;
        height: 96px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
        background: #30204f;
        color: #d8c8ff;
        font-size: 28px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #181724;
        border: 1px solid #30284c;
        border-radius: 14px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.22);
    }

  
    div[data-baseweb="slider"] [role="slider"]{
        background-color:#A855F7 !important;
        border:2px solid #A855F7 !important;
        box-shadow:0 0 10px rgba(168,85,247,.5) !important;
    }


    div[data-baseweb="slider"] [data-testid="stThumbValue"] + div{
        background:#A855F7 !important;
    }


    div[data-baseweb="slider"]{
        accent-color:#A855F7 !important;
    }

    button[kind="primary"],
    [data-testid="stBaseButton-primary"] {
        border: none;
        background: linear-gradient(135deg, #a855f7, #7c3aed);
        box-shadow: 0 0 16px rgba(168, 85, 247, 0.42);
    }

    button[kind="secondary"],
    [data-testid="stBaseButton-secondary"] {
        border: 1px solid #6d4cd2;
        background: #211b37;
        color: #d8c8ff;
    }

    .hasil-kosong {
    min-height: 170px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    color: #aaa5bf;
    font-size: 13px;
    line-height: 1.6;
}

.empty-icon {
    font-size: 28px;
    margin-bottom: 8px;
    color: #a855f7;
}

.empty-title {
    color: #f5f3ff;
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 4px;
}

.empty-text {
    color: #aaa5bf;
    font-size: 11px;
}
            
    /* ===== Header versi Figma ===== */

    .hero-figma {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 28px;
        padding: 34px 38px;
        margin-bottom: 18px;
        border: 1px solid #3c2a62;
        border-radius: 18px;
        background:
            radial-gradient(circle at 15% 15%, rgba(139, 92, 246, 0.22), transparent 30%),
            linear-gradient(135deg, #251044 0%, #17182a 62%, #101421 100%);
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.28);
    }

    .hero-kiri-figma {
        display: flex;
        align-items: center;
        gap: 18px;
    }

    .hero-icon-figma {
        width: 62px;
        height: 62px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        border-radius: 15px;
        background: linear-gradient(135deg, #a855f7, #6d28d9);
        box-shadow: 0 0 20px rgba(168, 85, 247, 0.48);
        font-size: 30px;
        line-height: 1;
    }
            
    .hero-film-logo {
    width: 31px;
    height: 24px;
    position: relative;
    box-sizing: border-box;
    border: 2px solid #ffffff;
    border-radius: 4px;
}

.hero-film-logo::before {
    content: "";
    position: absolute;
    left: -2px;
    top: -10px;
    width: 31px;
    height: 8px;
    box-sizing: border-box;
    border: 2px solid #ffffff;
    border-bottom: 0;
    border-radius: 4px 4px 0 0;
    background:
        repeating-linear-gradient(
            135deg,
            transparent 0 5px,
            rgba(255, 255, 255, 0.95) 5px 9px,
            transparent 9px 14px
        );
}

.hero-film-logo::after {
    content: "▶";
    position: absolute;
    left: 9px;
    top: 3px;
    color: #ffffff;
    font-size: 11px;
    line-height: 1;
}

    .hero-title-figma {
        color: #f5f3ff;
        font-size: 30px;
        font-weight: 700;
        line-height: 1.15;
    }

    .hero-title-figma span {
        color: #a855f7;
    }

    .hero-subtitle-figma {
        margin-top: 7px;
        color: #b8b2c8;
        font-size: 13px;
    }

    .method-card-figma {
        min-width: 310px;
        display: flex;
        align-items: center;
        gap: 13px;
        padding: 15px 17px;
        border: 1px solid #33244f;
        border-radius: 12px;
        background: rgba(19, 18, 31, 0.72);
    }

    .method-icon-figma {
        color: #a855f7;
        font-size: 27px;
    }

    .method-small-figma {
        color: #aaa5bf;
        font-size: 10px;
    }

    .method-title-figma {
        margin-top: 3px;
        color: #f5f3ff;
        font-size: 12px;
        font-weight: 700;
    }

    .method-info-figma {
        margin-top: 4px;
        color: #d8c8ff;
        font-size: 10px;
    }

    /* ===== Alur 1 - 2 - 3 ===== */

    .alur-figma {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 22px;
    max-width: 1080px;
    margin: 0 auto 30px auto;
    padding: 18px 28px;
    border: 1px solid #29243c;
    border-radius: 14px;
    background: rgba(17, 17, 27, 0.84);
}

.alur-step-figma {
    display: flex;
    align-items: center;
    gap: 10px;
    flex: 0 1 250px;
}

.alur-number-figma {
    width: 30px;
    height: 30px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    background: #2b2939;
    color: #d8c8ff;
    font-size: 12px;
    font-weight: 700;
}

.alur-number-active {
    background: #8b5cf6;
    color: #ffffff;
    box-shadow: 0 0 14px rgba(139, 92, 246, 0.60);
}

.alur-title-figma {
    color: #d8c8ff;
    font-size: 11px;
    font-weight: 700;
}

.alur-text-figma {
    margin-top: 3px;
    color: #aaa5bf;
    font-size: 10px;
    line-height: 1.35;
}

.alur-line-figma {
    width: 112px;
    height: 1px;
    flex-shrink: 0;
    background: #50466b;
}

    .alur-step-figma {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .alur-number-figma {
        width: 29px;
        height: 29px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        background: #2b2939;
        color: #d8c8ff;
        font-size: 11px;
        font-weight: 700;
    }

    .alur-number-active {
        background: #8b5cf6;
        color: #ffffff;
        box-shadow: 0 0 14px rgba(139, 92, 246, 0.60);
    }

    .alur-title-figma {
        color: #d8c8ff;
        font-size: 11px;
        font-weight: 700;
    }

    .alur-text-figma {
        margin-top: 2px;
        color: #aaa5bf;
        font-size: 10px;
    }

    .alur-line-figma {
        height: 1px;
        background: #50466b;
    }

    /* ===== Bagian bawah seperti Figma ===== */

    .footer-figma {
        display: grid;
        grid-template-columns: 1.1fr 2.1fr 1.15fr;
        gap: 14px;
        margin-top: 40px;
    }

    .footer-card-figma {
        min-height: 142px;
        padding: 18px;
        border: 1px solid #30284c;
        border-radius: 14px;
        background: #181724;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.20);
    }

    .footer-title-figma {
        margin-bottom: 12px;
        color: #f5f3ff;
        font-size: 13px;
        font-weight: 700;
    }

    .footer-text-figma {
        color: #aaa5bf;
        font-size: 11px;
        line-height: 1.55;
    }

    .footer-proses-figma {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        color: #aaa5bf;
        font-size: 10px;
        text-align: center;
    }

    .footer-proses-icon {
        display: block;
        margin-bottom: 7px;
        color: #a855f7;
        font-size: 20px;
    }

    .footer-arrow {
        color: #77658f;
        font-size: 17px;
    }

    .dataset-row-figma {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        margin-top: 7px;
        color: #aaa5bf;
        font-size: 10px;
    }

    .dataset-row-figma strong {
        color: #d8c8ff;
        font-weight: 600;
        text-align: right;
    }

    .footer-credit-figma {
        margin-top: 18px;
        color: #77738a;
        font-size: 10px;
        text-align: center;
    }

    @media (max-width: 850px) {
        .hero-figma {
            flex-direction: column;
            align-items: flex-start;
            padding: 24px;
        }

        .method-card-figma {
            width: 100%;
            min-width: 0;
        }

        .alur-figma {
            grid-template-columns: 1fr;
            gap: 13px;
        }

        .alur-line-figma {
            display: none;
        }

        .footer-figma {
            grid-template-columns: 1fr;
        }

        .footer-proses-figma {
            flex-wrap: wrap;
        }

        .footer-arrow {
            display: none;
        }
    }
        
        /* Track slider */
    .stSlider [data-baseweb="slider"] > div > div {
        background: #3A3154 !important;
    }

    /* Track aktif */
    .stSlider [data-baseweb="slider"] > div > div > div {
        background: #A855F7 !important;
    }

</style>
""", unsafe_allow_html=True)

# Header versi Figma
# Header dan alur proses
st.html("""
<div class="hero-figma">
    <div class="hero-kiri-figma">
        <div class="hero-icon-figma">🎞️</div>

        <div>
            <div class="hero-title-figma">
                Sistem <span>Rekomendasi</span> Film
            </div>

            <div class="hero-subtitle-figma">
                Berikan rating pada beberapa film untuk mendapatkan rekomendasi personal.
            </div>
        </div>
    </div>

    <div class="method-card-figma">
        <div class="method-icon-figma">👥</div>

        <div>
            <div class="method-small-figma">Menggunakan</div>

            <div class="method-title-figma">
                User-Based Collaborative Filtering
            </div>

            <div class="method-info-figma">
                K = 20 Neighbors &nbsp;•&nbsp; Cosine Similarity
            </div>
        </div>
    </div>
</div>
""")

st.html("""
<div class="alur-figma">
    <div class="alur-step-figma">
        <div class="alur-number-figma alur-number-active">1</div>

        <div>
            <div class="alur-title-figma">Input Rating</div>
            <div class="alur-text-figma">
                Beri rating pada beberapa film
            </div>
        </div>
    </div>

    <div class="alur-line-figma"></div>

    <div class="alur-step-figma">
        <div class="alur-number-figma">2</div>

        <div>
            <div class="alur-title-figma">Proses Sistem</div>
            <div class="alur-text-figma">
                Cari user mirip dan prediksi rating
            </div>
        </div>
    </div>

    <div class="alur-line-figma"></div>

    <div class="alur-step-figma">
        <div class="alur-number-figma">3</div>

        <div>
            <div class="alur-title-figma">Rekomendasi</div>
            <div class="alur-text-figma">
                Tampilkan Top-10 film untuk user
            </div>
        </div>
    </div>
</div>
""")

# Bagian input rating dan hasil rekomendasi
kolom_kiri, kolom_kanan = st.columns(
    [1, 1.45],
    gap="medium",
    vertical_alignment="top"
)

with kolom_kiri:
    st.html("""
    <div class="judul-bagian">✎ 1. Berikan Rating pada Beberapa Film</div>
    <div class="keterangan-bagian">
        Berikan rating sesuai dengan film yang pernah Anda tonton.
    </div>
    """)

    rating_user = {}

    for film in st.session_state["film_tampil"]:
        with st.container(border=True):
            poster_kolom, isi_kolom = st.columns([1, 4])

        with poster_kolom:
            if os.path.exists(film["poster"]):
                st.image(film["poster"], width=68)
            else:
                st.html('<div class="poster-kosong">🎬</div>')

        with isi_kolom:
            key_rating = f"rating_{film['movieId']}"

            if key_rating not in st.session_state:
                st.session_state[key_rating] = film["rating_awal"]

            nilai_rating = st.session_state[key_rating]

            st.html(f"""
            <div class="film-atas">
                <div>
                    <div class="judul-film">{film["judul"]}</div>
                    <div class="tahun-film">({film["tahun"]})</div>
                    <div class="genre-film">{film["genre"]}</div>
                </div>

                <div>
                    <div class="nilai-rating">{nilai_rating:.1f}</div>
                    <div class="bintang-rating">
                        {tampilkan_bintang(nilai_rating)}
                    </div>
                </div>
            </div>
            """)

            nilai_rating = st.select_slider(
                f"Rating untuk {film['judul']}",
                options=[0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0],
                value=st.session_state[key_rating],
                key=key_rating,
                label_visibility="collapsed"
            )

            rating_user[film["movieId"]] = nilai_rating

    # Tombol di bawah daftar film
tombol_rekomendasi, tombol_ganti = st.columns([1.5, 1])

with tombol_rekomendasi:
    klik_rekomendasi = st.button(
        "✦ Dapatkan Rekomendasi",
        type="primary",
        use_container_width=True
    )

with tombol_ganti:
    klik_ganti = st.button(
        "↻ Ganti Film",
        use_container_width=True
    )

if klik_rekomendasi:
    with st.spinner("Sedang membuat rekomendasi..."):
        hasil_rekomendasi = buat_rekomendasi_user_baru(
            rating_user,
            k=20,
            jumlah_rekomendasi=10
        )

    st.session_state["hasil_rekomendasi"] = hasil_rekomendasi

if klik_ganti:
    st.session_state["form_ganti_film"] = not st.session_state["form_ganti_film"]

if st.session_state["form_ganti_film"]:
    st.markdown("#### Ganti Salah Satu Film")

    daftar_card = [
        f"{film['judul']} ({film['tahun']})"
        for film in st.session_state["film_tampil"]
    ]

    film_lama = st.selectbox(
        "Film yang ingin diganti",
        daftar_card
    )

    pilihan_baru = st.selectbox(
        "Pilih film lain",
        sorted(film_pilihan.keys())
    )

    if st.button("Simpan Film Pengganti", use_container_width=True):
        posisi_film = daftar_card.index(film_lama)

        data_film = movies[movies["title"] == pilihan_baru].iloc[0]
        info_film = film_pilihan[pilihan_baru]

        tahun = pilihan_baru[-5:-1]
        genre = data_film["genres"].replace("|", ", ")

        st.session_state["film_tampil"][posisi_film] = {
            "movieId": int(data_film["movieId"]),
            "judul": info_film["judul"],
            "tahun": tahun,
            "genre": genre,
            "poster": info_film["poster"],
            "rating_awal": 3.0
        }

        st.session_state["form_ganti_film"] = False
        st.rerun()

with kolom_kanan:
    st.html("""
    <div class="judul-bagian">★ 2. Top-10 Rekomendasi untuk Anda</div>
    <div class="keterangan-bagian">
        Film direkomendasikan berdasarkan preferensi dan kemiripan dengan user lain.
    </div>
    """)

    hasil_rekomendasi = st.session_state.get("hasil_rekomendasi")

    with st.container(border=True):
        if hasil_rekomendasi is None:
            st.html("""
            <div class="hasil-kosong">
                <div class="empty-icon">✦</div>
                <div class="empty-title">Belum Ada Rekomendasi</div>
                <div class="empty-text">
                    Atur rating film di sebelah kiri, lalu tekan<br>
                    <b>Dapatkan Rekomendasi</b> untuk melihat Top-10 film.
                </div>
            </div>
    """)

        elif hasil_rekomendasi.empty:
            st.warning(
                "Rekomendasi belum ditemukan. "
                "Coba ubah rating atau ganti beberapa film."
            )

        else:
            tabel_rekomendasi = hasil_rekomendasi[
                ["title", "genres", "rating_prediksi"]
            ].copy()

            tabel_rekomendasi["genres"] = tabel_rekomendasi[
                "genres"
            ].str.replace("|", ", ", regex=False)

            tabel_rekomendasi["rating_prediksi"] = tabel_rekomendasi[
                "rating_prediksi"
            ].map(lambda nilai: f"★ {nilai:.2f}")

            tabel_rekomendasi.columns = [
                "Judul Film",
                "Genre",
                "Prediksi Rating"
            ]

            tabel_rekomendasi.insert(
                0,
                "No.",
                range(1, len(tabel_rekomendasi) + 1)
            )

            st.dataframe(
                tabel_rekomendasi,
                width="stretch",
                height=420,
                hide_index=True
            )

# Bagian bawah halaman
st.html(f"""
<div class="footer-figma">
    <div class="footer-card-figma">
        <div class="footer-title-figma">● Tentang Sistem</div>

        <div class="footer-text-figma">
            Sistem rekomendasi film menggunakan
            User-Based Collaborative Filtering untuk
            memberikan rekomendasi berdasarkan
            kemiripan preferensi rating antaruser.
        </div>
    </div>

    <div class="footer-card-figma">
        <div class="footer-title-figma">Proses yang Dilakukan</div>

        <div class="footer-proses-figma">
            <div>
                <span class="footer-proses-icon">☷</span>
                Menerima rating<br>dari user
            </div>

            <div class="footer-arrow">→</div>

            <div>
                <span class="footer-proses-icon">👥</span>
                Mencari 20 user<br>paling mirip
            </div>

            <div class="footer-arrow">→</div>

            <div>
                <span class="footer-proses-icon">✦</span>
                Memprediksi dan<br>mengurutkan rating
            </div>

            <div class="footer-arrow">→</div>

            <div>
                <span class="footer-proses-icon">★</span>
                Menampilkan Top-10<br>rekomendasi
            </div>
        </div>
    </div>

    <div class="footer-card-figma">
        <div class="footer-title-figma">Informasi Dataset</div>

        <div class="dataset-row-figma">
            <span>Dataset</span>
            <strong>MovieLens Latest Small</strong>
        </div>

        <div class="dataset-row-figma">
            <span>Jumlah User</span>
            <strong>{teks_user}</strong>
        </div>

        <div class="dataset-row-figma">
            <span>Jumlah Film</span>
            <strong>{teks_film}</strong>
        </div>

        <div class="dataset-row-figma">
            <span>Jumlah Rating</span>
            <strong>{teks_rating}</strong>
        </div>
    </div>
</div>

<div class="footer-credit-figma">
    Sistem Rekomendasi Film &nbsp;•&nbsp;
    User-Based Collaborative Filtering &nbsp;•&nbsp;
    MovieLens Latest Small
</div>
""")