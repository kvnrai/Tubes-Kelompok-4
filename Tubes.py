import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import numpy as np

# Load data
file_path = "day.csv"
df = pd.read_csv(file_path)

# Convert date column to datetime format
df['dteday'] = pd.to_datetime(df['dteday'])
df['year'] = df['dteday'].dt.year
df['month'] = df['dteday'].dt.month
df['day'] = df['dteday'].dt.day

# Sidebar for user input
st.sidebar.title("Pengaturan Analisis")

# Filter selection
st.sidebar.header("Filter Data")
selected_year = st.sidebar.selectbox("Pilih Tahun", df['year'].unique())
selected_month = st.sidebar.selectbox("Pilih Bulan", range(1, 13))

# Analysis selection
st.sidebar.header("Pilih Analisis")
analysis_options = {
    "Distribusi Penyewaan Sepeda": "Distribusi Penyewaan Sepeda",
    "Distribusi Penyewaan berdasarkan Cuaca": "Distribusi Penyewaan berdasarkan Cuaca",
    "Rata-rata Penyewaan Sepeda per Musim": "Rata-rata Penyewaan Sepeda per Musim",
    "Tren Penyewaan Sepeda per Hari dalam Sebulan": "Tren Penyewaan Sepeda per Hari dalam Sebulan",
    "Korelasi antara Faktor Cuaca dan Penyewaan Sepeda": "Korelasi antara Faktor Cuaca dan Penyewaan Sepeda",
    "Distribusi Penyewaan Sepeda Berdasarkan Hari Kerja & Akhir Pekan": "Distribusi Penyewaan Sepeda Berdasarkan Hari Kerja & Akhir Pekan",
    "Pengaruh Suhu terhadap Penyewaan Sepeda": "Pengaruh Suhu terhadap Penyewaan Sepeda",
    "Analisis Time Series Penyewaan Sepeda": "Analisis Time Series Penyewaan Sepeda",
    "Analisis Klaster Penyewaan Sepeda": "Analisis Klaster Penyewaan Sepeda",
    "Analisis Korelasi Cuaca vs. Penggunaan Sepeda": "Analisis Korelasi Cuaca vs. Penggunaan Sepeda"
}
selected_analysis = st.sidebar.selectbox("Pilih Jenis Analisis", list(analysis_options.keys()))

# Filter data based on selections
filtered_df = df[(df['year'] == selected_year) & (df['month'] == selected_month)]

# Display header
st.title("Analisis Data day.csv")
st.write("Menampilkan data yang difilter berdasarkan pilihan tahun dan bulan.")

# Display filtered data
st.dataframe(filtered_df)

# Display Dataset Description
st.write("Dataset day.csv berisi data peminjaman sepeda dari sistem berbagi sepeda selama beberapa tahun. Data ini mencakup berbagai informasi seperti tanggal, musim, kondisi cuaca, suhu, kelembaban, kecepatan angin, serta jumlah total penyewaan sepeda setiap hari. Dengan dataset ini, kita dapat menganalisis tren penyewaan sepeda, faktor yang mempengaruhinya, serta melakukan klasterisasi atau analisis korelasi.")

# Conditional analysis display
if selected_analysis == "Distribusi Penyewaan Sepeda":
    st.subheader("Distribusi Penyewaan Sepeda")
    st.bar_chart(filtered_df[['dteday', 'cnt']].set_index('dteday'))

elif selected_analysis == "Distribusi Penyewaan berdasarkan Cuaca":
    st.subheader("Distribusi Penyewaan berdasarkan Cuaca")
    st.bar_chart(filtered_df[['weathersit', 'cnt']].groupby('weathersit').sum())

elif selected_analysis == "Rata-rata Penyewaan Sepeda per Musim":
    st.subheader("Rata-rata Penyewaan Sepeda per Musim")
    season_avg = df.groupby('season')['cnt'].mean()
    st.bar_chart(season_avg)

elif selected_analysis == "Tren Penyewaan Sepeda per Hari dalam Sebulan":
    st.subheader("Tren Penyewaan Sepeda per Hari dalam Sebulan")
    daily_trend = filtered_df.groupby('day')['cnt'].sum()
    st.line_chart(daily_trend)

elif selected_analysis == "Korelasi antara Faktor Cuaca dan Penyewaan Sepeda":
    st.subheader("Korelasi antara Faktor Cuaca dan Penyewaan Sepeda")
    st.scatter_chart(filtered_df, x='temp', y='cnt')

elif selected_analysis == "Distribusi Penyewaan Sepeda Berdasarkan Hari Kerja & Akhir Pekan":
    st.subheader("Distribusi Penyewaan Sepeda Berdasarkan Hari Kerja & Akhir Pekan")
    st.bar_chart(filtered_df.groupby('workingday')['cnt'].sum())

elif selected_analysis == "Pengaruh Suhu terhadap Penyewaan Sepeda":
    st.subheader("Pengaruh Suhu terhadap Penyewaan Sepeda")
    fig, ax = plt.subplots()
    ax.scatter(df['temp'], df['cnt'], alpha=0.5)
    ax.set_xlabel("Suhu")
    ax.set_ylabel("Jumlah Penyewaan")
    st.pyplot(fig)

elif selected_analysis == "Analisis Time Series Penyewaan Sepeda":
    st.subheader("Analisis Time Series Penyewaan Sepeda")
    ts_data = df[['dteday', 'cnt']].set_index('dteday')
    fig, ax = plt.subplots(figsize=(12,6))
    ts_data.plot(ax=ax)
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Jumlah Penyewaan")
    ax.set_title("Tren Penyewaan Sepeda dari Waktu ke Waktu")
    st.pyplot(fig)

elif selected_analysis == "Analisis Klaster Penyewaan Sepeda":
    st.subheader("Analisis Klaster Penyewaan Sepeda")
    features = df[['temp', 'hum', 'windspeed', 'cnt']]
    kmeans = KMeans(n_clusters=3, random_state=42).fit(features)
    df['cluster'] = kmeans.labels_
    st.bar_chart(df.groupby('cluster')['cnt'].mean())

elif selected_analysis == "Analisis Korelasi Cuaca vs. Penggunaan Sepeda":
    st.subheader("Analisis Korelasi Cuaca vs. Penggunaan Sepeda")
    corr_matrix = df[['temp', 'atemp', 'hum', 'windspeed', 'cnt']].corr()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# CSV Download Feature
@st.cache_data
def convert_df(df):
    return df.to_csv().encode("utf-8")

csv = convert_df(df)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name="bike_sharing_data.csv",
    mime="text/csv",
)
