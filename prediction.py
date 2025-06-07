import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import re

st.title("Analisis Regresi Linear Berat Sampah Kota Sukabumi")

@st.cache_data
def load_data():
    # Baca file mentah
    with open("data-sampah-kota-sukabumi.txt", "r", encoding="utf-8") as f:
        raw = f.read()

    lines = [line.strip() for line in raw.strip().split("\n") if line.strip()]
    cleaned_rows = []

    # Skip 2 baris header awal
    for line in lines[2:]:
        parts = re.split(r'\s{2,}', line)
        if len(parts) >= 2:
            month = parts[0].strip()
            # Gabungkan angka dan pisahkan dengan spasi jika ada titik mepet angka
            numbers = re.findall(r'\d+\.\d+', parts[1])
            if len(numbers) == 6:  # 6 tahun
                cleaned_rows.append([month] + [float(n) for n in numbers])

    df = pd.DataFrame(cleaned_rows, columns=["Bulan", "2017", "2018", "2020", "2021", "2022", "2023"])
    df.set_index("Bulan", inplace=True)
    return df

df = load_data()

st.subheader("Data Berat Sampah per Bulan (Ton)")
st.dataframe(df)

# Tampilkan total tahunan
yearly = df.sum().reset_index()
yearly.columns = ["Tahun", "Total Sampah (Ton)"]
st.subheader("Total Sampah per Tahun")
st.dataframe(yearly)

# Grafik batang
st.subheader("Grafik Total Sampah per Tahun")
fig1, ax1 = plt.subplots()
sns.barplot(data=yearly, x="Tahun", y="Total Sampah (Ton)", ax=ax1)
st.pyplot(fig1)

# Regresi Linear
X = yearly["Tahun"].astype(int).values.reshape(-1, 1)
y = yearly["Total Sampah (Ton)"].values
model = LinearRegression()
model.fit(X, y)
preds = model.predict(X)

# Visualisasi regresi
st.subheader("Regresi Linear")
fig2, ax2 = plt.subplots()
ax2.scatter(X, y, label="Data Aktual")
ax2.plot(X, preds, color="red", label="Regresi Linear")
ax2.set_xlabel("Tahun")
ax2.set_ylabel("Total Sampah (Ton)")
ax2.legend()
st.pyplot(fig2)

# Prediksi tahun baru
st.subheader("Prediksi Berat Sampah Tahun Berikutnya")
tahun_pred = st.number_input("Masukkan tahun", min_value=2024, max_value=2100, value=2025)
hasil_pred = model.predict([[tahun_pred]])
st.write(f"Prediksi total sampah pada tahun {tahun_pred} adalah {hasil_pred[0]:.2f} ton")
