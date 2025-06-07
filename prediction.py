import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import re

st.title("Analisis Regresi Linear Berat Sampah Kota Sukabumi")

@st.cache_data
def load_data():
    with open("data-sampah-kota-sukabumi.txt", "r", encoding="utf-8") as f:
        raw = f.read()

    lines = [line.strip() for line in raw.strip().split("\n") if line.strip()]
    cleaned_rows = []

    for line in lines[2:]:
        parts = re.split(r'\s{2,}', line)
        if len(parts) >= 2:
            month = parts[0].strip()
            numbers = re.findall(r'\d+\.\d+', parts[1])
            if len(numbers) == 6:
                cleaned_rows.append([month] + [float(n) for n in numbers])

    df = pd.DataFrame(cleaned_rows, columns=["Bulan", "2017", "2018", "2020", "2021", "2022", "2023"])
    return df

df = load_data()

# Hapus baris 'TAHUNAN' jika ada, supaya tidak jadi kolom yang bikin error
df = df[df["Bulan"].str.upper() != "TAHUNAN"]

# Set index lagi setelah hapus baris yang tidak perlu
df.set_index("Bulan", inplace=True)

bulan_urut = [
    'JANUARI', 'FEBRUARI', 'MARET', 'APRIL', 'MEI', 'JUNI',
    'JULI', 'AGUSTUS', 'SEPTEMBER', 'OKTOBER', 'NOVEMBER', 'DESEMBER'
]

df_reset = df.reset_index()

# Transpose: ubah jadi per tahun index, bulan kolom
df_bulan_per_tahun = df_reset.set_index("Bulan").T

# Ubah semua nama kolom (bulan) ke uppercase agar konsisten
df_bulan_per_tahun.columns = df_bulan_per_tahun.columns.str.upper()

st.write("Kolom df_bulan_per_tahun:", df_bulan_per_tahun.columns.tolist())
st.write("Bulan urut:", bulan_urut)

missing = [b for b in bulan_urut if b not in df_bulan_per_tahun.columns]
if missing:
    st.error(f"Kolom bulan hilang: {missing}")
else:
    df_bulan_per_tahun = df_bulan_per_tahun[bulan_urut]

    st.subheader("Data Berat Sampah per Bulan per Tahun (Ton)")
    st.dataframe(df_bulan_per_tahun)

# Total sampah tahunan
yearly = df.sum().reset_index()
yearly.columns = ["Tahun", "Total Sampah (Ton)"]
st.subheader("Total Sampah per Tahun")
st.dataframe(yearly)

# Grafik batang total tahunan
st.subheader("Grafik Total Sampah per Tahun")
fig1, ax1 = plt.subplots()
sns.barplot(data=yearly, x="Tahun", y="Total Sampah (Ton)", ax=ax1)
st.pyplot(fig1)

# Regresi Linear total tahunan
X = yearly["Tahun"].astype(int).values.reshape(-1, 1)
y = yearly["Total Sampah (Ton)"].values
model_total = LinearRegression()
model_total.fit(X, y)
preds_total = model_total.predict(X)

st.subheader("Regresi Linear Total Tahunan")
fig2, ax2 = plt.subplots()
ax2.scatter(X, y, label="Data Aktual")
ax2.plot(X, preds_total, color="red", label="Regresi Linear")
ax2.set_xlabel("Tahun")
ax2.set_ylabel("Total Sampah (Ton)")
ax2.legend()
st.pyplot(fig2)

# Prediksi berdasarkan input user
st.subheader("Prediksi Berat Sampah Berdasarkan Tahun dan Bulan")
tahun_input = st.number_input("Masukkan tahun", min_value=2024, max_value=2100, value=2025)

# Pilihan bulan sudah uppercase
bulan_input = st.selectbox("Pilih bulan", df.index.str.upper())

df_bulanan = df.reset_index().melt(id_vars="Bulan", var_name="Tahun", value_name="Ton")
df_bulanan["Tahun"] = df_bulanan["Tahun"].astype(int)
df_bulanan["Bulan"] = df_bulanan["Bulan"].str.upper()

df_bulan_terpilih = df_bulanan[df_bulanan["Bulan"] == bulan_input]

X_bulan = df_bulan_terpilih["Tahun"].values.reshape(-1, 1)
y_bulan = df_bulan_terpilih["Ton"].values
model_bulan = LinearRegression()
model_bulan.fit(X_bulan, y_bulan)

pred_bulan = model_bulan.predict([[tahun_input]])
st.success(f"Prediksi berat sampah untuk bulan **{bulan_input} {tahun_input}** adalah **{pred_bulan[0]:.2f} ton**")
