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

    for line in lines[2:]:  # asumsi 2 baris header
        parts = re.split(r'\s{2,}', line)
        if len(parts) >= 2:
            month = parts[0].strip().upper()  # normalisasi bulan ke huruf besar
            # Gabungkan seluruh kolom angka setelah bulan
            angka_str = " ".join(parts[1:])
            numbers = re.findall(r'\d+\.\d+', angka_str)
            if len(numbers) == 6:
                cleaned_rows.append([month] + [float(n) for n in numbers])

    df = pd.DataFrame(cleaned_rows, columns=["BULAN", "2017", "2018", "2020", "2021", "2022", "2023"])
    df.set_index("BULAN", inplace=True)
    return df


df = load_data()

# Atur urutan bulan dengan urutan eksplisit
bulan_urut = [
    'JANUARI', 'FEBRUARI', 'MARET', 'APRIL', 'MEI', 'JUNI',
    'JULI', 'AGUSTUS', 'SEPTEMBER', 'OKTOBER', 'NOVEMBER', 'DESEMBER'
]

# Reset index agar 'Bulan' jadi kolom biasa
df_reset = df.reset_index()

# Ubah ke bentuk per tahun sebagai index, bulan jadi kolom
df_bulan_per_tahun = df_reset.set_index("Bulan").T  # Transpose

# Normalisasi nama kolom: hapus spasi dan ubah ke huruf kapital semua
df_bulan_per_tahun.columns = df_bulan_per_tahun.columns.str.strip().str.upper()

# Pastikan hanya ambil bulan yang ada di dataframe dan sesuai urutan
existing_months = [month for month in bulan_urut if month in df_bulan_per_tahun.columns]
df_bulan_per_tahun = df_bulan_per_tahun[existing_months]

# Tampilkan data
st.subheader("Data Berat Sampah per Bulan per Tahun (Ton)")
st.dataframe(df_bulan_per_tahun)

# Tampilkan total tahunan
yearly = df.sum().reset_index()
yearly.columns = ["Tahun", "Total Sampah (Ton)"]
st.subheader("Total Sampah per Tahun")
st.dataframe(yearly)

# Grafik batang total tahunan
st.subheader("Grafik Total Sampah per Tahun")
fig1, ax1 = plt.subplots()
sns.barplot(data=yearly, x="Tahun", y="Total Sampah (Ton)", ax=ax1)
st.pyplot(fig1)

# Regresi Linear berdasarkan total tahunan
X = yearly["Tahun"].astype(int).values.reshape(-1, 1)
y = yearly["Total Sampah (Ton)"].values
model_total = LinearRegression()
model_total.fit(X, y)
preds_total = model_total.predict(X)

# Visualisasi regresi total
st.subheader("Regresi Linear Total Tahunan")
fig2, ax2 = plt.subplots()
ax2.scatter(X, y, label="Data Aktual")
ax2.plot(X, preds_total, color="red", label="Regresi Linear")
ax2.set_xlabel("Tahun")
ax2.set_ylabel("Total Sampah (Ton)")
ax2.legend()
st.pyplot(fig2)

# Prediksi berdasarkan input user (tahun & bulan)
st.subheader("Prediksi Berat Sampah Berdasarkan Tahun dan Bulan")
tahun_input = st.number_input("Masukkan tahun", min_value=2024, max_value=2100, value=2025)
bulan_input = st.selectbox("Pilih bulan", list(df.index))

# Buat model per bulan
df_bulanan = df.reset_index().melt(id_vars="Bulan", var_name="Tahun", value_name="Ton")
df_bulanan["Tahun"] = df_bulanan["Tahun"].astype(int)

# Filter bulan yang dipilih
df_bulan_terpilih = df_bulanan[df_bulanan["Bulan"] == bulan_input]

# Regresi Linear untuk bulan terpilih
X_bulan = df_bulan_terpilih["Tahun"].values.reshape(-1, 1)
y_bulan = df_bulan_terpilih["Ton"].values
model_bulan = LinearRegression()
model_bulan.fit(X_bulan, y_bulan)

pred_bulan = model_bulan.predict([[tahun_input]])
st.success(f"Prediksi berat sampah untuk bulan **{bulan_input} {tahun_input}** adalah **{pred_bulan[0]:.2f} ton**")
