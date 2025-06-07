import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import re
import calendar

st.title("Analisis Regresi Linear Berat Sampah Kota Sukabumi")

# Load data mentah
@st.cache_data
def load_data():
    with open("data-sampah-kota-sukabumi.txt", "r", encoding="utf-8") as f:
        raw = f.read()

    lines = [line.strip() for line in raw.strip().split("\n") if line.strip()]
    cleaned_rows = []

    # Ambil nama bulan dari modul calendar
    nama_bulan = [calendar.month_name[i] for i in range(1, 13)]

    # Ambil tahun dari baris kedua (header)
    header_line = lines[1]
    tahun_list = re.findall(r'\d{4}', header_line)
    tahun_columns = [t.strip() for t in tahun_list]

    for line in lines[2:]:
        parts = re.split(r'\s{2,}', line)
        if len(parts) >= 2:
            month = parts[0].strip()
            if month not in nama_bulan:
                continue
            numbers = re.findall(r'\d+\.\d+', line)
            row = [month] + [float(n) for n in numbers]
            # Tambah None jika data tidak lengkap
            while len(row) < len(tahun_columns) + 1:
                row.append(None)
            cleaned_rows.append(row)

    df = pd.DataFrame(cleaned_rows, columns=["Bulan"] + tahun_columns)
    df.set_index("Bulan", inplace=True)
    return df

df = load_data()

# Tampilkan data bulanan per tahun
st.subheader("Data Berat Sampah per Bulan per Tahun (Ton)")
st.dataframe(df)

# Tampilkan total tahunan
yearly = df.sum(numeric_only=True).reset_index()
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
df_bulan_terpilih = df_bulanan[df_bulanan["Bulan"] == bulan_input].dropna()

# Regresi Linear untuk bulan terpilih
X_bulan = df_bulan_terpilih["Tahun"].values.reshape(-1, 1)
y_bulan = df_bulan_terpilih["Ton"].values
model_bulan = LinearRegression()
model_bulan.fit(X_bulan, y_bulan)

pred_bulan = model_bulan.predict([[tahun_input]])
st.success(f"Prediksi berat sampah untuk bulan **{bulan_input} {tahun_input}** adalah **{pred_bulan[0]:.2f} ton**")
