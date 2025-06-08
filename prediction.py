import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import re

st.title("Analisis Regresi Linear Berat Sampah Kota Sukabumi")

# Load data mentah
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
    df.set_index("Bulan", inplace=True)
    return df

df = load_data()

# Tampilkan data bulanan per tahun (transpos)
st.subheader("Data Berat Sampah per Bulan per Tahun (Transpos)")
df_transposed = df.transpose()
st.dataframe(df_transposed)

# Tampilkan data asli per bulan (semua tahun)
st.subheader("Data Berat Sampah per Bulan (Semua Tahun)")
st.dataframe(df)

# Grafik garis: tren bulanan per tahun
st.subheader("Grafik Tren Berat Sampah Bulanan per Tahun")
fig3, ax3 = plt.subplots()
for tahun in df.columns:
    ax3.plot(df.index, df[tahun], marker='o', label=tahun)
ax3.set_xlabel("Bulan")
ax3.set_ylabel("Berat Sampah (Ton)")
ax3.set_title("Tren Berat Sampah Bulanan per Tahun")
ax3.legend()
plt.xticks(rotation=45)
st.pyplot(fig3)

# Pilih tahun & tampilkan data bulanan
st.subheader("Lihat Data Bulanan untuk Tahun Tertentu")
tahun_dipilih = st.selectbox("Pilih Tahun", df.columns.tolist())
st.write(f"Berat Sampah Bulanan di Tahun {tahun_dipilih}:")
st.dataframe(df[[tahun_dipilih]])

# Total tahunan
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

# Prediksi berdasarkan input user
st.subheader("Prediksi Berat Sampah Berdasarkan Tahun dan Bulan")
tahun_input = st.number_input("Masukkan tahun", min_value=2024, max_value=2100, value=2025)
bulan_input = st.selectbox("Pilih bulan", list(df.index))

# Buat model per bulan
df_bulanan = df.reset_index().melt(id_vars="Bulan", var_name="Tahun", value_name="Ton")
df_bulanan["Tahun"] = df_bulanan["Tahun"].astype(int)

# Filter data berdasarkan bulan
df_bulan_terpilih = df_bulanan[df_bulanan["Bulan"] == bulan_input]

# Regresi Linear untuk bulan terpilih
X_bulan = df_bulan_terpilih["Tahun"].values.reshape(-1, 1)
y_bulan = df_bulan_terpilih["Ton"].values
model_bulan = LinearRegression()
model_bulan.fit(X_bulan, y_bulan)

# Prediksi untuk input tahun
pred_bulan = model_bulan.predict([[tahun_input]])
st.success(f"Prediksi berat sampah untuk bulan **{bulan_input} {tahun_input}** adalah **{pred_bulan[0]:.2f} ton**")

# Visualisasi regresi linear untuk bulan yang dipilih
st.subheader(f"Grafik Regresi Linear Sampah Bulanan - {bulan_input}")

fig4, ax4 = plt.subplots()
ax4.scatter(X_bulan, y_bulan, color="blue", label="Data Aktual")
ax4.plot(X_bulan, model_bulan.predict(X_bulan), color="red", label="Regresi Linear")
ax4.scatter(tahun_input, pred_bulan[0], color="green", s=100, label="Prediksi")
ax4.set_xlabel("Tahun")
ax4.set_ylabel("Berat Sampah (Ton)")
ax4.set_title(f"Regresi Linear Berat Sampah Bulan {bulan_input}")
ax4.legend()
st.pyplot(fig4)
