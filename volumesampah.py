import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Prediksi Sampah Sukabumi", layout="wide")

st.title("📊 Prediksi Total Sampah Kota Sukabumi per Tahun")

# === Load dan Preprocessing Data ===
df = pd.read_csv("data sampah kota sukabumi (3).csv", sep=";", skiprows=1)
df.columns = df.columns.astype(str)

# Filter baris 'TAHUNAN' saja
df = df[df['BULAN'].str.upper().str.strip() == 'TAHUNAN']
df = df.drop(columns=['BULAN']).T.reset_index()
df.columns = ['Tahun', 'Total_Sampah']

# Preprocessing
df = df.dropna()
df = df[df['Tahun'].str.isdigit()]
df['Tahun'] = df['Tahun'].astype(int)
df['Total_Sampah'] = df['Total_Sampah'].astype(float)
df = df.sort_values('Tahun').reset_index(drop=True)

data_tahun = df  # simpan untuk diagram batang

st.success("✅ Data berhasil dimuat dan diproses!")

# === Model dan Prediksi ===
X = df[['Tahun']]
y = df['Total_Sampah']

tahun_min = df['Tahun'].min()
tahun_max = df['Tahun'].max()
tahun_pred_all = pd.DataFrame({'Tahun': list(range(tahun_min, 2028))})

model = LinearRegression()
model.fit(X, y)

y_pred_train = model.predict(X)
y_pred_future = model.predict(tahun_pred_all)

# Evaluasi
mse = mean_squared_error(y, y_pred_train)
r2 = r2_score(y, y_pred_train)

st.subheader("📈 Evaluasi Model Linear Regression")
st.write(f"- **MSE (Mean Squared Error):** {mse:.2f}")
st.write(f"- **R² (Koefisien Determinasi):** {r2:.4f}")

# === Prediksi Tahun Tertentu ===
tahun_input = st.number_input("Masukkan tahun yang ingin diprediksi", 
                              min_value=tahun_min, max_value=2030, step=1, value=tahun_max + 1)
tahun_input_df = pd.DataFrame({'Tahun': [tahun_input]})
pred_input = model.predict(tahun_input_df)[0]

st.success(f"📌 Prediksi Total Sampah untuk Tahun {tahun_input}: **{pred_input:.2f} ton**")

# === Visualisasi: Linear Regression ===
st.subheader("📉 Visualisasi Linear Regression")
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.scatter(X, y, color='black', label='Data Asli')
ax1.plot(tahun_pred_all, y_pred_future, color='red', linestyle='--', label='Linear Regression')
ax1.scatter(tahun_input, pred_input, color='red', s=100, zorder=5)
ax1.annotate(f"{pred_input:.1f}", (tahun_input, pred_input), textcoords="offset points", xytext=(0,10), ha='center', color='red')

ax1.set_xlabel("Tahun")
ax1.set_ylabel("Total Sampah (ton)")
ax1.set_title("Prediksi Total Sampah Kota Sukabumi per Tahun")
ax1.grid(True)
ax1.legend()
st.pyplot(fig1)

# === Visualisasi: Diagram Batang Data Historis + Prediksi Beberapa Tahun ===
st.subheader("📊 Diagram Batang: Total Sampah Historis dan Prediksi")

# Prediksi semua tahun dari tahun terakhir data hingga 2030
tahun_terakhir = data_tahun['Tahun'].max()
tahun_prediksi = pd.DataFrame({'Tahun': list(range(tahun_terakhir + 1, 2031))})
y_pred_future = model.predict(tahun_prediksi)

# Gabungkan data historis + prediksi
df_pred = pd.DataFrame({
    'Tahun': tahun_prediksi['Tahun'],
    'Total_Sampah': y_pred_future
})
df_pred['Status'] = 'Prediksi'

data_tahun['Status'] = 'Historis'
df_all = pd.concat([data_tahun, df_pred], ignore_index=True)

# Visualisasi
fig3, ax3 = plt.subplots(figsize=(12, 6))
colors = df_all['Status'].map({'Historis': 'skyblue', 'Prediksi': 'orange'})
bars = ax3.bar(df_all['Tahun'], df_all['Total_Sampah'], color=colors)

# Tambahkan garis batas prediksi
ax3.axvline(x=tahun_terakhir + 0.5, color='gray', linestyle='--', linewidth=1)
ax3.text(tahun_terakhir + 0.6, ax3.get_ylim()[1]*0.95, "Mulai Prediksi →", color='gray')

ax3.set_xlabel("Tahun")
ax3.set_ylabel("Total Sampah (ton)")
ax3.set_title("Diagram Batang: Total Sampah Historis dan Prediksi")
ax3.grid(axis='y', linestyle='--', alpha=0.7)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='skyblue', label='Data Historis'),
    Patch(facecolor='orange', label='Data Prediksi')
]
ax3.legend(handles=legend_elements)

st.pyplot(fig3)
