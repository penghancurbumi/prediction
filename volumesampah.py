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

# === Visualisasi: Diagram Batang ===
st.subheader("📊 Diagram Batang Total Sampah")
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.bar(data_tahun['Tahun'], data_tahun['Total_Sampah'], color='skyblue', label='Total Sampah')

# Tambahkan prediksi jika belum ada di data
if tahun_input not in data_tahun['Tahun'].values:
    ax2.bar(tahun_input, pred_input, color='orange', label=f'Prediksi {tahun_input}')
    ax2.annotate(f"{pred_input:.1f}", (tahun_input, pred_input), textcoords="offset points", xytext=(0,5), ha='center', color='red')

ax2.set_xlabel("Tahun")
ax2.set_ylabel("Total Sampah (ton)")
ax2.set_title("Total Sampah Kota Sukabumi per Tahun (Diagram Batang)")
ax2.grid(axis='y', linestyle='--', alpha=0.7)
ax2.legend()
st.pyplot(fig2)
