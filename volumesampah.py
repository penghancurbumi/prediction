import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Prediksi Total Sampah Kota Sukabumi", layout="wide")

st.title("♻️ Prediksi Total Sampah Kota Sukabumi")
st.markdown("Model prediksi menggunakan **Linear Regression** berdasarkan data tahunan.")

# === Load Data ===
@st.cache_data
def load_data():
    df = pd.read_csv("data sampah kota sukabumi.csv", sep=";", skiprows=1)
    df.columns = df.columns.astype(str)
    baris_tahunan = df[df['BULAN'].str.upper().str.strip() == 'TAHUNAN']
    data_tahun = baris_tahunan.drop(columns=['BULAN']).T.reset_index()
    data_tahun.columns = ['Tahun', 'Total_Sampah']
    
    # Preprocessing
    data_tahun = data_tahun.dropna()
    data_tahun = data_tahun[data_tahun['Tahun'].str.isdigit()]  # Pastikan Tahun bisa dikonversi ke int
    data_tahun['Tahun'] = data_tahun['Tahun'].astype(int)
    data_tahun['Total_Sampah'] = data_tahun['Total_Sampah'].astype(float)
    data_tahun = data_tahun.sort_values('Tahun').reset_index(drop=True)
    
    return data_tahun

data_tahun = load_data()

# === Cek Nilai Kosong ===
st.subheader("🧼 Cek Nilai Kosong Setelah Preprocessing")
st.write(data_tahun.isnull().sum())

# === Model & Prediksi ===
X = data_tahun[['Tahun']]
y = data_tahun['Total_Sampah']

# Tahun prediksi dari tahun min sampai 2027
tahun_min = data_tahun['Tahun'].min()
tahun_max = data_tahun['Tahun'].max()
tahun_pred_all = pd.DataFrame({'Tahun': list(range(tahun_min, 2028))})

# Model Linear Regression
linreg = LinearRegression()
linreg.fit(X, y)
y_lin_pred = linreg.predict(X)
y_lin_future = linreg.predict(tahun_pred_all)

# Evaluasi model
mse_lin = mean_squared_error(y, y_lin_pred)
r2_lin = r2_score(y, y_lin_pred)

# Tampilkan evaluasi
st.subheader("📊 Evaluasi Model pada Data Historis")
st.write(f"Linear Regression — MSE: {mse_lin:.2f}, R²: {r2_lin:.4f}")

# Input tahun untuk prediksi
st.subheader("🔮 Prediksi Total Sampah per Tahun")
tahun_input = st.number_input(
    "Masukkan Tahun untuk Prediksi", 
    min_value=tahun_min, max_value=2030, value=tahun_max + 1, step=1)

tahun_input_df = pd.DataFrame({'Tahun': [tahun_input]})
pred_lin = linreg.predict(tahun_input_df)[0]

st.markdown(f"### Hasil Prediksi Tahun {tahun_input}:")
st.write(f"- Linear Regression: **{pred_lin:.2f} ton**")

# Buat salinan data historis + tambahkan data prediksi user
data_bar = data_tahun.copy()
if tahun_input not in data_bar['Tahun'].values:
    data_pred_input = pd.DataFrame({'Tahun': [tahun_input], 'Total_Sampah': [pred_lin]})
    data_bar = pd.concat([data_bar, data_pred_input], ignore_index=True)
    data_bar = data_bar.sort_values('Tahun').reset_index(drop=True)

# === Diagram Batang Data Historis + Tahun Prediksi User ===
st.subheader("📊 Diagram Batang Data Historis + Prediksi")

fig_bar, ax_bar = plt.subplots(figsize=(12, 6))

colors = ['skyblue' if tahun != tahun_input else 'salmon' for tahun in data_bar['Tahun']]

ax_bar.bar(data_bar['Tahun'], data_bar['Total_Sampah'], color=colors)
ax_bar.set_xlabel('Tahun')
ax_bar.set_ylabel('Total Sampah (ton)')
ax_bar.set_title('Total Sampah Kota Sukabumi per Tahun')
ax_bar.grid(True)

# Highlight nilai prediksi
if tahun_input not in data_tahun['Tahun'].values:
    ax_bar.annotate(f"{pred_lin:.1f}", (tahun_input, pred_lin), textcoords="offset points", xytext=(0,10), ha='center', color='red')

st.pyplot(fig_bar)

# === Grafik Prediksi Linear Regression ===
st.subheader("📈 Grafik Prediksi Linear Regression")

fig_line, ax_line = plt.subplots(figsize=(12, 6))
ax_line.plot(tahun_pred_all['Tahun'], y_lin_future, linestyle='--', color='red', label='Linear Regression')
ax_line.scatter(X, y, color='black', label='Data Asli')
ax_line.scatter(tahun_input, pred_lin, color='red', s=100, zorder=5)
ax_line.annotate(f"{pred_lin:.1f}", (tahun_input, pred_lin), textcoords="offset points", xytext=(0,10), ha='center', color='red')

ax_line.set_xlabel('Tahun')
ax_line.set_ylabel('Total Sampah (ton)')
ax_line.set_title('Prediksi Total Sampah Kota Sukabumi per Tahun')
ax_line.grid(True)
ax_line.legend()

st.pyplot(fig_line)
