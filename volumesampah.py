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
    df.columns = ['Tahun', 'Total_Sampah_Tahunan(Ton)']
    df['Tahun'] = df['Tahun'].astype(int)
    df['Total_Sampah_Tahunan(Ton)'] = df['Total_Sampah_Tahunan(Ton)'].astype(float)
    return df

df = load_data()

X = df[['Tahun']]
y = df['Total_Sampah_Tahunan(Ton)']

# Tahun prediksi dari tahun min sampai 2027
tahun_min = df['Tahun'].min()
tahun_max = df['Tahun'].max()
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

# Visualisasi
st.subheader("📈 Grafik Prediksi Linear Regression")

fig, ax = plt.subplots(figsize=(12, 6))

# Plot data asli
ax.scatter(X, y, color='black', label='Data Asli')

# Plot prediksi masa depan
ax.plot(tahun_pred_all, y_lin_future, linestyle='--', color='red', label='Linear Regression')

# Highlight prediksi tahun input user
ax.scatter(tahun_input, pred_lin, color='red', s=100, zorder=5)
ax.annotate(f"{pred_lin:.1f}", (tahun_input, pred_lin), textcoords="offset points", xytext=(0,10), ha='center', color='red')

ax.set_xlabel('Tahun')
ax.set_ylabel('Total Sampah (ton)')
ax.set_title('Prediksi Total Sampah Kota Sukabumi per Tahun')
ax.grid(True)
ax.legend()

st.pyplot(fig)

st.subheader("📊 Diagram Batang Total Sampah per Tahun")

# Gabungkan data historis dengan prediksi tahun input user
data_batang = data_tahun.copy()
if tahun_input not in data_batang['Tahun'].values:
    data_batang = pd.concat([
        data_batang,
        pd.DataFrame({'Tahun': [tahun_input], 'Total_Sampah_Tahunan(Ton)': [pred_lin]})
    ], ignore_index=True)

# Urutkan berdasarkan tahun agar grafik rapi
data_batang = data_batang.sort_values(by='Tahun')

fig_bar, ax_bar = plt.subplots(figsize=(12, 6))

# Bar biasa untuk data historis
historis = data_batang['Tahun'] != tahun_input
ax_bar.bar(
    data_batang['Tahun'][historis],
    data_batang['Total_Sampah_Tahunan(Ton)'][historis],
    color='green',
    label='Data Historis'
)

# Bar khusus untuk data input user (prediksi)
ax_bar.bar(
    data_batang['Tahun'][~historis],
    data_batang['Total_Sampah_Tahunan(Ton)'][~historis],
    color='red',
    label=f'Prediksi {tahun_input}'
)

ax_bar.set_xlabel('Tahun')
ax_bar.set_ylabel('Total Sampah (ton)')
ax_bar.set_title('Diagram Batang Total Sampah Kota Sukabumi')
ax_bar.grid(True, axis='y')
ax_bar.legend()

st.pyplot(fig_bar)

