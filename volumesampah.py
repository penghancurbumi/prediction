import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Prediksi Total Sampah Kota Sukabumi", layout="wide")

st.title("♻️ Prediksi Total Sampah Kota Sukabumi")
st.markdown("Model prediksi menggunakan **Linear Regression** berdasarkan data tahunan.")

# === Preprocessing Function ===
def preprocess_data(df_raw):
    """
    Membersihkan dan memproses DataFrame mentah dari file CSV.
    """
    df = df_raw.copy()
    df.columns = df.columns.astype(str)

    # Filter hanya baris 'TAHUNAN'
    df = df[df['BULAN'].str.upper().str.strip() == 'TAHUNAN']

    # Drop kolom 'BULAN', lalu transpose
    df = df.drop(columns=['BULAN']).T.reset_index()
    df.columns = ['Tahun', 'Total_Sampah']

    # Bersihkan dan ubah tipe data
    df = df.dropna()
    df = df.drop_duplicates()
    df = df[df['Tahun'].apply(lambda x: str(x).isdigit())]
    df['Tahun'] = df['Tahun'].astype(int)
    df['Total_Sampah'] = df['Total_Sampah'].astype(float)

    # Validasi rentang wajar total sampah (0 - 1.000.000 ton)
    df = df[(df['Total_Sampah'] >= 0) & (df['Total_Sampah'] <= 1_000_000)]

    # Urutkan berdasarkan tahun
    df = df.sort_values('Tahun').reset_index(drop=True)

    return df

# === Load and Clean Data ===
@st.cache_data
def load_data():
    df_raw = pd.read_csv("data sampah kota sukabumi.csv", sep=";", skiprows=1)
    return preprocess_data(df_raw)

print("=== Data Setelah Preprocessing ===")
print(data_tahun)

data_tahun = load_data()

# === Modeling ===
X = data_tahun[['Tahun']]
y = data_tahun['Total_Sampah']

tahun_min = data_tahun['Tahun'].min()
tahun_max = data_tahun['Tahun'].max()
tahun_pred_all = pd.DataFrame({'Tahun': list(range(tahun_min, 2028))})

linreg = LinearRegression()
linreg.fit(X, y)
y_lin_pred = linreg.predict(X)
y_lin_future = linreg.predict(tahun_pred_all)

mse_lin = mean_squared_error(y, y_lin_pred)
r2_lin = r2_score(y, y_lin_pred)

# === Output Evaluasi ===
st.subheader("📊 Evaluasi Model pada Data Historis")
st.write(f"Linear Regression — MSE: {mse_lin:.2f}, R²: {r2_lin:.4f}")

# === Prediksi Tahun Input User ===
st.subheader("🔮 Prediksi Total Sampah per Tahun")
tahun_input = st.number_input(
    "Masukkan Tahun untuk Prediksi", 
    min_value=tahun_min, max_value=2030, value=tahun_max + 1, step=1
)

tahun_input_df = pd.DataFrame({'Tahun': [tahun_input]})
pred_lin = linreg.predict(tahun_input_df)[0]

st.markdown(f"### Hasil Prediksi Tahun {tahun_input}:")
st.write(f"- Linear Regression: **{pred_lin:.2f} ton**")

# === Visualisasi ===
st.subheader("📈 Grafik Prediksi Linear Regression")
fig, ax = plt.subplots(figsize=(12, 6))

ax.scatter(X, y, color='black', label='Data Asli')
ax.plot(tahun_pred_all, y_lin_future, linestyle='--', color='red', label='Linear Regression')
ax.scatter(tahun_input, pred_lin, color='red', s=100, zorder=5)
ax.annotate(f"{pred_lin:.1f}", (tahun_input, pred_lin), textcoords="offset points", xytext=(0,10), ha='center', color='red')

ax.set_xlabel('Tahun')
ax.set_ylabel('Total Sampah (ton)')
ax.set_title('Prediksi Total Sampah Kota Sukabumi per Tahun')
ax.grid(True)
ax.legend()

st.pyplot(fig)
