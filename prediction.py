import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import re

# Mengatur layout halaman menjadi 'wide' untuk tampilan yang lebih baik
st.set_page_config(layout="wide")

st.title("Analisis Regresi Linear Berat Sampah Kota Sukabumi")

@st.cache_data
def load_data():
    """
    Fungsi untuk memuat dan membersihkan data berat sampah dari file teks.
    Fungsi ini akan memisahkan data bulanan dan data total tahunan (baris 'TAHUNAN').
    Jika baris 'TAHUNAN' tidak ditemukan, total tahunan akan dihitung dari data bulanan.
    """
    # Membaca file mentah
    with open("data-sampah-kota-sukabumi (1).txt", "r", encoding="utf-8") as f:
        raw_data = f.read()

    # Memisahkan baris-baris data dan menghapus baris kosong
    lines = [line.strip() for line in raw_data.strip().split("\n") if line.strip()]
    
    # Mendefinisikan nama-nama kolom tahun berdasarkan struktur data
    years = ["2017", "2018", "2020", "2021", "2022", "2023"]
    
    monthly_records = []
    explicit_yearly_total_found = False # Flag untuk menandakan apakah baris 'TAHUNAN' ditemukan

    # Melewati 2 baris header awal
    for line in lines[2:]:
        # Memisahkan bagian-bagian baris menggunakan regex untuk spasi ganda atau tab
        parts = re.split(r'\s{2,}', line.strip())
        
        if not parts or len(parts) < 2:
            continue

        month_or_type = parts[0].strip()
        raw_numeric_values = [p.strip() for p in parts[1:] if p.strip()]
        
        processed_year_values = []
        for val_str in raw_numeric_values:
            if val_str.count('.') > 1:
                cleaned_val_str = val_str.replace('.', '')
            else:
                cleaned_val_str = val_str
            
            try:
                processed_year_values.append(float(cleaned_val_str))
            except ValueError:
                processed_year_values.append(None)

        if len(processed_year_values) == len(years):
            if month_or_type.upper() == "TAHUNAN":
                # Jika baris 'TAHUNAN' ditemukan, gunakan data ini sebagai total tahunan
                yearly_total_data = dict(zip(years, processed_year_values))
                explicit_yearly_total_found = True
            else:
                monthly_records.append([month_or_type] + processed_year_values)
        else:
            st.warning(f"Melewati baris karena jumlah nilai tahun tidak sesuai atau data tidak valid: {line}")

    # Membuat DataFrame untuk data bulanan
    df_monthly = pd.DataFrame(monthly_records, columns=["Bulan"] + years)
    df_monthly.set_index("Bulan", inplace=True)
    
    # Membuat DataFrame untuk data total tahunan
    if explicit_yearly_total_found:
        # Jika baris 'TAHUNAN' ditemukan di file, gunakan itu
        df_yearly = pd.DataFrame([yearly_total_data]).T.reset_index()
        df_yearly.columns = ["Tahun", "Total Sampah (Ton)"]
        df_yearly["Tahun"] = df_yearly["Tahun"].astype(int)
        st.info("Data total tahunan diambil dari baris 'TAHUNAN' di file asli.")
    else:
        # Jika baris 'TAHUNAN' tidak ditemukan, hitung dari data bulanan
        st.warning("Baris total 'TAHUNAN' tidak ditemukan. Menghitung total sampah per tahun dari data bulanan.")
        # Mengubah kolom tahun menjadi numerik untuk perhitungan sum
        df_monthly_numeric = df_monthly[years].apply(pd.to_numeric, errors='coerce')
        df_yearly = df_monthly_numeric.sum().reset_index()
        df_yearly.columns = ["Tahun", "Total Sampah (Ton)"]
        df_yearly["Tahun"] = df_yearly["Tahun"].astype(int)

    return df_monthly, df_yearly

# Memuat data
df_monthly, df_yearly = load_data()

# Menampilkan data bulanan
st.subheader("Data Berat Sampah per Bulan (Ton)")
st.dataframe(df_monthly)

# Menampilkan data total tahunan
st.subheader("Total Sampah per Tahun")
st.dataframe(df_yearly)

# Seluruh bagian plotting dan regresi sekarang tidak memerlukan kondisi if/else
yearly_for_analysis = df_yearly.copy()

# Grafik Batang Total Sampah per Tahun
st.subheader("Grafik Total Sampah per Tahun")
fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.barplot(data=yearly_for_analysis, x="Tahun", y="Total Sampah (Ton)", ax=ax1, palette="viridis")
ax1.set_title("Total Sampah per Tahun Kota Sukabumi")
ax1.set_xlabel("Tahun")
ax1.set_ylabel("Total Sampah (Ton)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig1)

# Regresi Linear
X = yearly_for_analysis["Tahun"].values.reshape(-1, 1) 
y = yearly_for_analysis["Total Sampah (Ton)"].values

model = LinearRegression()
model.fit(X, y)
preds = model.predict(X)

# Visualisasi Regresi Linear
st.subheader("Visualisasi Regresi Linear")
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.scatter(X, y, label="Data Aktual", color="blue", s=100, edgecolors="black")
ax2.plot(X, preds, color="red", linestyle="--", linewidth=2, label="Garis Regresi")
ax2.set_title("Regresi Linear Berat Sampah per Tahun")
ax2.set_xlabel("Tahun")
ax2.set_ylabel("Total Sampah (Ton)")
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
st.pyplot(fig2)

# Menampilkan persamaan regresi dan R-squared
st.subheader("Model Regresi Linear")
st.markdown(f"Persamaan Regresi: $Y = {model.coef_[0]:.2f}X + {model.intercept_:.2f}$")
st.write(f"Koefisien Regresi (Slope): {model.coef_[0]:.2f} (Setiap peningkatan 1 tahun, total sampah diprediksi meningkat sebesar {model.coef_[0]:.2f} ton)")
st.write(f"Intercept: {model.intercept_:.2f} (Total sampah diprediksi pada Tahun 0, yang mungkin tidak relevan secara fisik dalam konteks ini)")
st.write(f"R-squared: {model.score(X, y):.4f} (Proporsi varians dalam total sampah yang dapat dijelaskan oleh tahun. Semakin mendekati 1, semakin baik model.)")

# Prediksi Berat Sampah Tahun Berikutnya
st.subheader("Prediksi Berat Sampah Tahun Berikutnya")
latest_year = int(yearly_for_analysis["Tahun"].max())
tahun_pred = st.number_input(
    "Masukkan tahun untuk prediksi", 
    min_value=latest_year + 1,
    max_value=2100, 
    value=latest_year + 1
)
hasil_pred = model.predict([[tahun_pred]])
st.info(f"Prediksi total sampah pada tahun **{tahun_pred}** adalah **{hasil_pred[0]:,.2f}** ton.")

# Interpretasi Hasil
st.subheader("Interpretasi Hasil")
st.markdown("""
Berdasarkan analisis regresi linear, kita dapat melihat tren perubahan berat sampah dari tahun ke tahun.
* **Grafik Batang:** Menunjukkan visualisasi total berat sampah per tahun dari data yang tersedia, memberikan gambaran cepat tentang distribusi sampah dari tahun ke tahun.
* **Visualisasi Regresi Linear:** Menampilkan titik-titik data aktual dan garis regresi yang merepresentasikan tren linier data. Garis ini menunjukkan hubungan antara tahun dan total berat sampah.
* **Koefisien Regresi (Slope):** Menjelaskan seberapa besar perubahan total sampah (dalam ton) untuk setiap peningkatan satu tahun. Nilai positif menunjukkan tren peningkatan berat sampah seiring waktu, sedangkan nilai negatif menunjukkan penurunan.
* **Intercept:** Merupakan nilai prediksi total sampah ketika tahun adalah nol. Secara fisik, ini mungkin tidak memiliki interpretasi langsung dalam konteks berat sampah dari waktu ke waktu.
* **R-squared:** Merupakan metrik statistik yang menunjukkan proporsi varians dalam variabel dependen (Total Sampah) yang dapat diprediksi dari variabel independen (Tahun). Semakin mendekati 1, semakin baik model regresi dalam menjelaskan variasi data dan memprediksi nilai masa depan.
""")
