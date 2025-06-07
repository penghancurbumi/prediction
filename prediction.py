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
