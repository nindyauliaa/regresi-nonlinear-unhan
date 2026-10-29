
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

# =========================================================
# KONFIGURASI
# =========================================================
st.set_page_config(
    page_title="Regresi Nonlinear - UNHAN RI",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
:root{--navy:#062b55;--blue:#0b67c2;--blue-soft:#eaf4ff;--red:#c91f2b;--red-soft:#fff0f2;--ink:#14213d;--muted:#68758a;--line:#dfe7f0}
.stApp{background:#f5f8fc;color:var(--ink)}
.block-container{padding-top:1.2rem!important;padding-bottom:2rem!important;max-width:1500px!important}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#041d3d 0%,#062b55 65%,#071f3d 100%);border-right:1px solid rgba(255,255,255,.08)}
[data-testid="stSidebar"] *{color:#f7fbff}
.unhan-brand{text-align:center;padding:4px 6px 14px}
.unhan-brand-title{font-size:22px;font-weight:800;letter-spacing:.5px;margin-top:7px}
.unhan-brand-subtitle{font-size:11px;line-height:1.35;color:#d5e5f8!important}
.sidebar-section{border-top:1px solid rgba(255,255,255,.12);margin-top:10px;padding-top:16px}
.sidebar-label{font-size:13px;font-weight:700;color:#fff!important;margin-bottom:7px}
.sidebar-source{background:rgba(13,103,194,.22);border:1px solid rgba(92,171,255,.24);border-radius:12px;padding:10px 12px;margin-top:10px;color:#eaf5ff!important;font-size:12px}
.unhan-hero{position:relative;overflow:hidden;min-height:160px;border-radius:20px;padding:28px 34px;margin-bottom:22px;background:linear-gradient(110deg,rgba(255,255,255,.98),rgba(238,246,255,.98) 58%,rgba(222,237,253,.92));border:1px solid #d8e5f2;box-shadow:0 8px 28px rgba(6,43,85,.09)}
.unhan-hero:before{content:"";position:absolute;left:0;top:0;width:115px;height:100%;background:linear-gradient(135deg,var(--red) 0 18%,transparent 18% 30%,var(--navy) 30% 53%,transparent 53%)}
.unhan-hero:after{content:"";position:absolute;right:-70px;top:-70px;width:250px;height:250px;border-radius:50%;background:radial-gradient(circle,rgba(11,103,194,.16),transparent 68%)}
.hero-content{position:relative;z-index:2;margin-left:105px}
.hero-kicker{color:var(--red);font-size:12px;font-weight:800;letter-spacing:2px;text-transform:uppercase}
.hero-title{color:var(--navy);font-size:38px;font-weight:850;line-height:1.1;margin:0}
.hero-subtitle{color:#53657b;font-size:15px;margin-top:8px}
.hero-university{display:inline-block;margin-top:14px;padding:7px 15px;border-radius:999px;background:rgba(255,255,255,.82);border:1px solid #d9e5f2;color:var(--navy);font-size:11px;font-weight:800;letter-spacing:2px}
.section-title{color:var(--navy);font-size:25px;font-weight:800;margin:20px 0 10px}
.section-caption{color:var(--muted);font-size:13px;margin-top:-5px;margin-bottom:14px}
div[data-testid="stMetric"]{background:#fff;border:1px solid var(--line);border-radius:16px;padding:16px 18px;box-shadow:0 5px 18px rgba(6,43,85,.06)}
div[data-testid="stMetricLabel"]{color:#607087!important;font-weight:700!important}
div[data-testid="stMetricValue"]{color:var(--navy)!important;font-weight:850!important}
.chart-card{background:#fff;border:1px solid #dce6f0;border-top:4px solid var(--red);border-radius:18px;padding:18px 20px 10px;box-shadow:0 7px 22px rgba(6,43,85,.06);margin-top:8px}
.chart-title{color:var(--navy);font-size:22px;font-weight:800}.chart-caption{color:var(--muted);font-size:13px}
.stButton>button,.stDownloadButton>button{border-radius:10px;font-weight:700}
button[data-baseweb="tab"]{font-weight:700}button[data-baseweb="tab"][aria-selected="true"]{color:var(--red)!important}
.unhan-footer{margin-top:35px;padding:18px;border-top:1px solid #dbe4ee;text-align:center;color:#718096;font-size:11px}
.footer-red{color:var(--red);font-weight:800}.footer-blue{color:var(--navy);font-weight:800}

/* ===== CSV UPLOADER — SIDEBAR ===== */
[data-testid="stFileUploaderDropzone"] {
    background: #0b3d73 !important;
    border: 1px solid rgba(123, 190, 255, .45) !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploaderDropzone"] * {
    color: #ffffff !important;
}
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] p {
    color: #eaf4ff !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: #ffffff !important;
    color: #062b55 !important;
    border: 1px solid #dbe7f3 !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background: #eaf4ff !important;
    color: #062b55 !important;
}


/* ===== USER PROFILE ===== */
.user-profile {
    position: fixed;
    top: 18px;
    right: 62px;
    z-index: 999999;
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 8px 15px 8px 10px;
    border-radius: 999px;
    background: rgba(255,255,255,.97);
    border: 1px solid #d5e1ee;
    box-shadow: 0 5px 18px rgba(6,43,85,.12);
    color: #062b55 !important;
    font-size: 13px;
    font-weight: 800;
}
.user-avatar {
    width: 29px;
    height: 29px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg,#c91f2b,#e84b59);
    color: #ffffff !important;
    font-size: 11px;
    font-weight: 850;
}
.user-role {
    display: block;
    font-size: 9px;
    color: #718096 !important;
    font-weight: 600;
    margin-top: 1px;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div class="unhan-hero">
        <div class="hero-content">
            <div class="hero-kicker">UNIVERSITAS PERTAHANAN REPUBLIK INDONESIA</div>
            <div class="hero-title">Analisis Regresi Nonlinear</div>
            <div class="hero-subtitle">Perbandingan 7 model regresi menggunakan data real NIST Rat43</div>
            <div class="hero-university">REGRESI NONLINEAR • SEMESTER 7 • UNHAN RI</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# MODEL
# =========================================================
def linear(x, a, b):
    return a + b * x

def quadratic(x, a, b, c):
    return a + b*x + c*x**2

def exponential(x, a, b):
    return a * np.exp(np.clip(b*x, -700, 700))

def logistic(x, L, k, x0):
    z = np.clip(-k*(x-x0), -700, 700)
    return L / (1 + np.exp(z))

def saturation(x, L, K):
    return L*x / (K+x)

def saturated_exp(x, L, k, c):
    return c + L*(1-np.exp(np.clip(-k*x, -700, 700)))

def saturated_hyperbola(x, L, K, c):
    return c + L*x/(K+x)

MODELS = {
    "Linear": (linear, "y = a + bx"),
    "Kuadratik": (quadratic, "y = a + bx + cx²"),
    "Eksponensial": (exponential, "y = ae^(bx)"),
    "Logistik": (logistic, "y = L/[1 + e^(-k(x-x₀))]"),
    "Saturasi": (saturation, "y = Lx/(K+x)"),
    "Jenuh 1": (saturated_exp, "y = c + L(1-e^(-kx))"),
    "Jenuh 2": (saturated_hyperbola, "y = c + Lx/(K+x)")
}

# =========================================================
# FIT MODEL
# =========================================================
def fit_model(name, x, y):
    fun = MODELS[name][0]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    xspan = max(xmax-xmin, 1.0)
    xmid = (xmin+xmax)/2
    yrange = max(ymax-ymin, 1.0)

    if name == "Linear":
        b, a = np.polyfit(x, y, 1)
        p = np.array([a, b])

    elif name == "Kuadratik":
        c, b, a = np.polyfit(x, y, 2)
        p = np.array([a, b, c])

    elif name == "Eksponensial":
        p, _ = curve_fit(
            fun, x, y,
            p0=[max(ymin, 1e-3), 0.1],
            bounds=([1e-10, -10], [max(ymax*100, 1), 10]),
            maxfev=200000
        )

    elif name == "Logistik":
        p, _ = curve_fit(
            fun, x, y,
            p0=[max(ymax*1.1, 1), 0.5, xmid],
            bounds=(
                [max(ymax*0.5, 1e-6), 1e-8, xmin-10*xspan],
                [max(ymax*100, 1), 20, xmax+10*xspan]
            ),
            maxfev=200000
        )

    elif name == "Saturasi":
        p, _ = curve_fit(
            fun, x, y,
            p0=[max(ymax*1.2, 1), xmid],
            bounds=([0, 1e-8], [max(ymax*100, 1), np.inf]),
            maxfev=200000
        )

    elif name == "Jenuh 1":
        p, _ = curve_fit(
            fun, x, y,
            p0=[yrange*1.2, 0.1, ymin],
            bounds=(
                [0, 0, -max(abs(ymax)*2, 1)],
                [max(ymax*100, 1), 20, max(abs(ymax)*2, 1)]
            ),
            maxfev=200000
        )

    else:
        p, _ = curve_fit(
            fun, x, y,
            p0=[yrange*1.2, xmid, ymin],
            bounds=(
                [0, 1e-8, -max(abs(ymax)*2, 1)],
                [max(ymax*100, 1), np.inf, max(abs(ymax)*2, 1)]
            ),
            maxfev=200000
        )

    yhat = fun(x, *p)
    residual = y-yhat
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    sse = float(np.sum(residual**2))
    sst = float(np.sum((y-y.mean())**2))
    r2 = float(1-sse/sst) if sst > 0 else np.nan

    return p, yhat, rmse, mae, r2

# =========================================================
# DATA
# =========================================================
def load_default_data():
    candidates = [
        Path(__file__).resolve().parent.parent / "DATA" / "data_real_Rat43_NIST.csv",
        Path(__file__).resolve().parent / "data_real_Rat43_NIST.csv",
        Path.cwd() / "DATA" / "data_real_Rat43_NIST.csv",
    ]

    for p in candidates:
        if p.exists():
            return pd.read_csv(p), p

    # fallback agar aplikasi tetap bisa dibuka
    data = pd.DataFrame({
        "x": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
        "y": [16.08,33.83,65.80,97.20,191.55,326.20,386.87,
              520.53,590.03,651.92,724.93,699.56,689.96,637.56,717.41]
    })
    return data, None

default_df, default_path = load_default_data()

# =========================================================
# SIDEBAR
# =========================================================
logo_candidates = [
    Path(__file__).resolve().parent / "assets" / "logo_unhan.png",
    Path(__file__).resolve().parent / "logo_unhan.png",
    Path(__file__).resolve().parent.parent / "DATA" / "logo_unhan.png",
]
logo_path = next((p for p in logo_candidates if p.exists()), None)

if logo_path is not None:
    st.sidebar.image(str(logo_path), width=145)

st.sidebar.markdown(
    """
    <div class="unhan-brand">
        <div class="unhan-brand-title">UNHAN RI</div>
        <div class="unhan-brand-subtitle">UNIVERSITAS PERTAHANAN<br>REPUBLIK INDONESIA</div>
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown(
    '<div class="sidebar-section"><div class="sidebar-label">⚙️ PENGATURAN DATA</div></div>',
    unsafe_allow_html=True
)

uploaded = st.sidebar.file_uploader(
    "Upload CSV sendiri",
    type=["csv"]
)

if uploaded is not None:
    df = pd.read_csv(uploaded)
    source = "CSV yang di-upload"
else:
    df = default_df.copy()
    source = "Dataset NIST Rat43"

if "x" not in df.columns or "y" not in df.columns:
    st.error("CSV harus memiliki kolom bernama 'x' dan 'y'.")
    st.stop()

df["x"] = pd.to_numeric(df["x"], errors="coerce")
df["y"] = pd.to_numeric(df["y"], errors="coerce")
df = df.dropna(subset=["x", "y"]).sort_values("x").reset_index(drop=True)

st.sidebar.markdown(
    f'<div class="sidebar-source">🗄️ <b>Sumber:</b> {source}</div>',
    unsafe_allow_html=True
)
st.sidebar.markdown(
    f"""
    <div style="margin-top:12px;background:rgba(255,255,255,.08);
    border:1px solid rgba(255,255,255,.12);border-radius:12px;padding:12px;">
        <div style="font-size:11px;color:#cfe0f4;">JUMLAH OBSERVASI</div>
        <div style="font-size:28px;font-weight:850;color:white;">{len(df)}</div>
        <div style="font-size:11px;color:#b9cbe0;">data</div>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# HEADER DATA
# =========================================================
st.markdown('<div class="section-title">📊 Data yang Digunakan</div><div class="section-caption">Ringkasan dataset yang sedang dianalisis.</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
col1.metric("Observasi", len(df))
col2.metric("X minimum", f"{df.x.min():.2f}")
col3.metric("Y maksimum", f"{df.y.max():.2f}")

st.info(
    "Dataset default adalah Rat43 dari NIST Statistical Reference Datasets. "
    "Data digunakan untuk melihat pola pertumbuhan nonlinear dan membandingkan "
    "7 bentuk model."
)

with st.expander("Lihat tabel data"):
    st.dataframe(df, width="stretch")

# =========================================================
# PLOT DATA
# =========================================================
st.markdown('<div class="section-title">📈 Pola Data</div><div class="section-caption">Visualisasi hubungan antara variabel x dan y pada data observasi.</div>', unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(df["x"], df["y"], s=65, label="Data observasi")
ax.plot(df["x"], df["y"], alpha=0.35)
ax.set_xlabel("x — waktu pertumbuhan")
ax.set_ylabel("y — berat kering")
ax.set_title("Pola Data Rat43")
ax.grid(alpha=0.25)
ax.legend()
st.pyplot(fig, width="stretch")
plt.close(fig)

# =========================================================
# FIT SEMUA MODEL
# =========================================================
results = {}
errors = []

for name in MODELS:
    try:
        p, yhat, rmse, mae, r2 = fit_model(name, df["x"].values, df["y"].values)
        results[name] = {
            "params": p,
            "yhat": yhat,
            "RMSE": rmse,
            "MAE": mae,
            "R²": r2
        }
    except Exception as e:
        errors.append(f"{name}: {e}")

# =========================================================
# PERBANDINGAN
# =========================================================
st.markdown('<div class="section-title">🔎 Perbandingan 7 Model</div><div class="section-caption">Evaluasi model berdasarkan RMSE, MAE, dan R².</div>', unsafe_allow_html=True)

rows = []
for name, result in results.items():
    rows.append({
        "Model": name,
        "RMSE": result["RMSE"],
        "MAE": result["MAE"],
        "R²": result["R²"],
        "Persamaan": MODELS[name][1]
    })

comparison = pd.DataFrame(rows)
comparison = comparison.sort_values("RMSE").reset_index(drop=True)

st.dataframe(
    comparison.style.format({
        "RMSE": "{:.4f}",
        "MAE": "{:.4f}",
        "R²": "{:.4f}"
    }),
    width="stretch"
)

if errors:
    with st.expander("Model yang gagal fitting"):
        for e in errors:
            st.warning(e)

# =========================================================
# KURVA 7 MODEL
# =========================================================
st.markdown('<div class="section-title">🎨 Visualisasi 7 Model</div><div class="section-caption">Perbandingan bentuk kurva hasil fitting pada dataset yang sama.</div>', unsafe_allow_html=True)

xgrid = np.linspace(df.x.min(), df.x.max(), 500)

fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(df.x, df.y, s=60, label="Data observasi", zorder=10)

for name, result in results.items():
    fun = MODELS[name][0]
    try:
        curve = fun(xgrid, *result["params"])
        ax.plot(
            xgrid,
            curve,
            linewidth=2,
            label=f"{name} | RMSE={result['RMSE']:.2f}"
        )
    except Exception:
        pass

ax.set_xlabel("x — waktu pertumbuhan")
ax.set_ylabel("y — berat kering")
ax.set_title("Perbandingan Kurva 7 Model Regresi")
ax.grid(alpha=0.25)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
st.pyplot(fig, width="stretch")
plt.close(fig)

# =========================================================
# DETAIL MODEL
# =========================================================
st.markdown('<div class="section-title">🧮 Persamaan dan Parameter</div><div class="section-caption">Parameter hasil fitting untuk setiap model.</div>', unsafe_allow_html=True)

for name, result in results.items():
    p = result["params"]

    with st.expander(name):
        st.write(f"**Bentuk model:** {MODELS[name][1]}")
        st.write("**Parameter hasil fitting:**")
        for i, value in enumerate(p, start=1):
            st.write(f"Parameter {i}: `{value:.6f}`")

        st.write(
            f"**RMSE:** {result['RMSE']:.6f}  |  "
            f"**MAE:** {result['MAE']:.6f}  |  "
            f"**R²:** {result['R²']:.6f}"
        )

# =========================================================
# PREDIKSI
# =========================================================
st.markdown('<div class="section-title">🔮 Prediksi dengan 7 Model</div><div class="section-caption">Masukkan nilai x untuk melihat prediksi dari seluruh model.</div>', unsafe_allow_html=True)

xmin = float(df.x.min())
xmax = float(df.x.max())

x_new = st.number_input(
    "Masukkan nilai x yang ingin diprediksi",
    min_value=float(xmin),
    max_value=float(xmax),
    value=float((xmin+xmax)/2),
    step=0.1
)

pred_rows = []

for name, result in results.items():
    fun = MODELS[name][0]
    prediction = float(fun(np.array([x_new]), *result["params"])[0])

    pred_rows.append({
        "Model": name,
        "Prediksi y": prediction,
        "RMSE": result["RMSE"]
    })

pred_df = pd.DataFrame(pred_rows)
st.dataframe(
    pred_df.style.format({
        "Prediksi y": "{:.4f}",
        "RMSE": "{:.4f}"
    }),
    width="stretch"
)

# =========================================================
# KESIMPULAN PEMBELAJARAN
# =========================================================
st.markdown('<div class="section-title">📚 Interpretasi</div><div class="section-caption">Penjelasan singkat karakteristik masing-masing model.</div>', unsafe_allow_html=True)

st.markdown("""
**Linear** digunakan ketika hubungan x-y mendekati garis lurus.

**Kuadratik** dapat menggambarkan pola melengkung, tetapi bentuknya tidak
secara khusus memiliki batas atas.

**Eksponensial** menggambarkan pertumbuhan atau penurunan yang mengikuti
laju eksponensial.

**Logistik** cocok untuk pola sigmoid: pertumbuhan lambat → cepat →
melambat menuju batas.

**Saturasi** menggambarkan kenaikan yang semakin lambat ketika mendekati
nilai maksimum.

**Jenuh 1** menggunakan bentuk eksponensial menuju plateau.

**Jenuh 2** menggunakan bentuk hiperbola menuju plateau.

RMSE digunakan untuk mengukur besar error fitting. Semakin kecil RMSE,
semakin kecil selisih rata-rata kuadrat antara data observasi dan hasil
prediksi model pada dataset tersebut.
""")

st.markdown(
    """
    <div class="unhan-footer">
        <span class="footer-blue">UNIVERSITAS PERTAHANAN REPUBLIK INDONESIA</span>
        &nbsp;•&nbsp;
        <span class="footer-red">ANALISIS REGRESI NONLINEAR</span>
        &nbsp;•&nbsp; Dataset NIST Rat43
    </div>
    """,
    unsafe_allow_html=True
)
