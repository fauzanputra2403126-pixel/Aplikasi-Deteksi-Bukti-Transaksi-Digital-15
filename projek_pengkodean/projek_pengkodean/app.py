import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from modules.preprocess import load_image, binarize, read_software_tag
from modules.reference_store import (
    PROVIDER_CONFIGS, MIN_REFS, RECOMMENDED_REFS,
    list_references, save_reference, delete_reference, build_model,
)
from modules.hamming import forensic_compare
from modules.field_forensics import build_field_model, check_fields, final_verdict, FINAL_TITLES

# ---------------------------------------------------------------- Konfigurasi
st.set_page_config(
    page_title="VerifyID - Multi-Bank Transaction Forensics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0b0f19 0%, #111827 100%); color: #f3f4f6; }
    .badge { color: white; padding: 18px 24px; border-radius: 14px; font-weight: 700;
             font-size: 26px; text-align: center; margin-bottom: 24px; }
    .badge-asli  { background: linear-gradient(135deg,#059669,#10b981); box-shadow: 0 4px 25px rgba(16,185,129,.45); }
    .badge-warn  { background: linear-gradient(135deg,#d97706,#f59e0b); box-shadow: 0 4px 25px rgba(245,158,11,.45); }
    .badge-palsu { background: linear-gradient(135deg,#dc2626,#ef4444); box-shadow: 0 4px 25px rgba(239,68,68,.45); }
</style>
""", unsafe_allow_html=True)

BADGE_CLASS = {"VALID_ASLI": "badge-asli", "MENCURIGAKAN": "badge-warn", "PALSU": "badge-palsu"}

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


def admin_ok() -> bool:
    """Proteksi opsional: set ADMIN_PASSWORD di .streamlit/secrets.toml."""
    try:
        pwd = st.secrets.get("ADMIN_PASSWORD", "")
    except Exception:
        pwd = ""
    if not pwd:
        return True
    return st.text_input("Password admin", type="password") == pwd


def make_error_map(y: np.ndarray, mask: np.ndarray, e_full: np.ndarray) -> Image.Image:
    """Merah = bit berubah, biru = area dinamis yang diabaikan."""
    rgb = np.stack([y * 255] * 3, axis=-1).astype(np.uint8)
    ignored = ~mask
    rgb[ignored] = (rgb[ignored] * 0.5 + np.array([60, 60, 200]) * 0.5).astype(np.uint8)
    rgb[e_full == 1] = [255, 0, 0]
    return Image.fromarray(rgb)


# ---------------------------------------------------------------- Header
st.title("🛡️ VerifyID: Sistem Deteksi Transaksi Digital")
st.caption("Verifikasi keaslian resi m-Banking & e-Wallet berbasis Hamming Distance & Hamming Weight")
st.markdown("---")

# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.header("⚙️ Parameter")
    provider = st.selectbox("Pilih m-Banking / e-Wallet:", list(PROVIDER_CONFIGS.keys()))
    tolerance = st.slider(
        "Toleransi ambang batas", 0.5, 5.0, 1.0, 0.1,
        help="Naikkan jika resi asli sering terdeteksi mencurigakan (positif palsu). "
             "Turunkan agar lebih ketat.",
    )
    n_refs = len(list_references(provider))
    st.markdown("---")
    st.info(f"**Provider:** {provider}\n\n**Referensi asli tersimpan:** {n_refs}\n\n"
            f"**Metode:** Coding Theory (XOR bit-wise)")

tab_verify, tab_db, tab_eval, tab_info = st.tabs(
    ["🔍 Verifikasi Resi", "📚 Database Referensi Asli", "📊 Evaluasi Model", "ℹ️ Cara Kerja & Batasan"]
)

# ---------------------------------------------------------------- Tab 1: Verifikasi
with tab_verify:
    model = build_model(provider)
    if model is None:
        st.warning(f"Referensi asli untuk **{provider}** baru {n_refs}. Minimal {MIN_REFS} "
                   f"(disarankan {RECOMMENDED_REFS}+). Tambahkan di tab *Database Referensi Asli*.")
    elif model["n_refs"] < RECOMMENDED_REFS:
        st.info(f"Model memakai {model['n_refs']} referensi. Akurasi lebih baik jika "
                f"≥ {RECOMMENDED_REFS} referensi dari perangkat berbeda.")

    col_left, col_right = st.columns(2)
    y = None
    img = None

    with col_left:
        st.subheader("📤 Unggah Bukti Transaksi")
        uploaded = st.file_uploader(f"Pilih resi ({provider}) - JPG/PNG",
                                    type=["png", "jpg", "jpeg"], key="verify_upload")
        if uploaded:
            st.image(uploaded, caption=f"Gambar uji ({provider})")

    with col_right:
        st.subheader("🔬 Pratinjau Biner")
        if uploaded:
            software = read_software_tag(uploaded)
            uploaded.seek(0)
            img = load_image(uploaded)
            y = binarize(img)
            st.image((y * 255).astype(np.uint8), caption="Representasi biner gambar uji (y)")

    analyze = st.button("🚀 Verifikasi Keaslian Resi", type="primary",
                        disabled=(model is None or uploaded is None))

    if analyze and model is not None and y is not None:
        st.markdown("---")
        st.header("📋 Hasil Evaluasi Integritas Data")

        cfg = PROVIDER_CONFIGS[provider]
        thr = model["threshold"] * tolerance
        res = forensic_compare(model["consensus"], y, model["mask"], thr)

        fmodel = build_field_model(list_references(provider), cfg)
        fres = check_fields(img, cfg, fmodel)
        final = final_verdict(res["verdict"], fres["verdict"] if fres["available"] else None)

        st.markdown(f'<div class="badge {BADGE_CLASS[final]}">{FINAL_TITLES[final]}</div>',
                    unsafe_allow_html=True)
        st.write(f"**Pemeriksaan tata letak (Hamming):** {res['description']}")
        if fres["available"]:
            if fres["verdict"] == "OK":
                st.write("**Pemeriksaan isi kolom:** Posisi, tinta, dan bentuk huruf pada kolom nominal, "
                         "nama, dan waktu sesuai pola resi asli.")
            else:
                st.write(f"**Pemeriksaan isi kolom:** ditemukan {fres['n_anomali']} kolom anomali dan "
                         f"{fres['n_curiga']} kolom mencurigakan (rincian di bawah).")
        else:
            st.warning(fres["reason"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Bit Dibandingkan (n)", f"{res['total_bits']:,}")
        m2.metric("Hamming Distance d(x,y)", f"{res['hamming_distance']:,}")
        m3.metric("Hamming Weight wt(e)", f"{res['hamming_weight']:,}")
        m4.metric("Rasio Perubahan Bit", f"{res['bit_change_ratio']:.4f} %")

        st.caption(f"Ambang batas asli: {thr * 100:.4f} %  |  "
                   f"wt(x) = {res['wt_x']:,}, wt(y) = {res['wt_y']:,}  |  "
                   f"Area tetap yang dibandingkan: {model['stable_fraction'] * 100:.1f} % dari gambar")

        st.markdown("### 🗺️ Peta Perubahan Bit")
        c1, c2 = st.columns(2)
        with c1:
            st.image(make_error_map(y, model["mask"], res["e_full"]),
                     caption="Merah = bit berubah | Biru = area dinamis (diabaikan)")
        with c2:
            st.image((model["consensus"] * 255).astype(np.uint8),
                     caption="Vektor acuan x (konsensus resi asli)")

        st.markdown("### 📐 Verifikasi Formulasi Teori Pengkodean")
        if res["is_verified"]:
            st.success(f"✅ **Terbukti ekuivalen:** d(x,y) = wt(x ⊕ y) = **{res['hamming_distance']:,} bit**, "
                       f"dan |wt(x) − wt(y)| ≤ d(x,y) terpenuhi.")
        else:
            st.error("⚠️ Ketidaksesuaian perhitungan biner!")

        if fres["available"]:
            st.markdown("### 🔎 Pemeriksaan Isi Kolom (nominal, nama, tanggal)")
            st.caption("Kolom isi berbeda di tiap transaksi, jadi yang dibandingkan adalah gaya tampilannya: "
                       "posisi rata kanan/tengah, ketebalan tinta antar-kolom, dan bentuk karakter "
                       "(Hamming Distance pada bitmap karakter).")
            table = pd.DataFrame([{"Kolom": r["Kolom"], "Status": r["Status"], "Keterangan": r["Keterangan"]}
                                  for r in fres["rows"]])
            st.dataframe(table)

        if software:
            st.warning(f"Metadata gambar memuat tag Software: **{software}**. "
                       f"Screenshot asli biasanya tidak memiliki tag ini; periksa kemungkinan penyuntingan.")

        st.info("ℹ️ Suntingan yang meniru posisi, warna tinta, dan font aslinya dengan sempurna masih bisa lolos. "
                "Untuk transaksi penting, cocokkan juga dengan mutasi rekening.")
        st.caption("Hasil ini berupa indikasi kesesuaian dengan pola resi asli, bukan bukti hukum.")
    elif uploaded is None:
        st.info("👆 Pilih provider di sidebar lalu unggah gambar resi untuk diverifikasi.")

# ---------------------------------------------------------------- Tab 2: Database
with tab_db:
    st.subheader(f"📚 Referensi Resi Asli: {provider}")
    if not admin_ok():
        st.warning("Masukkan password admin untuk mengelola database referensi.")
    else:
        st.write("Unggah **hanya resi yang pasti asli**. Resi palsu yang ikut tersimpan akan merusak model. "
                 f"Gunakan variasi transaksi dan perangkat (minimal {MIN_REFS}, disarankan {RECOMMENDED_REFS}+).")

        files = st.file_uploader("Unggah resi asli (boleh banyak)", type=["png", "jpg", "jpeg"],
                                 accept_multiple_files=True,
                                 key=f"ref_upload_{st.session_state.uploader_key}")
        if st.button("💾 Simpan sebagai referensi asli", disabled=not files):
            for f in files:
                save_reference(provider, f)
            st.session_state.uploader_key += 1
            st.rerun()

        refs = list_references(provider)
        if refs:
            st.markdown(f"**{len(refs)} referensi tersimpan**")
            cols = st.columns(5)
            for i, path in enumerate(refs):
                with cols[i % 5]:
                    st.image(path)
                    if st.button("🗑️ Hapus", key=f"del_{path}"):
                        delete_reference(path)
                        st.rerun()

            mdl = build_model(provider)
            if mdl:
                st.markdown("---")
                a, b = st.columns(2)
                a.metric("Area tetap (dibandingkan)", f"{mdl['stable_fraction'] * 100:.1f} %")
                b.metric("Ambang batas hasil kalibrasi", f"{mdl['threshold'] * 100:.4f} %")
        else:
            st.info("Belum ada referensi untuk provider ini.")

        st.caption("Catatan: di Streamlit Cloud penyimpanan bersifat sementara. "
                   "Commit folder data/references ke GitHub agar referensi tidak hilang saat restart.")

# ---------------------------------------------------------------- Tab 3: Evaluasi
with tab_eval:
    st.subheader("📊 Evaluasi Model (bahan Bab IV)")
    st.write("Unggah resi **asli uji** (yang BUKAN bagian dari referensi) dan resi **palsu uji** "
             "(hasil suntingan buatan sendiri). Sistem menghitung rasio perubahan bit tiap gambar "
             "dan akurasi pada ambang batas saat ini.")
    model_e = build_model(provider)
    if model_e is None:
        st.warning(f"Referensi {provider} belum cukup (minimal {MIN_REFS}).")
    else:
        ev1, ev2 = st.columns(2)
        genuine_files = ev1.file_uploader("Resi ASLI uji", type=["png", "jpg", "jpeg"],
                                          accept_multiple_files=True, key="eval_genuine")
        fake_files = ev2.file_uploader("Resi PALSU uji", type=["png", "jpg", "jpeg"],
                                       accept_multiple_files=True, key="eval_fake")
        if st.button("▶️ Jalankan evaluasi", disabled=not (genuine_files or fake_files)):
            thr_e = model_e["threshold"] * tolerance
            cfg_e = PROVIDER_CONFIGS[provider]
            fmodel_e = build_field_model(list_references(provider), cfg_e)
            rows = []
            for label, files in (("ASLI", genuine_files), ("PALSU", fake_files)):
                for f in files:
                    f.seek(0)
                    img_f = load_image(f)
                    r = forensic_compare(model_e["consensus"], binarize(img_f),
                                         model_e["mask"], thr_e)
                    fr = check_fields(img_f, cfg_e, fmodel_e)
                    fv = fr["verdict"] if fr["available"] else "-"
                    final_e = final_verdict(r["verdict"], fr["verdict"] if fr["available"] else None)
                    flagged = final_e != "VALID_ASLI"
                    rows.append({
                        "Berkas": f.name,
                        "Label sebenarnya": label,
                        "d(x,y)": r["hamming_distance"],
                        "Rasio (%)": round(r["bit_change_ratio"], 4),
                        "Vonis tata letak": r["verdict"],
                        "Vonis isi kolom": fv,
                        "Vonis akhir": final_e,
                        "Prediksi benar": flagged == (label == "PALSU"),
                    })
            df = pd.DataFrame(rows)
            st.dataframe(df)

            n_g = int((df["Label sebenarnya"] == "ASLI").sum())
            n_f = int((df["Label sebenarnya"] == "PALSU").sum())
            acc = df["Prediksi benar"].mean() * 100
            tpr = df.loc[df["Label sebenarnya"] == "PALSU", "Prediksi benar"].mean() * 100 if n_f else float("nan")
            fpr = (1 - df.loc[df["Label sebenarnya"] == "ASLI", "Prediksi benar"].mean()) * 100 if n_g else float("nan")
            k1, k2, k3 = st.columns(3)
            k1.metric("Akurasi", f"{acc:.1f} %")
            k2.metric("Palsu terdeteksi", "-" if n_f == 0 else f"{tpr:.1f} %")
            k3.metric("Asli salah ditandai", "-" if n_g == 0 else f"{fpr:.1f} %")
            st.caption("Vonis MENCURIGAKAN dan PALSU dihitung sebagai 'ditandai'. "
                       "Ubah slider toleransi di sidebar untuk melihat pengaruhnya.")

# ---------------------------------------------------------------- Tab 4: Info
with tab_info:
    st.markdown("""
### Cara kerja
1. Admin mengunggah beberapa resi asli per provider (database referensi).
2. Tiap gambar dinormalisasi: grayscale → ukuran tetap → blur ringan → threshold Otsu → bitstring.
3. **Pemeriksaan tata letak:** bagian tetap (logo, judul, label baris, footer) dibandingkan dengan
   vektor acuan: **e = x ⊕ y**, **d(x,y) = wt(e)**. Kolom isi diabaikan di tahap ini.
4. **Pemeriksaan isi kolom:** nominal, nama, dan waktu diperiksa gayanya: posisi rata kanan/tengah,
   ketebalan tinta dibanding kolom lain, dan bentuk karakter (Hamming Distance pada bitmap 12×20).
5. Ambang batas dikalibrasi dari variasi alami resi asli (leave-one-out).
6. Vonis akhir adalah yang terburuk dari kedua pemeriksaan.

### Batasan
- Hasil adalah **indikasi**, bukan bukti pasti.
- Suntingan kolom isi terdeteksi lewat bekas gayanya (posisi, tinta, font). Suntingan yang meniru
  ketiganya dengan sempurna bisa lolos.
- Pemeriksaan bentuk karakter hanya indikasi lemah: kompresi berat pada resi asli bisa memicu vonis *mencurigakan*.
- Pemeriksaan isi kolom butuh kartu putih terdeteksi (mode terang) dan minimal 2 resi asli.
- Kompresi, ukuran layar, dan pembaruan tampilan aplikasi bisa memicu positif palsu.
- Kualitas deteksi bergantung pada jumlah dan keragaman referensi asli.
""")
