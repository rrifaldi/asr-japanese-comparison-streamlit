import streamlit as st
from transformers import pipeline
import torch
import os
import io
import time
import soundfile as sf
import pykakasi # Impor library untuk transliterasi Romaji

# --- KONFIGURASI MODEL ---
MODEL_WHISPER_BASE = "openai/whisper-base"
MODEL_ANIME_WHISPER = "litagin/anime-whisper"

# Inisialisasi Kakasi (untuk konversi Jepang ke Romaji)
kks = pykakasi.kakasi()
kks.setMode("H", "a") # Hiragana to Alphabet
kks.setMode("K", "a") # Katakana to Alphabet
kks.setMode("J", "a") # Kanji to Alphabet
kks.setMode("r", "Hepburn") # Romaji system
converter = kks.getConverter()


# --- CACHE MODEL ---
@st.cache_resource
def load_asr_model(model_name):
    device = 0 if torch.cuda.is_available() else -1
    asr_pipe = pipeline("automatic-speech-recognition", model=model_name, device=device)
    return asr_pipe

# Fungsi untuk memuat model terjemahan tidak diperlukan lagi (sesuai revisi sebelumnya)


# --- FUNGSI PEMROSESAN AUDIO ---
def convert_to_romaji(text_japanese):
    """Mengonversi teks Jepang (Kanji/Kana) ke Romaji."""
    if not text_japanese:
        return ""
    return converter.do(text_japanese)

def process_audio_with_model(audio_path, asr_pipeline, model_label):
    """Memproses audio dengan model ASR tertentu."""
    st.subheader(f"Hasil dari: {model_label}")
    
    # st.status() untuk progress yang lebih interaktif (Streamlit 1.25+)
    # Jika versi Streamlit Anda lebih rendah, gunakan st.spinner seperti sebelumnya
    try:
        # Pengecekan versi Streamlit untuk st.status
        if hasattr(st, 'status'):
            with st.status(f"⏳ Sedang mentranskripsi audio dengan {model_label}...", expanded=True) as status_box:
                start_time_asr = time.time()
                transcription_japanese = asr_pipeline(audio_path)["text"]
                end_time_asr = time.time()
                status_box.update(label=f"✅ Transkripsi {model_label} Selesai!", state="complete", expanded=False)
                st.write(f"Waktu Transkripsi: **{end_time_asr - start_time_asr:.2f} detik**")
        else: # Fallback untuk Streamlit versi lama
            with st.spinner(f"⏳ Sedang mentranskripsi audio dengan {model_label}..."):
                start_time_asr = time.time()
                transcription_japanese = asr_pipeline(audio_path)["text"]
                end_time_asr = time.time()
                st.success("✅ Transkripsi Jepang Selesai!")
                st.write(f"Waktu Transkripsi: **{end_time_asr - start_time_asr:.2f} detik**")
            
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat transkripsi: {e}")
        transcription_japanese = "Error saat transkripsi." 
    
    # Tampilkan Transkripsi Jepang Asli dalam expander
    with st.expander("Lihat Transkripsi Jepang Asli (Kanji/Kana)"):
        st.code(transcription_japanese)

    # Tampilkan Romaji dalam expander
    with st.expander("Lihat Romaji"):
        romaji_text = convert_to_romaji(transcription_japanese)
        st.code(romaji_text)
    
    st.write("---") # Garis pemisah antar model


# --- INTERFACE PENGGUNA STREAMLIT ---
st.set_page_config(layout="wide", page_title="Perbandingan ASR Audio Jepang")

st.title("🗣️ Perbandingan ASR Audio Jepang")
st.markdown("Unggah file audio berbahasa Jepang untuk membandingkan transkripsi dari dua model Whisper, serta mendapatkan Romaji.")
st.markdown("---")

# Gunakan st.tabs untuk mengatur konten
tab1, tab2 = st.tabs(["Aplikasi Utama", "Tentang Proyek Ini"])

with tab1: # Konten utama aplikasi
    # Muat model ASR
    with st.spinner("⏳ Memuat semua model AI (ASR)... Ini mungkin butuh beberapa saat."):
        asr_pipeline_base = load_asr_model(MODEL_WHISPER_BASE)
        anime_whisper_pipeline = load_asr_model(MODEL_ANIME_WHISPER)
    st.success("✅ Semua model AI berhasil dimuat dan siap digunakan.")


    st.header("1. Unggah File Audio Bahasa Jepang")
    uploaded_file = st.file_uploader(
        "Pilih file audio (.wav, .mp3, .flac) berbahasa Jepang:",
        type=["wav", "mp3", "flac"]
    )

    audio_path = None
    if uploaded_file is not None:
        st.audio(uploaded_file, format=uploaded_file.type)
        
        try:
            audio_path = "temp_uploaded_audio.wav"
            with open(audio_path, "wb") as f:
                f.write(uploaded_file.read())
            st.success("✅ File audio berhasil diunggah.")
        except Exception as e:
            st.error(f"❌ Error membaca file audio: {e}")
            st.info("Pastikan format file audio kompatibel dan tidak korup. Coba unggah file lain.")
            audio_path = None
    else:
        st.info("👆 Silakan unggah file audio berbahasa Jepang untuk memulai proses.")


    st.markdown("---")
    st.header("2. Hasil Perbandingan Transkripsi & Romaji")

    if st.button("▶️ Mulai Perbandingan!"):
        if audio_path:
            process_audio_with_model(audio_path, asr_pipeline_base, "OpenAI Whisper (Base)")
            process_audio_with_model(audio_path, anime_whisper_pipeline, "litagin/anime-whisper")

            if os.path.exists(audio_path):
                os.remove(audio_path)
        else:
            st.warning("Silakan unggah file audio Anda terlebih dahulu di bagian '1. Unggah File Audio'.")

with tab2: # Konten tentang proyek
    st.header("Tentang Proyek Ini")
    st.markdown("""
    Ini adalah aplikasi demonstrasi yang dibuat untuk membandingkan kemampuan dua model Automatic Speech Recognition (ASR) dalam mentranskripsi audio berbahasa Jepang, serta mengonversinya ke Romaji.

    ### Model yang Digunakan:
    - **OpenAI Whisper (Base):** Model ASR umum yang dilatih oleh OpenAI, dikenal dengan kemampuannya dalam berbagai bahasa.
    - **litagin/anime-whisper:** Model ASR yang di-fine-tune khusus untuk audio dari konten anime.

    ### Fitur:
    - Mengunggah file audio (.wav, .mp3, .flac) berbahasa Jepang.
    - Menampilkan transkripsi asli dalam aksara Jepang (Kanji/Kana).
    - Mengonversi transkripsi ke Romaji menggunakan pustaka PyKakasi.
    - Membandingkan hasil dari kedua model untuk audio yang sama.

    ### Metode:
    - **ASR:** Model Transformer Encoder-Decoder mengubah audio menjadi teks.
    - **Transliterasi:** PyKakasi untuk konversi Kanji/Kana ke Romaji.
    - **Hosting:** Aplikasi web dibangun dengan Streamlit dan di-deploy menggunakan GitHub, dengan dependensi sistem seperti FFmpeg diatur melalui `packages.txt`.

    ### Kontak:
    Untuk pertanyaan atau informasi lebih lanjut, silakan hubungi [Nama Anda/Link GitHub Anda].
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Streamlit_logo_black_text.svg/1200px-Streamlit_logo_black_text.svg.png", width=150)
    st.image("https://huggingface.co/front/assets/huggingface_logo.svg", width=150)


st.markdown("---")
st.caption("Aplikasi ini dibuat dengan Streamlit dan Hugging Face Transformers.")
