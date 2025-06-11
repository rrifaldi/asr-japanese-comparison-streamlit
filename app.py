# Sel 2: Membuat file app.py (Revisi Final Hanya ASR & Romaji)

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

# Model terjemahan tidak diperlukan lagi

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

# Fungsi untuk memuat model terjemahan tidak diperlukan lagi

# --- FUNGSI PEMROSESAN AUDIO ---
def convert_to_romaji(text_japanese):
    """Mengonversi teks Jepang (Kanji/Kana) ke Romaji."""
    if not text_japanese:
        return ""
    return converter.do(text_japanese)

# Fungsi download_audio dan logic WER tidak diperlukan lagi
def process_audio_with_model(audio_path, asr_pipeline, model_label): # argumen translator_ja_en_pipeline dihapus
    """Memproses audio dengan model ASR tertentu."""
    st.subheader(f"Hasil dari: {model_label}")
    
    # Transkripsi ASR (Output Jepang Asli)
    with st.spinner(f"⏳ Sedang mentranskripsi audio dengan {model_label}..."):
        start_time_asr = time.time()
        try:
            transcription_japanese = asr_pipeline(audio_path)["text"]
            end_time_asr = time.time()
            st.success("✅ Transkripsi Jepang Selesai!")
            st.write(f"Waktu Transkripsi: **{end_time_asr - start_time_asr:.2f} detik**")
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat transkripsi: {e}")
            transcription_japanese = "Error saat transkripsi." 
    
    # Tampilkan Transkripsi Jepang Asli
    st.markdown("**Transkripsi Jepang Asli (Kanji/Kana):**")
    st.code(transcription_japanese)

    # Konversi ke Romaji
    st.markdown("**Romaji:**")
    romaji_text = convert_to_romaji(transcription_japanese)
    st.code(romaji_text)

    # Bagian terjemahan tidak diperlukan lagi
    st.write("---") # Garis pemisah antar model


# --- INTERFACE PENGGUNA STREAMLIT ---
st.set_page_config(layout="wide", page_title="Perbandingan ASR Audio Jepang") # Judul diubah

st.title("🗣️ Perbandingan ASR Audio Jepang")
st.markdown("Unggah file audio berbahasa Jepang untuk membandingkan transkripsi dari dua model Whisper, serta mendapatkan Romaji.") # Deskripsi diubah
st.markdown("---")

# Muat model ASR
with st.spinner("⏳ Memuat semua model AI (ASR)... Ini mungkin butuh beberapa saat."):
    whisper_base_pipeline = load_asr_model(MODEL_WHISPER_BASE)
    anime_whisper_pipeline = load_asr_model(MODEL_ANIME_WHISPER)
    # Model terjemahan tidak diperlukan lagi
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
st.header("2. Hasil Perbandingan Transkripsi & Romaji") # Judul diubah

if st.button("▶️ Mulai Perbandingan!"):
    if audio_path:
        # Proses audio dengan OpenAI Whisper (Base)
        process_audio_with_model(audio_path, whisper_base_pipeline, "OpenAI Whisper (Base)") # argumen translator_ja_en_pipeline dihapus
        
        # Proses audio dengan litagin/anime-whisper
        process_audio_with_model(audio_path, anime_whisper_pipeline, "litagin/anime-whisper") # argumen translator_ja_en_pipeline dihapus

        # Hapus file sementara setelah selesai memproses
        if os.path.exists(audio_path):
            os.remove(audio_path)
    else:
        st.warning("Silakan unggah file audio Anda terlebih dahulu di bagian '1. Unggah File Audio'.")

st.markdown("---")
st.caption("Aplikasi ini dibuat dengan Streamlit dan Hugging Face Transformers.")
