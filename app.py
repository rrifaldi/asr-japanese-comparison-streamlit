# Sel 2: Membuat file app.py (Revisi Final untuk Terjemahan Melalui Bahasa Inggris - Coba Model Lain)

%%writefile app.py
import streamlit as st
from transformers import pipeline
import torch
import os
import io
import time
import soundfile as sf
import pykakasi # Impor library untuk transliterasi Romaji

# --- KONFIGURASI MODEL ASR ---
MODEL_WHISPER_BASE = "openai/whisper-base"
MODEL_ANIME_WHISPER = "litagin/anime-whisper"

# Model terjemahan Jepang ke Inggris (Nama model yang lebih umum/standar)
MODEL_TRANSLATION_JA_EN = "Helsinki-NLP/opus-mt-ja-en"
# Model terjemahan Inggris ke Indonesia
MODEL_TRANSLATION_EN_ID = "Helsinki-NLP/opus-mt-en-id"

# Inisialisasi Kakasi (untuk konversi Jepang ke Romaji)
kks = pykakasi.kakasi()
kks.setMode("H", "a") # Hiragana to Alphabet
kks.setMode("K", "a") # Katakana to Alphabet
kks.setMode("J", "a") # Kanji to Alphabet
kks.setMode("r", "Hepburn") # Romaji system
converter = kks.getConverter()


# --- CACHE MODEL ---
# Fungsi untuk memuat model ASR
@st.cache_resource
def load_asr_model(model_name):
    st.write(f"⏳ Memuat model ASR: **{model_name}**... Ini mungkin butuh beberapa detik.")
    device = 0 if torch.cuda.is_available() else -1
    asr_pipe = pipeline("automatic-speech-recognition", model=model_name, device=device)
    st.success(f"✅ Model **{model_name}** berhasil dimuat.")
    return asr_pipe

# Fungsi untuk memuat model terjemahan (Jepang ke Inggris)
@st.cache_resource
def load_translator_ja_en(model_name):
    st.write(f"⏳ Memuat model Terjemahan: **{model_name}** (Jepang ke Inggris)... Ini mungkin butuh beberapa detik.")
    try:
        device = 0 if torch.cuda.is_available() else -1
        translator_pipe = pipeline("translation", model=model_name, device=device)
        st.success(f"✅ Model **{model_name}** berhasil dimuat.")
        return translator_pipe
    except Exception as e:
        st.error(f"❌ Gagal memuat model terjemahan {model_name}: {e}. Terjemahan Jepang-Inggris mungkin tidak berfungsi.")
        return None # Return None jika gagal dimuat

# Fungsi untuk memuat model terjemahan (Inggris ke Indonesia)
@st.cache_resource
def load_translator_en_id(model_name):
    st.write(f"⏳ Memuat model Terjemahan: **{model_name}** (Inggris ke Indonesia)... Ini mungkin butuh beberapa detik.")
    try:
        device = 0 if torch.cuda.is_available() else -1
        translator_pipe = pipeline("translation", model=model_name, device=device)
        st.success(f"✅ Model **{model_name}** berhasil dimuat.")
        return translator_pipe
    except Exception as e:
        st.error(f"❌ Gagal memuat model terjemahan {model_name}: {e}. Terjemahan Inggris-Indonesia mungkin tidak berfungsi.")
        return None # Return None jika gagal dimuat

# --- FUNGSI PEMROSESAN AUDIO ---
def convert_to_romaji(text_japanese):
    """Mengonversi teks Jepang (Kanji/Kana) ke Romaji."""
    if not text_japanese:
        return ""
    return converter.do(text_japanese)

def process_audio_with_model(audio_path, asr_pipeline, translator_ja_en_pipeline, translator_en_id_pipeline, model_label):
    """Memproses audio dengan model ASR tertentu, lalu menerjemahkan."""
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
            # st.stop() # Jangan stop, biar bagian lain tetap jalan

    # Tampilkan Transkripsi Jepang Asli
    st.markdown("**Transkripsi Jepang Asli (Kanji/Kana):**")
    st.code(transcription_japanese)

    # Konversi ke Romaji
    st.markdown("**Romaji:**")
    romaji_text = convert_to_romaji(transcription_japanese)
    st.code(romaji_text)

    # Terjemahan ke Bahasa Inggris (Langkah Perantara)
    english_translation = None
    if transcription_japanese and translator_ja_en_pipeline:
        st.markdown("**Terjemahan ke Bahasa Inggris:**")
        with st.spinner(f"⏳ Sedang menerjemahkan dari Jepang ke Inggris..."):
            start_time_ja_en = time.time()
            try:
                english_translation_output = translator_ja_en_pipeline(transcription_japanese)
                english_translation = english_translation_output[0]['translation_text']
                end_time_ja_en = time.time()
                st.success("✅ Terjemahan Jepang ke Inggris Selesai!")
                st.code(english_translation)
                st.write(f"Waktu Terjemahan (Jepang ke Inggris): **{end_time_ja_en - start_time_ja_en:.2f} detik**")
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat terjemahan Jepang ke Inggris: {e}")
    elif transcription_japanese:
        st.warning("Model terjemahan Jepang ke Inggris tidak dimuat. Terjemahan ke Bahasa Inggris tidak tersedia.")

    # Terjemahan ke Bahasa Indonesia (Langkah Akhir)
    if english_translation and translator_en_id_pipeline:
        st.markdown("**Terjemahan ke Bahasa Indonesia:**")
        with st.spinner(f"⏳ Sedang menerjemahkan dari Inggris ke Bahasa Indonesia..."):
            start_time_en_id = time.time()
            try:
                indonesian_translation_output = translator_en_id_pipeline(english_translation)
                indonesian_translation = indonesian_translation_output[0]['translation_text']
                end_time_en_id = time.time()
                st.success("✅ Terjemahan Bahasa Indonesia Selesai!")
                st.code(indonesian_translation)
                st.write(f"Waktu Terjemahan (Inggris ke Indonesia): **{end_time_en_id - start_time_en_id:.2f} detik**")
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat terjemahan Inggris ke Bahasa Indonesia: {e}")
    elif english_translation:
        st.warning("Model terjemahan Inggris ke Indonesia tidak dimuat. Terjemahan ke Bahasa Indonesia tidak tersedia.")

    st.write("---") # Garis pemisah antar model


# --- INTERFACE PENGGUNA STREAMLIT ---
# PENTING: st.set_page_config harus menjadi perintah Streamlit pertama yang dieksekusi.
st.set_page_config(layout="wide", page_title="Perbandingan ASR & Terjemahan Audio Jepang")

# Judul utama aplikasi
st.title("🗣️ Perbandingan ASR & Terjemahan Audio Jepang")
st.markdown("Unggah file audio berbahasa Jepang untuk membandingkan transkripsi dari dua model Whisper, serta mendapatkan Romaji dan terjemahan Bahasa Indonesia.")
st.markdown("---")

# Muat model ASR dan Terjemahan
with st.spinner("⏳ Memuat semua model AI (ASR & Terjemahan)... Ini mungkin butuh beberapa saat."):
    whisper_base_pipeline = load_asr_model(MODEL_WHISPER_BASE)
    anime_whisper_pipeline = load_asr_model(MODEL_ANIME_WHISPER)
    translator_ja_en_pipeline = load_translator_ja_en(MODEL_TRANSLATION_JA_EN)
    translator_en_id_pipeline = load_translator_en_id(MODEL_TRANSLATION_EN_ID)
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
st.header("2. Hasil Perbandingan Transkripsi & Terjemahan")

if st.button("▶️ Mulai Perbandingan!"):
    if audio_path:
        # Proses audio dengan OpenAI Whisper (Base)
        process_audio_with_model(audio_path, whisper_base_pipeline, translator_ja_en_pipeline, translator_en_id_pipeline, "OpenAI Whisper (Base)")

        # Proses audio dengan litagin/anime-whisper
        process_audio_with_model(audio_path, anime_whisper_pipeline, translator_ja_en_pipeline, translator_en_id_pipeline, "litagin/anime-whisper")

        # Hapus file sementara setelah selesai memproses
        if os.path.exists(audio_path):
            os.remove(audio_path)
    else:
        st.warning("Silakan unggah file audio Anda terlebih dahulu di bagian '1. Unggah File Audio'.")

st.markdown("---")
st.caption("Aplikasi ini dibuat dengan Streamlit dan Hugging Face Transformers.")
