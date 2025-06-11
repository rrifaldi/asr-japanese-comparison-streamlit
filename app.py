import streamlit as st
from transformers import pipeline
import torch
import os
import io
import time
import soundfile as sf
import pykakasi # Impor library untuk transliterasi Romaji
from jiwer import wer # Impor jiwer untuk WER
import requests # Untuk mengunduh file audio dari URL

# --- KONFIGURASI MODEL ---
MODEL_WHISPER_BASE = "openai/whisper-base"
MODEL_ANIME_WHISPER = "litagin/anime-whisper"

# Model terjemahan Jepang ke Inggris (Nama model yang lebih umum/standar)
MODEL_TRANSLATION_JA_EN = "Helsinki-NLP/opus-mt-ja-en" 
# MODEL_TRANSLATION_EN_ID tidak diperlukan lagi

# Inisialisasi Kakasi (untuk konversi Jepang ke Romaji)
kks = pykakasi.kakasi()
kks.setMode("H", "a") # Hiragana to Alphabet
kks.setMode("K", "a") # Katakana to Alphabet
kks.setMode("J", "a") # Kanji to Alphabet
kks.setMode("r", "Hepburn") # Romaji system
converter = kks.getConverter()

# --- DATA UJI OTOMATIS (CONTOH) ---
# Anda bisa mengganti URL dan ground truth ini dengan data Anda sendiri
# Pastikan URL audio dapat diakses publik.
TEST_AUDIOS = {
    "Sapaan Pagi (Kon'nichiwa)": { # Contoh URL
        "url": "https://www.learning-japanese.com/sounds/konnichiwa.mp3",
        "ground_truth_jp": "こんにちは",
        "ground_truth_romaji": "konnichiwa", # Untuk referensi
        "ground_truth_id": "Halo" # Untuk referensi
    },
    "Terima Kasih (Arigatou Gozaimasu)": {
        "url": "https://www.learning-japanese.com/sounds/arigatou.mp3",
        "ground_truth_jp": "ありがとうございます",
        "ground_truth_romaji": "arigatou gozaimasu",
        "ground_truth_id": "Terima kasih banyak"
    },
    "Contoh Frasa Anime (Fiksi)": {
        "url": "https://file-examples.com/storage/feae07a82762b3225a07c00/2017/11/file_example_MP3_700KB.mp3", # Ganti dengan audio anime jika ada!
        "ground_truth_jp": "行くぞ", 
        "ground_truth_romaji": "iku zo",
        "ground_truth_id": "Ayo pergi!"
    }
}

# --- CACHE MODEL ---
@st.cache_resource
def load_asr_model(model_name):
    st.write(f"⏳ Memuat model ASR: **{model_name}**... Ini mungkin butuh beberapa detik.")
    device = 0 if torch.cuda.is_available() else -1
    asr_pipe = pipeline("automatic-speech-recognition", model=model_name, device=device)
    st.success(f"✅ Model **{model_name}** berhasil dimuat.")
    return asr_pipe

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
        return None

# load_translator_en_id tidak diperlukan lagi

# --- FUNGSI PEMROSESAN AUDIO ---
def convert_to_romaji(text_japanese):
    if not text_japanese:
        return ""
    return converter.do(text_japanese)

def download_audio(url, filename="downloaded_audio.mp3"):
    """Mengunduh file audio dari URL."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return filename
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Gagal mengunduh audio dari URL: {e}")
        return None

def process_audio_with_model(audio_path, asr_pipeline, translator_ja_en_pipeline, model_label, ground_truth_jp): # translator_en_id_pipeline dihapus
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
    
    # Tampilkan Transkripsi Jepang Asli
    st.markdown("**Transkripsi Jepang Asli (Kanji/Kana):**")
    st.code(transcription_japanese)

    # Hitung dan Tampilkan WER (Word Error Rate)
    if ground_truth_jp and transcription_japanese and transcription_japanese != "Error saat transkripsi.":
        gt_normalized = " ".join(ground_truth_jp.split())
        trans_normalized = " ".join(transcription_japanese.split())
        current_wer = wer(gt_normalized, trans_normalized)
        st.markdown(f"**Word Error Rate (WER):** `{current_wer:.4f}` (Semakin rendah, semakin baik)")
    else:
        st.info("WER tidak dapat dihitung tanpa ground truth atau jika transkripsi gagal.")

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
    
    st.write("---") # Garis pemisah antar model


# --- INTERFACE PENGGUNA STREAMLIT ---
st.set_page_config(layout="wide", page_title="Perbandingan ASR & Terjemahan Audio Jepang Otomatis")

st.title("🗣️ Perbandingan ASR & Terjemahan Audio Jepang (Otomatis)")
st.markdown("Pilih file audio uji untuk membandingkan transkripsi dari dua model Whisper, serta mendapatkan Romaji dan terjemahan Bahasa Inggris secara otomatis.")
st.markdown("---")

# Muat model ASR dan Terjemahan
with st.spinner("⏳ Memuat semua model AI (ASR & Terjemahan)... Ini mungkin butuh beberapa saat."):
    asr_pipeline_base = load_asr_model(MODEL_WHISPER_BASE)
    asr_pipeline_anime = load_asr_model(MODEL_ANIME_WHISPER)
    translator_ja_en_pipeline = load_translator_ja_en(MODEL_TRANSLATION_JA_EN)
    # translator_en_id_pipeline tidak diperlukan lagi
st.success("✅ Semua model AI berhasil dimuat dan siap digunakan.")


st.header("1. Pilih File Audio Uji")
selected_test_audio_name = st.selectbox(
    "Pilih skenario audio untuk diuji:",
    list(TEST_AUDIOS.keys())
)

selected_test_audio_data = TEST_AUDIOS[selected_test_audio_name]
audio_url_to_test = selected_test_audio_data["url"]
ground_truth_japanese = selected_test_audio_data["ground_truth_jp"]
ground_truth_romaji = selected_test_audio_data["ground_truth_romaji"]
ground_truth_indonesian = selected_test_audio_data["ground_truth_id"] # Masih ada untuk display ground truth

st.write(f"Audio yang dipilih: `{selected_test_audio_name}`")
# Tampilkan ground truth
st.markdown("### Ground Truth (Verifikasi Manusia)")
st.write(f"**Jepang (Kanji/Kana):** `{ground_truth_japanese}`")
st.write(f"**Romaji:** `{ground_truth_romaji}`")
st.write(f"**Indonesia (Untuk Referensi):** `{ground_truth_indonesian}`") # Tetap tampilkan referensi ID

st.markdown("---")
st.header("2. Mulai Perbandingan Otomatis")

if st.button("▶️ Jalankan Perbandingan!"):
    if audio_url_to_test:
        with st.spinner("⏳ Mengunduh file audio uji..."):
            audio_file_temp_path = download_audio(audio_url_to_test)
        
        if audio_file_temp_path:
            st.audio(audio_file_temp_path) # Putar audio yang diunduh
            
            # Proses audio dengan OpenAI Whisper (Base)
            # translator_en_id_pipeline dihapus dari argumen
            process_audio_with_model(audio_file_temp_path, asr_pipeline_base, translator_ja_en_pipeline, "OpenAI Whisper (Base)", ground_truth_japanese)
            
            # Proses audio dengan litagin/anime-whisper
            # translator_en_id_pipeline dihapus dari argumen
            process_audio_with_model(audio_file_temp_path, asr_pipeline_anime, translator_ja_en_pipeline, "litagin/anime-whisper", ground_truth_japanese)

            # Hapus file sementara setelah selesai memproses
            if os.path.exists(audio_file_temp_path):
                os.remove(audio_file_temp_path)
        else:
            st.error("Gagal mengunduh audio uji.")
    else:
        st.warning("Silakan pilih skenario audio untuk memulai perbandingan.")

st.markdown("---")
st.caption("Aplikasi ini dibuat dengan Streamlit, Hugging Face Transformers, dan jiwer.")
