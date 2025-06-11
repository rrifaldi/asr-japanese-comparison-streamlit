Anda benar sekali! Saya melihat screenshot perbandingan langsung yang Anda berikan. Meskipun tabelnya muncul, highlighting perbedaan teksnya sangat samar atau tidak terlihat jelas karena warnanya terlalu mirip dengan latar belakang gelap. Sulit sekali membedakannya, apalagi jika tidak bisa Bahasa Jepang.

Kita perlu membuat highlighting-nya jauh lebih jelas dan menambahkan pesan eksplisit apakah ada perbedaan atau tidak.

Solusi: Perbaiki Warna Highlight dan Tambahkan Pesan Perbandingan
Saya akan memodifikasi fungsi highlight_diff di app.py untuk:

Mengubah Warna Latar Belakang Highlight: Menggunakan warna yang kontras dengan latar belakang gelap Streamlit (misalnya, merah terang untuk yang dihapus/berbeda di satu sisi, dan hijau terang untuk yang ditambahkan/berbeda di sisi lain).
Menambahkan Pesan Ekplisit: Memberi tahu pengguna secara langsung apakah ada perbedaan dan berapa banyak perbedaan yang ditemukan (dalam jumlah "kata").
Revisi Kode Sel Kedua: app.py (Perbaikan Highlight dan Pesan Perbandingan)
Tempelkan seluruh kode ini di sel kedua notebook Colab Anda (Gantikan kode app.py yang lama). Jalankan sel ini.

Python

# Sel 2: Membuat file app.py (Revisi untuk Perbaikan Highlight dan Pesan Perbandingan)

%%writefile app.py
import streamlit as st
from transformers import pipeline
import torch
import os
import io
import time
import soundfile as sf
import pykakasi # Impor library untuk transliterasi Romaji
import difflib # Impor library untuk membandingkan teks

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


# --- UTILITY FUNCTION FOR TEXT COMPARISON ---
def highlight_diff(text1, text2, label1="Teks 1", label2="Teks 2"):
    """
    Membandingkan dua teks dan mengembalikan HTML yang menyorot perbedaan.
    Digunakan untuk menampilkan perbedaan antar transkripsi side-by-side.
    Juga mengembalikan jumlah perbedaan.
    """
    s = difflib.SequenceMatcher(None, text1.split(), text2.split())
    
    diff_text1_html = []
    diff_text2_html = []
    num_diffs = 0

    for opcode, a_start, a_end, b_start, b_end in s.get_opcodes():
        if opcode == 'equal':
            diff_text1_html.extend(text1.split()[a_start:a_end])
            diff_text2_html.extend(text2.split()[b_start:b_end])
        elif opcode == 'replace':
            num_diffs += max(a_end - a_start, b_end - b_start)
            deleted_from_1 = [f"<span style='background-color: #FF6B6B; color: black; font-weight: bold;'>{word}</span>" for word in text1.split()[a_start:a_end]] # Merah terang
            inserted_in_2 = [f"<span style='background-color: #6BFF6B; color: black; font-weight: bold;'>{word}</span>" for word in text2.split()[b_start:b_end]] # Hijau terang
            
            diff_text1_html.extend(deleted_from_1)
            diff_text2_html.extend(inserted_in_2)
        elif opcode == 'delete':
            num_diffs += (a_end - a_start)
            deleted_from_1 = [f"<span style='background-color: #FF6B6B; color: black; font-weight: bold;'>{word}</span>" for word in text1.split()[a_start:a_end]]
            
            diff_text1_html.extend(deleted_from_1)
            diff_text2_html.extend([f"<span style='color: #888888; font-style: italic;'>[kosong]</span>" for _ in range(a_end - a_start)])
        elif opcode == 'insert':
            num_diffs += (b_end - b_start)
            inserted_in_2 = [f"<span style='background-color: #6BFF6B; color: black; font-weight: bold;'>{word}</span>" for word in text2.split()[b_start:b_end]]
            
            diff_text1_html.extend([f"<span style='color: #888888; font-style: italic;'>[kosong]</span>" for _ in range(b_end - b_start)])
            diff_text2_html.extend(inserted_in_2)
            
    html_output = f"""
    <style>
        .diff-table {{
            width: 100%;
            border-collapse: collapse;
            font-family: 'Inter', monospace; /* Menggunakan Inter jika ada, fallback monospace */
            word-break: break-word; /* Memastikan kata panjang tidak merusak layout */
        }}
        .diff-table th, .diff-table td {{
            border: 1px solid #444;
            padding: 8px;
            text-align: left;
            vertical-align: top;
        }}
        .diff-table th {{
            background-color: #333;
            color: white;
        }}
        .diff-table td {{
            background-color: #1a1a1a; /* Latar belakang lebih gelap dari sebelumnya */
            color: #eee;
        }}
        /* Warna highlight yang diperbaiki */
        .diff-table span[style*="background-color: #FF6B6B"] {{ background-color: #FF6B6B; color: black; }} /* Merah terang */
        .diff-table span[style*="background-color: #6BFF6B"] {{ background-color: #6BFF6B; color: black; }} /* Hijau terang */
        .diff-table span[style*="text-decoration: line-through"] {{ text-decoration: line-through; }} /* Garis coret untuk deleted */
    </style>
    <table class="diff-table">
        <thead>
            <tr>
                <th>{label1}</th>
                <th>{label2}</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>{" ".join(diff_text1_html)}</td>
                <td>{" ".join(diff_text2_html)}</td>
            </tr>
        </tbody>
    </table>
    """
    return html_output, num_diffs


# --- FUNGSI PEMROSESAN AUDIO ---
def convert_to_romaji(text_japanese):
    """Mengonversi teks Jepang (Kanji/Kana) ke Romaji."""
    if not text_japanese:
        return ""
    return converter.do(text_japanese)

# Mengubah fungsi process_audio_with_model agar mengembalikan hasil transkripsi
def process_audio_with_model(audio_path, asr_pipeline, model_label):
    """Memproses audio dengan model ASR tertentu."""
    st.subheader(f"Hasil dari: {model_label}")
    
    transcription_japanese = "Error saat transkripsi." # Default error
    
    # st.status() untuk progress yang lebih interaktif (Streamlit 1.25+)
    try:
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
        # Log error detail ke console Streamlit untuk debugging
        st.exception(e) # Menampilkan traceback error di UI
    
    # Tampilkan Transkripsi Jepang Asli dan Romaji dalam expander
    st.markdown(f"**Transkripsi dari {model_label}:**")
    with st.expander(f"Lihat detail transkripsi {model_label}"):
        st.markdown("**Kanji/Kana:**")
        st.code(transcription_japanese)
        st.markdown("**Romaji:**")
        romaji_text = convert_to_romaji(transcription_japanese)
        st.code(romaji_text)
    
    st.write("---") # Garis pemisah antar model
    
    return transcription_japanese # Mengembalikan hasil transkripsi untuk perbandingan


# --- INTERFACE PENGGUNA STREAMLIT ---
st.set_page_config(layout="wide", page_title="Perbandingan ASR Audio Jepang")

st.title("🗣️ Perbandingan ASR Audio Jepang")
st.markdown("Unggah file audio berbahasa Jepang untuk membandingkan transkripsi dari dua model Whisper, serta mendapatkan Romaji.")
st.markdown("---")

# Muat model ASR
with st.spinner("⏳ Memuat semua model AI (ASR)... Ini mungkin butuh beberapa saat."):
    asr_pipeline_base = load_asr_model(MODEL_WHISPER_BASE)
    anime_whisper_pipeline = load_asr_model(MODEL_ANIME_WHISPER)
st.success("✅ Semua model AI berhasil dimuat dan siap digunakan.")

# Inisialisasi session_state untuk menyimpan hasil transkripsi
if 'base_transcript' not in st.session_state:
    st.session_state.base_transcript = ""
if 'anime_transcript' not in st.session_state:
    st.session_state.anime_transcript = ""


tab1, tab2 = st.tabs(["Aplikasi Utama", "Tentang Proyek Ini"])

with tab1: # Konten utama aplikasi
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
            # Panggil process_audio_with_model dan simpan hasilnya di session_state
            st.session_state.base_transcript = process_audio_with_model(audio_path, asr_pipeline_base, "OpenAI Whisper (Base)")
            st.session_state.anime_transcript = process_audio_with_model(audio_path, anime_whisper_pipeline, "litagin/anime-whisper")

            # Hapus file sementara setelah selesai memproses
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            # Setelah kedua transkripsi didapat, tampilkan perbandingan langsung
            if st.session_state.base_transcript and st.session_state.anime_transcript and \
               st.session_state.base_transcript != "Error saat transkripsi." and \
               st.session_state.anime_transcript != "Error saat transkripsi.":
                
                st.markdown("### Perbandingan Langsung Transkripsi (Kanji/Kana)")
                diff_output_html, num_diffs = highlight_diff(st.session_state.base_transcript, st.session_state.anime_transcript, "Whisper Base", "Anime-Whisper")
                
                if num_diffs > 0:
                    st.warning(f"⚠️ Ditemukan {num_diffs} perbedaan kata antara kedua transkripsi.")
                else:
                    st.success("🎉 Tidak ada perbedaan yang ditemukan antara kedua transkripsi!")

                st.markdown(diff_output_html, unsafe_allow_html=True)
            else:
                st.warning("Tidak dapat membandingkan transkripsi karena salah satu atau kedua model gagal atau menghasilkan error.")
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
    - Menampilkan **perbandingan langsung** antara hasil transkripsi dari kedua model untuk melihat perbedaannya dengan jelas.

    ### Metode:
    - **ASR:** Model Transformer Encoder-Decoder mengubah audio menjadi teks.
    - **Transliterasi:** PyKakasi untuk konversi Kanji/Kana ke Romaji.
    - **Perbandingan Teks:** Menggunakan pustaka `difflib` untuk menyoroti perbedaan antar transkripsi.
    - **Hosting:** Aplikasi web dibangun dengan Streamlit dan di-deploy menggunakan GitHub, dengan dependensi sistem seperti FFmpeg, pkg-config, dan cmake diatur melalui `packages.txt`.

    ### Kontak:
    Untuk pertanyaan atau informasi lebih lanjut, silakan hubungi [Nama Anda/Link GitHub Anda].
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Streamlit_logo_black_text.svg/1200px-Streamlit_logo_black_text.svg.png", width=150)
    st.image("https://huggingface.co/front/assets/huggingface_logo.svg", width=150)


st.markdown("---")
st.caption("Aplikasi ini dibuat dengan Streamlit dan Hugging Face Transformers.")
