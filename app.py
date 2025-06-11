import streamlit as st
from transformers import pipeline
import torch
import os
import io
import time
import soundfile as sf
import pykakasi
import difflib
from jiwer import wer, cer # Import untuk menghitung Word Error Rate dan Character Error Rate

# --- KONFIGURASI MODEL ---
# Lebih baik gunakan model yang seukuran atau mendekati seukuran untuk perbandingan yang lebih adil
# Whisper Base lebih kecil. Anime-Whisper mungkin di-fine-tune dari model yang lebih besar.
# Jika Anda ingin membandingkan akurasi di domain anime, ini mungkin pilihan yang tepat,
# tapi jika kecepatan adalah metrik utama, perhatikan ukuran modelnya.
MODEL_WHISPER_BASE = "openai/whisper-base"
MODEL_ANIME_WHISPER = "litagin/anime-whisper" # Model ini cenderung lebih besar dari 'base'

# Inisialisasi Kakasi (untuk konversi Jepang ke Romaji)
kks = pykakasi.kakasi()
kks.setMode("H", "a") # Hiragana to Alphabet
kks.setMode("K", "a") # Katakana to Alphabet
kks.setMode("J", "a") # Kanji to Alphabet
kks.setMode("r", "Hepburn") # Romaji system (paling umum)
converter = kks.getConverter()


# --- CACHE MODEL ---
@st.cache_resource
def load_asr_model(model_name):
    device = 0 if torch.cuda.is_available() else -1
    # Tentukan torch_dtype untuk optimasi GPU (float16)
    torch_dtype = torch.float16 if torch.cuda.is_available() and device != -1 else torch.float32

    # Menambahkan low_cpu_mem_usage dan use_safetensors untuk efisiensi memori
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=device,
        torch_dtype=torch_dtype,
        # Menambahkan parameter berikut untuk membantu loading model besar
        model_kwargs={"low_cpu_mem_usage": True, "use_safetensors": True}
    )
    return asr_pipe


# --- UTILITY FUNCTION FOR TEXT COMPARISON (REVISED) ---
def highlight_diff(text1, text2, label1="Teks 1", label2="Teks 2"):
    """
    Compares two texts character by character and returns HTML that highlights the differences.
    This version aims for better readability of highlighted Japanese text.
    """
    # Use SequenceMatcher on character level for better granularity
    # Memastikan tidak ada None atau non-string input
    text1 = str(text1) if text1 is not None else ""
    text2 = str(text2) if text2 is not None else ""

    matcher = difflib.SequenceMatcher(None, text1, text2)

    diff_html1 = []
    diff_html2 = []

    for opcode, a_start, a_end, b_start, b_end in matcher.get_opcodes():
        if opcode == 'equal':
            diff_html1.append(text1[a_start:a_end])
            diff_html2.append(text2[b_start:b_end])
        elif opcode == 'replace':
            deleted_from_1 = f"<span style='background-color: rgba(255, 120, 120, 0.3);'>{text1[a_start:a_end]}</span>" # Merah muda transparan
            inserted_in_2 = f"<span style='background-color: rgba(120, 255, 120, 0.3);'>{text2[b_start:b_end]}</span>" # Hijau muda transparan

            diff_html1.append(deleted_from_1)
            diff_html2.append(inserted_in_2)
        elif opcode == 'delete':
            deleted_from_1 = f"<span style='background-color: rgba(255, 120, 120, 0.3); text-decoration: line-through;'>{text1[a_start:a_end]}</span>"
            diff_html1.append(deleted_from_1)
            diff_html2.append(f"<span style='color: #888888; font-style: italic;'>[{' ' * max(1, a_end - a_start)}]</span>") # placeholder for missing text
        elif opcode == 'insert':
            inserted_in_2 = f"<span style='background-color: rgba(120, 255, 120, 0.3);'>__{text2[b_start:b_end]}__</span>" # Tambahkan underline atau tebal
            diff_html1.append(f"<span style='color: #888888; font-style: italic;'>[{' ' * max(1, b_end - b_start)}]</span>") # placeholder
            diff_html2.append(inserted_in_2)

    html_output = f"""
    <style>
        .diff-table {{
            width: 100%;
            border-collapse: collapse;
            font-family: 'Noto Sans JP', 'Segoe UI', 'Meiryo', 'Yu Gothic', sans-serif;
            font-size: 1.1em;
            line-height: 1.5;
            margin-top: 15px;
            margin-bottom: 15px;
        }}
        .diff-table th, .diff-table td {{
            border: 1px solid #444;
            padding: 10px;
            text-align: left;
            vertical-align: top;
            word-break: break-word;
        }}
        .diff-table th {{
            background-color: #333;
            color: white;
        }}
        .diff-table td {{
            background-color: #222;
            color: #eee;
        }}
        .diff-table span[style*="background-color: rgba(255, 120, 120, 0.3)"] {{
            background-color: rgba(255, 120, 120, 0.3);
            color: #eee;
        }}
        .diff-table span[style*="background-color: rgba(120, 255, 120, 0.3)"] {{
            background-color: rgba(120, 255, 120, 0.3);
            color: #eee;
        }}
        .diff-table span[style*="text-decoration: line-through"] {{
            text-decoration: line-through;
            opacity: 0.7;
        }}
        .diff-table span[style*="font-style: italic"] {{
            color: #888888;
            font-style: italic;
        }}
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
                <td>{''.join(diff_html1)}</td>
                <td>{''.join(diff_html2)}</td>
            </tr>
        </tbody>
    </table>
    """
    return html_output


# --- FUNGSI PEMROSESAN AUDIO ---
def convert_to_romaji(text_japanese):
    """Mengonversi teks Jepang (Kanji/Kana) ke Romaji."""
    if not text_japanese:
        return ""
    return converter.do(text_japanese)

def process_audio_with_model(audio_path, asr_pipeline, model_label):
    """Memproses audio dengan model ASR tertentu."""
    st.subheader(f"Hasil dari: {model_label}")

    transcription_japanese = "Error saat transkripsi."
    duration_asr = 0.0 # Inisialisasi durasi

    try:
        if hasattr(st, 'status'):
            with st.status(f"⏳ Sedang mentranskripsi audio dengan {model_label}...", expanded=True) as status_box:
                start_time_asr = time.time()
                result = asr_pipeline(audio_path)
                transcription_japanese = result["text"]
                end_time_asr = time.time()
                duration_asr = end_time_asr - start_time_asr
                status_box.update(label=f"✅ Transkripsi {model_label} Selesai!", state="complete", expanded=False)
                st.write(f"Waktu Transkripsi: **{duration_asr:.2f} detik**")
        else:
            with st.spinner(f"⏳ Sedang mentranskripsi audio dengan {model_label}..."):
                start_time_asr = time.time()
                result = asr_pipeline(audio_path)
                transcription_japanese = result["text"]
                end_time_asr = time.time()
                duration_asr = end_time_asr - start_time_asr
                st.success("✅ Transkripsi Jepang Selesai!")
                st.write(f"Waktu Transkripsi: **{duration_asr:.2f} detik**")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat transkripsi {model_label}: {e}")
        st.exception(e)

    st.markdown(f"**Transkripsi dari {model_label}:**")
    with st.expander(f"Lihat detail transkripsi {model_label}"):
        st.markdown("**Kanji/Kana:**")
        st.code(transcription_japanese)
        st.markdown("**Romaji:**")
        romaji_text = convert_to_romaji(transcription_japanese)
        st.code(romaji_text)

    st.write("---")

    return transcription_japanese, duration_asr # Mengembalikan transkripsi dan durasi


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

# Inisialisasi session_state untuk menyimpan hasil transkripsi dan durasi
if 'base_transcript' not in st.session_state:
    st.session_state.base_transcript = ""
if 'anime_transcript' not in st.session_state:
    st.session_state.anime_transcript = ""
if 'base_duration' not in st.session_state:
    st.session_state.base_duration = 0.0
if 'anime_duration' not in st.session_state:
    st.session_state.anime_duration = 0.0


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
            # Sekarang fungsi ini mengembalikan tuple (transkripsi, durasi)
            st.session_state.base_transcript, st.session_state.base_duration = process_audio_with_model(audio_path, asr_pipeline_base, "OpenAI Whisper (Base)")
            st.session_state.anime_transcript, st.session_state.anime_duration = process_audio_with_model(audio_path, anime_whisper_pipeline, "litagin/anime-whisper")

            # Hapus file sementara setelah selesai memproses
            if os.path.exists(audio_path):
                os.remove(audio_path)

            # --- Bagian KESIMPULAN OTOMATIS ---
            if st.session_state.base_transcript and st.session_state.anime_transcript and \
               st.session_state.base_transcript != "Error saat transkripsi." and \
               st.session_state.anime_transcript != "Error saat transkripsi.":

                st.markdown("### Perbandingan Langsung Transkripsi (Kanji/Kana)")
                diff_output_html = highlight_diff(st.session_state.base_transcript, st.session_state.anime_transcript, "Whisper Base", "Anime-Whisper")
                st.markdown(diff_output_html, unsafe_allow_html=True)

                st.markdown("---")
                st.header("3. Kesimpulan Perbandingan Otomatis")

                # Perbandingan Waktu
                st.subheader("Perbandingan Waktu Transkripsi:")
                time_diff = abs(st.session_state.base_duration - st.session_state.anime_duration)
                if st.session_state.base_duration < st.session_state.anime_duration:
                    faster_model = "OpenAI Whisper (Base)"
                    slower_model = "litagin/anime-whisper"
                    speed_diff_percent = (time_diff / st.session_state.anime_duration) * 100
                else:
                    faster_model = "litagin/anime-whisper"
                    slower_model = "OpenAI Whisper (Base)"
                    speed_diff_percent = (time_diff / st.session_state.base_duration) * 100

                st.info(f"**{faster_model}** menyelesaikan transkripsi dalam **{min(st.session_state.base_duration, st.session_state.anime_duration):.2f} detik**, sedangkan **{slower_model}** membutuhkan **{max(st.session_state.base_duration, st.session_state.anime_duration):.2f} detik**. Ini berarti {faster_model} lebih cepat sekitar **{speed_diff_percent:.2f}%** pada audio ini.")

                # Perbandingan Akurasi (dengan asumsi salah satu model adalah "ground truth" untuk WER/CER)
                # Catatan: Tanpa ground truth yang sebenarnya, WER/CER ini hanya menunjukkan
                # seberapa berbeda kedua transkripsi satu sama lain, BUKAN akurasi terhadap audio.
                # Untuk WER/CER yang sesungguhnya, Anda perlu memiliki teks yang benar.
                
                # Kita akan menganggap "Whisper Base" sebagai referensi untuk menghitung WER/CER Anime-Whisper terhadapnya.
                # Atau lebih baik, kita hitung CER/WER antara kedua model.
                
                st.subheader("Perbandingan Kesamaan Teks (antara kedua model):")
                # jiwer butuh string kosong jika input None
                ref_text = st.session_state.base_transcript if st.session_state.base_transcript else ""
                hyp_text = st.session_state.anime_transcript if st.session_state.anime_transcript else ""

                # Tokenisasi teks untuk WER (jiwer secara internal menangani ini, tapi pastikan input bersih)
                # Untuk bahasa Jepang, WER mungkin kurang relevan karena tidak ada spasi antar kata.
                # CER (Character Error Rate) seringkali lebih cocok.
                # Mari kita gunakan CER.
                
                # Fungsi jiwer menerima list of strings.
                # Misalnya, split teks menjadi karakter untuk CER.
                # Atau jika ingin WER, split menjadi "kata" (walaupun tidak standar di Jepang)
                
                # Untuk CER, kita bisa langsung pakai string
                # Misalnya, untuk membandingkan Anime-Whisper terhadap Whisper Base sebagai referensi
                character_error_rate = cer(list(ref_text), list(hyp_text)) * 100
                st.metric("Character Error Rate (CER)", f"{character_error_rate:.2f}%", help="Persentase karakter yang berbeda antara dua transkripsi (dihitung terhadap Whisper Base sebagai referensi). Semakin rendah, semakin mirip.")

                # Analisis Perbedaan (berdasarkan difflib opcode)
                matcher = difflib.SequenceMatcher(None, ref_text, hyp_text)
                replacements = 0
                deletions = 0
                insertions = 0
                
                for opcode, a_start, a_end, b_start, b_end in matcher.get_opcodes():
                    if opcode == 'replace':
                        replacements += 1
                    elif opcode == 'delete':
                        deletions += 1
                    elif opcode == 'insert':
                        insertions += 1
                
                st.info(f"Ditemukan **{replacements}** penggantian, **{deletions}** penghapusan, dan **{insertions}** penambahan karakter antara kedua transkripsi.")

                st.markdown("#### Ringkasan Umum:")
                summary = ""
                if character_error_rate < 5:
                    summary += "Kedua model menghasilkan transkripsi yang sangat mirip dengan tingkat kesalahan karakter yang rendah. "
                elif character_error_rate < 20:
                    summary += "Kedua model menunjukkan kemiripan yang cukup, namun ada beberapa perbedaan karakter. "
                else:
                    summary += "Ada perbedaan signifikan antara transkripsi kedua model. "
                
                summary += f"Dalam hal kecepatan, **{faster_model}** secara signifikan lebih cepat daripada **{slower_model}** untuk audio ini."

                st.markdown(summary)

            else:
                st.warning("Tidak dapat membuat kesimpulan karena salah satu atau kedua model gagal atau menghasilkan error saat transkripsi.")
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
    - **Kesimpulan Otomatis:** Menyediakan ringkasan perbandingan termasuk waktu transkripsi dan tingkat kesalahan karakter.

    ### Metode:
    - **ASR:** Model Transformer Encoder-Decoder mengubah audio menjadi teks.
    - **Transliterasi:** PyKakasi untuk konversi Kanji/Kana ke Romaji.
    - **Perbandingan Teks:** Menggunakan pustaka `difflib` untuk menyoroti perbedaan antar transkripsi dan `jiwer` untuk menghitung Character Error Rate (CER).
    - **Hosting:** Aplikasi web dibangun dengan Streamlit dan di-deploy menggunakan GitHub, dengan dependensi sistem seperti FFmpeg, pkg-config, dan cmake diatur melalui `packages.txt`.

    ### Kontak:
    Untuk pertanyaan atau informasi lebih lanjut, silakan hubungi [Nama Anda/Link GitHub Anda].
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Streamlit_logo_black_text.svg/1200px-Streamlit_logo_black_text.svg.png", width=150)
    st.image("https://huggingface.co/front/assets/huggingface_logo.svg", width=150)


st.markdown("---")
st.caption("Aplikasi ini dibuat dengan Streamlit dan Hugging Face Transformers.")
