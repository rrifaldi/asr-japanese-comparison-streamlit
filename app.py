import streamlit as st
from transformers import pipeline
import torch
import os
import io
import time
import soundfile as sf
import pykakasi
import difflib
from jiwer import wer, cer

# --- FUNGSI UTILITY: HIGHLIGHT PERBEDAAN TEKS (TETAP SAMA) ---
# Fungsi ini relatif stabil, tidak perlu diubah lagi
def highlight_diff(text1, text2, label1="Teks 1", label2="Teks 2"):
    """
    Membandingkan dua teks karakter demi karakter dan menghasilkan HTML
    yang menyoroti perbedaan dengan warna latar belakang transparan.
    """
    text1 = str(text1 if text1 is not None else "")
    text2 = str(text2 if text2 is not None else "")

    matcher = difflib.SequenceMatcher(None, text1, text2)

    diff_html1 = []
    diff_html2 = []

    for opcode, a_start, a_end, b_start, b_end in matcher.get_opcodes():
        if opcode == 'equal':
            diff_html1.append(text1[a_start:a_end])
            diff_html2.append(text2[b_start:b_end])
        elif opcode == 'replace':
            deleted_part = f"<span style='background-color: rgba(255, 120, 120, 0.3);'>{text1[a_start:a_end]}</span>"
            inserted_part = f"<span style='background-color: rgba(120, 255, 120, 0.3);'>{text2[b_start:b_end]}</span>"
            diff_html1.append(deleted_part)
            diff_html2.append(inserted_part)
        elif opcode == 'delete':
            deleted_part = f"<span style='background-color: rgba(255, 120, 120, 0.3); text-decoration: line-through;'>{text1[a_start:a_end]}</span>"
            diff_html1.append(deleted_part)
            diff_html2.append(f"<span style='color: #888888; font-style: italic;'>[{' ' * max(1, a_end - a_start)}]</span>")
        elif opcode == 'insert':
            inserted_part = f"<span style='background-color: rgba(120, 255, 120, 0.3);'><u>{text2[b_start:b_end]}</u></span>"
            diff_html1.append(f"<span style='color: #888888; font-style: italic;'>[{' ' * max(1, b_end - b_start)}]</span>")
            diff_html2.append(inserted_part)

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
        .diff-table span[style*="text-decoration: underline"] {{
            text-decoration: underline;
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


# --- FUNGSI PEMROSESAN AUDIO DAN TRANSKRIPSI ---
def process_audio_with_model(audio_file_path, asr_pipeline, model_label, pykakasi_converter, pykakasi_initialized):
    """
    Memproses file audio menggunakan pipeline ASR yang diberikan
    dan menampilkan hasil transkripsi (Kanji/Kana dan Romaji) serta waktu.
    """
    st.subheader(f"Hasil dari: {model_label}")

    transcription_japanese = ""
    duration_asr = 0.0

    try:
        if hasattr(st, 'status'):
            with st.status(f"⏳ Sedang mentranskripsi audio dengan {model_label}...", expanded=True) as status_box:
                start_time_asr = time.time()
                result = asr_pipeline(audio_file_path)
                transcription_japanese = result["text"]
                end_time_asr = time.time()
                duration_asr = end_time_asr - start_time_asr
                status_box.update(label=f"✅ Transkripsi {model_label} Selesai!", state="complete", expanded=False)
                st.write(f"Waktu Transkripsi: **{duration_asr:.2f} detik**")
        else: # Fallback untuk Streamlit versi lama
            with st.spinner(f"⏳ Sedang mentranskripsi audio dengan {model_label}..."):
                start_time_asr = time.time()
                result = asr_pipeline(audio_file_path)
                transcription_japanese = result["text"]
                end_time_asr = time.time()
                duration_asr = end_time_asr - start_time_asr
                st.success("✅ Transkripsi Jepang Selesai!")
                st.write(f"Waktu Transkripsi: **{duration_asr:.2f} detik**")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat transkripsi dengan {model_label}: {e}")
        st.exception(e)
        transcription_japanese = "Error saat transkripsi."
        duration_asr = 0.0

    st.markdown(f"**Transkripsi dari {model_label}:**")
    with st.expander(f"Lihat detail transkripsi {model_label}"):
        st.markdown("**Kanji/Kana:**")
        st.code(transcription_japanese)
        st.markdown("**Romaji:**")
        # Menggunakan converter yang telah diinisialisasi
        romaji_text = pykakasi_converter.do(transcription_japanese) if pykakasi_initialized else "Transliterasi Romaji tidak tersedia."
        st.code(romaji_text)

    st.write("---")

    return transcription_japanese, duration_asr


# --- KONFIGURASI UMUM DAN INISIALISASI DI AWAL SKRIP ---
# Ini harus di awal agar st.set_page_config tidak error
MODEL_WHISPER_BASE = "openai/whisper-base"
MODEL_ANIME_WHISPER = "litagin/anime-whisper"

# Inisialisasi PYKAKASI DI SINI SEBELUM st.set_page_config
PYKAKASI_INITIALIZED = False
kakasi_converter = None # Nama variabel diubah agar tidak bentrok dengan "converter" di fungsi highlight_diff
try:
    kks_obj = pykakasi.kakasi()
    # Kembali ke setMode untuk kompatibilitas pykakasi v2.x
    # Ini akan memunculkan DeprecationWarning, tapi berfungsi.
    kks_obj.setMode("H", "a") # Hiragana to Alphabet
    kks_obj.setMode("K", "a") # Katakana to Alphabet
    kks_obj.setMode("J", "a") # Kanji to Alphabet
    kks_obj.setMode("r", "Hepburn") # Romaji system (paling umum)
    kakasi_converter = kks_obj.getConverter()
    PYKAKASI_INITIALIZED = True
except Exception as e:
    # st.warning() tidak bisa dipanggil di sini karena belum diinisialisasi UI Streamlit
    print(f"Peringatan: Gagal menginisialisasi pykakasi untuk transliterasi Romaji: {e}")
    print("Transliterasi Romaji tidak akan tersedia. Pastikan `pykakasi` terinstal dengan benar.")
    class DummyConverter: # Dummy converter jika pykakasi gagal
        def do(self, text):
            return text
    kakasi_converter = DummyConverter()


# --- KONFIGURASI HALAMAN STREAMLIT (HARUS JADI YANG PERTAMA) ---
st.set_page_config(layout="wide", page_title="Perbandingan ASR Audio Jepang")

st.title("🗣️ Perbandingan Model ASR Audio Jepang")
st.markdown("Unggah file audio berbahasa Jepang untuk membandingkan transkripsi dari dua model Whisper, serta mendapatkan Romaji dan analisis perbandingan.")
st.markdown("---")


# --- MUAT MODEL ASR (SETELAH st.set_page_config) ---
# Menggunakan cache Streamlit untuk efisiensi
@st.cache_resource
def load_asr_models_cached():
    """Wrapper untuk memuat kedua model ASR dengan caching."""
    # Menentukan perangkat GPU/CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_id = 0 if device == "cuda" else -1

    # Tentukan torch_dtype untuk optimasi GPU (float16)
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    # Fungsi internal untuk memuat satu model
    def _load_single_model(model_name_param):
        st.info(f"Mengunduh dan memuat model: {model_name_param}...")
        try:
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model_name_param,
                device=device_id,
                torch_dtype=torch_dtype,
                model_kwargs={"low_cpu_mem_usage": True, "use_safetensors": True}
            )
            st.success(f"Model '{model_name_param}' berhasil dimuat pada perangkat: {device.upper()}.")
            return pipe
        except Exception as e:
            st.error(f"❌ Gagal memuat model '{model_name_param}': {e}")
            st.info("Ini mungkin karena masalah jaringan, kurang memori, atau model tidak tersedia.")
            st.stop() # Hentikan aplikasi jika model penting tidak bisa dimuat
            return None

    asr_pipeline_base_local = _load_single_model(MODEL_WHISPER_BASE)
    anime_whisper_pipeline_local = _load_single_model(MODEL_ANIME_WHISPER)
    return asr_pipeline_base_local, anime_whisper_pipeline_local

with st.spinner("⏳ Memuat model AI untuk Automatic Speech Recognition (ASR)... Ini mungkin butuh beberapa saat, tergantung ukuran model dan koneksi internet."):
    asr_pipeline_base, anime_whisper_pipeline = load_asr_models_cached()


# Inisialisasi session_state untuk menyimpan hasil transkripsi dan durasi
if 'base_transcript' not in st.session_state:
    st.session_state.base_transcript = ""
if 'anime_transcript' not in st.session_state:
    st.session_state.anime_transcript = ""
if 'base_duration' not in st.session_state:
    st.session_state.base_duration = 0.0
if 'anime_duration' not in st.session_state:
    st.session_state.anime_duration = 0.0


# Tabs untuk navigasi aplikasi
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
            st.success("✅ File audio berhasil diunggah. Klik 'Mulai Perbandingan!'")
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
            # Panggil fungsi pemrosesan untuk kedua model dan simpan hasilnya
            st.session_state.base_transcript, st.session_state.base_duration = \
                process_audio_with_model(audio_path, asr_pipeline_base, "OpenAI Whisper (Base)", kakasi_converter, PYKAKASI_INITIALIZED)
            st.session_state.anime_transcript, st.session_state.anime_duration = \
                process_audio_with_model(audio_path, anime_whisper_pipeline, "litagin/anime-whisper", kakasi_converter, PYKAKASI_INITIALIZED)

            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            # --- TAMPILKAN KESIMPULAN OTOMATIS ---
            # Pastikan kedua transkripsi tidak error sebelum melanjutkan ke perbandingan
            if st.session_state.base_transcript and st.session_state.base_transcript != "Error saat transkripsi." and \
               st.session_state.anime_transcript and st.session_state.anime_transcript != "Error saat transkripsi.":

                st.markdown("### Perbandingan Langsung Transkripsi (Kanji/Kana)")
                diff_output_html = highlight_diff(st.session_state.base_transcript, st.session_state.anime_transcript, "Whisper Base", "Anime-Whisper")
                st.markdown(diff_output_html, unsafe_allow_html=True)

                st.markdown("---")
                st.header("3. Kesimpulan Perbandingan Otomatis")

                # Perbandingan Waktu Transkripsi
                st.subheader("📊 Perbandingan Waktu Transkripsi:")
                time_diff = abs(st.session_state.base_duration - st.session_state.anime_duration)
                
                faster_model = "OpenAI Whisper (Base)" if st.session_state.base_duration < st.session_state.anime_duration else "litagin/anime-whisper"
                slower_model = "litagin/anime-whisper" if st.session_state.base_duration < st.session_state.anime_duration else "OpenAI Whisper (Base)"
                
                speed_diff_percent = 0
                if max(st.session_state.base_duration, st.session_state.anime_duration) > 0:
                    speed_diff_percent = (time_diff / max(st.session_state.base_duration, st.session_state.anime_duration)) * 100

                st.info(f"**{faster_model}** menyelesaikan transkripsi dalam **{min(st.session_state.base_duration, st.session_state.anime_duration):.2f} detik**, sedangkan **{slower_model}** membutuhkan **{max(st.session_state.base_duration, st.session_state.anime_duration):.2f} detik**. Ini berarti {faster_model} lebih cepat sekitar **{speed_diff_percent:.2f}%** pada audio ini.")

                # Perbandingan Akurasi (Character Error Rate - CER)
                st.subheader("⚖️ Perbandingan Kesamaan Teks (antara kedua model):")
                ref_text_cleaned = st.session_state.base_transcript.strip()
                hyp_text_cleaned = st.session_state.anime_transcript.strip()

                if ref_text_cleaned or hyp_text_cleaned:
                    try:
                        character_error_rate = cer([ref_text_cleaned], [hyp_text_cleaned]) * 100
                        st.metric("Character Error Rate (CER)", f"{character_error_rate:.2f}%", help="Persentase karakter yang berbeda antara dua transkripsi. Dihitung sebagai (Substitusi + Penghapusan + Penambahan) / Total Karakter Referensi. Semakin rendah, semakin mirip.")

                        matcher = difflib.SequenceMatcher(None, ref_text_cleaned, hyp_text_cleaned)
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

                        st.markdown("#### 📝 Ringkasan Umum:")
                        summary_text = ""
                        if character_error_rate < 5:
                            summary_text += "Kedua model menghasilkan transkripsi yang sangat mirip dengan tingkat kesalahan karakter yang rendah. Ini menunjukkan konsistensi tinggi."
                        elif character_error_rate < 20:
                            summary_text += "Kedua model menunjukkan kemiripan yang cukup, namun ada beberapa perbedaan karakter. Mungkin salah satu model lebih baik untuk nuansa tertentu atau istilah spesifik."
                        else:
                            summary_text += "Ada perbedaan signifikan antara transkripsi kedua model. Ini mungkin mengindikasikan bahwa model 'litagin/anime-whisper' memiliki bias yang berbeda atau dilatih pada dataset yang sangat spesifik (anime), yang mungkin tidak cocok untuk semua jenis audio Jepang, atau sebaliknya."
                        
                        summary_text += f"\n\nDalam hal kecepatan, **{faster_model}** secara signifikan lebih cepat daripada **{slower_model}** untuk audio ini."

                        st.markdown(summary_text)

                    except ValueError as ve:
                        st.warning(f"Tidak dapat menghitung CER atau analisis perbedaan: {ve}. Ini mungkin terjadi jika transkripsi menghasilkan teks yang sangat tidak biasa atau kosong.")
                        st.info("Pastikan output transkripsi adalah teks yang valid. Anda dapat memeriksa detail log untuk informasi lebih lanjut.")
                else:
                    st.warning("Tidak dapat menghitung Character Error Rate (CER) atau ringkasan karena kedua transkripsi (setelah pembersihan) kosong.")

            else:
                st.error("❌ Gagal membuat kesimpulan perbandingan otomatis karena salah satu atau kedua model gagal dalam transkripsi.")
        else:
            st.warning("Silakan unggah file audio Anda terlebih dahulu di bagian '1. Unggah File Audio' untuk memulai perbandingan.")

with tab2: # Konten tentang proyek
    st.header("Tentang Proyek Ini")
    st.markdown("""
    Aplikasi ini dirancang untuk membandingkan kinerja dua model Automatic Speech Recognition (ASR) populer, **OpenAI Whisper (Base)** dan **litagin/anime-whisper**, dalam mentranskripsi audio berbahasa Jepang. Selain menampilkan transkripsi asli dalam aksara Jepang (Kanji/Kana), aplikasi ini juga menyediakan transliterasi ke Romaji dan analisis perbandingan mendalam.

    ### Model ASR yang Digunakan:
    - **OpenAI Whisper (Base):** Model ASR umum yang dikembangkan oleh OpenAI. Dikenal dengan kemampuannya yang serbaguna di berbagai bahasa dan akurasi yang baik. Versi 'Base' umumnya lebih cepat karena ukurannya yang lebih kecil.
    - **litagin/anime-whisper:** Model ASR yang di-fine-tune secara khusus untuk audio dari konten anime. Model ini diharapkan lebih akurat untuk dialog anime tetapi mungkin lebih lambat jika ukurannya lebih besar (seringkali berbasis pada Whisper Large).

    ### Fitur Utama:
    - **Pengunggahan Audio:** Pengguna dapat mengunggah file audio mereka sendiri (.wav, .mp3, .flac).
    - **Transkripsi Bilingual:** Menampilkan transkripsi dalam aksara Jepang (Kanji/Kana) dan transliterasi ke Romaji.
    - **Perbandingan Visual:** Menggunakan penyorotan perbedaan karakter-demi-karakter untuk memvisualisasikan ketidaksesuaian antara hasil transkripsi kedua model.
    - **Kesimpulan Otomatis:** Menyediakan ringkasan kuantitatif mengenai:
        - **Kecepatan Transkripsi:** Perbandingan waktu yang dibutuhkan kedua model.
        - **Kesamaan Teks (CER):** Character Error Rate (CER) yang menunjukkan seberapa mirip kedua transkripsi satu sama lain.
        - **Analisis Detail Perbedaan:** Jumlah penggantian, penghapusan, dan penambahan karakter.
        - **Ringkasan Narasi:** Penjelasan singkat mengenai implikasi dari hasil perbandingan.

    ### Teknologi yang Digunakan:
    - **Streamlit:** Framework Python untuk membangun aplikasi web interaktif.
    - **Hugging Face Transformers:** Pustaka terkemuka untuk model AI, termasuk Whisper.
    - **PyKakasi:** Pustaka Python untuk transliterasi Jepang ke Romaji.
    - **Jiwer:** Pustaka untuk menghitung metrik Word Error Rate (WER) dan Character Error Rate (CER).
    - **Difflib:** Pustaka standar Python untuk membandingkan urutan (digunakan untuk visualisasi perbedaan teks).

    ### Catatan Penting:
    - Kinerja (terutama kecepatan) sangat bergantung pada sumber daya komputasi yang tersedia (GPU direkomendasikan).
    - CER yang ditampilkan adalah perbandingan antara kedua model, bukan terhadap 'ground truth' dari audio yang sebenarnya.

    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Streamlit_logo_black_text.svg/1200px-Streamlit_logo_black_text.svg.png", width=150)
    st.image("https://huggingface.co/front/assets/huggingface_logo.svg", width=150)


st.markdown("---")
st.caption("Aplikasi ini dikembangkan dengan Streamlit dan Hugging Face Transformers.")
