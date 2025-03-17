import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from youtube_crawling import YoutubeCrawling
from comment_preprocessing import CommentPreprocessing
from sentiment_analysis import IndoBertSentiment

# Konfigurasi API YouTube
YOUTUBE_API_KEY = "AIzaSyCwGg_Quc1Ie99GeTvaoF_BHpqWcj0sdDc"

# Inisialisasi objek
crawler = YoutubeCrawling(YOUTUBE_API_KEY)
preprocessor = CommentPreprocessing()
sentiment_analyzer = IndoBertSentiment()

# Streamlit UI
st.title("Analisis Sentimen Komentar YouTube dengan IndoBERT")
st.write("Masukkan link video YouTube dan jumlah komentar yang ingin dianalisis.")

# Input pengguna
video_url = st.text_input("Masukkan URL Video YouTube")
num_comments = st.number_input("Jumlah komentar yang ingin diambil", min_value=1, value=100, step=10)

if st.button("Analisis Sentimen"):
    if not video_url:
        st.error("Harap masukkan URL YouTube.")
    else:
        st.info("Mengambil komentar dari YouTube...")
        try:
            comments = crawler.get_comments(video_url, max_comments=num_comments)

            if len(comments) == 0:
                st.warning("Tidak ada komentar yang ditemukan.")
            elif len(comments) < num_comments:
                st.warning(f"Hanya ditemukan {len(comments)} komentar dari {num_comments} yang diminta.")

            st.success(f"Ditemukan {len(comments)} komentar. Memproses sentimen...")

            # Cleansing data
            cleaned_comments = [preprocessor.clean_text(comment) for comment in comments]

            # Analisis sentimen
            results = [{"Komentar": comment, "Sentimen": sentiment_analyzer.predict(comment)} for comment in cleaned_comments]
            df = pd.DataFrame(results)

            # Menampilkan hasil
            st.write("### Hasil Analisis Sentimen")
            st.dataframe(df)

            # Visualisasi hasil
            sentiment_counts = df["Sentimen"].value_counts()
            fig, ax = plt.subplots()
            colors = {
                "Positif": "green",
                "Netral": "gray",
                "Negatif": "red"
            }
            ax.bar(sentiment_counts.index, sentiment_counts.values, color=[colors[label] for label in sentiment_counts.index])
            ax.set_title("Distribusi Sentimen")
            ax.set_xlabel("Sentimen")
            ax.set_ylabel("Jumlah Komentar")
            st.pyplot(fig)

            # Download hasil sebagai CSV
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(label="📥 Download CSV", data=csv, file_name="sentimen_youtube.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
