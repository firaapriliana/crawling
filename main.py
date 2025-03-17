import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from youtube_crawling import YoutubeCrawling
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam

from youtube_crawling import YoutubeCrawling

# Konfigurasi API YouTube
YOUTUBE_API_KEY = "AIzaSyCwGg_Quc1Ie99GeTvaoF_BHpqWcj0sdDc"

# Inisialisasi objek
crawler = YoutubeCrawling(YOUTUBE_API_KEY)

# Inisialisasi Stemmer dan Stopwords
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('indonesian'))
stemmer = StemmerFactory().create_stemmer()

# Fungsi pre-processing
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z ]', '', text.lower())  # Hapus karakter non-alfabet
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]  # Hapus stopwords
    tokens = [stemmer.stem(word) for word in tokens]  # Stemming
    return ' '.join(tokens)

# Load Model LSTM
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=200),
    LSTM(100, return_sequences=False),
    Dense(3, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Streamlit UI
st.title("Analisis Sentimen Komentar YouTube dengan LSTM")
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
            cleaned_comments = [preprocess_text(comment) for comment in comments]

            # TF-IDF
            vectorizer = TfidfVectorizer(max_features=5000)
            X_tfidf = vectorizer.fit_transform(cleaned_comments).toarray()
            X_padded = pad_sequences(X_tfidf, maxlen=200)

            # Prediksi sentimen
            predictions = model.predict(X_padded)
            sentiment_labels = ['Negatif', 'Netral', 'Positif']
            predicted_sentiments = [sentiment_labels[np.argmax(pred)] for pred in predictions]

            df = pd.DataFrame({"Komentar": comments, "Sentimen": predicted_sentiments})

            # Menampilkan hasil
            st.write("### Hasil Analisis Sentimen")
            st.dataframe(df)

            # Visualisasi hasil dengan Seaborn
            sentiment_counts = df["Sentimen"].value_counts()
            plt.figure(figsize=(6,4))
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=["red", "gray", "green"])
            plt.title("Distribusi Sentimen")
            plt.xlabel("Sentimen")
            plt.ylabel("Jumlah Komentar")
            st.pyplot(plt)

            # Download hasil sebagai CSV
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(label="📥 Download CSV", data=csv, file_name="sentimen_youtube.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
