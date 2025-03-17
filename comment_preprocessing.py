import re
import string

class CommentPreprocessing:
    @staticmethod
    def remove_urls(text: str) -> str:
        """Menghapus URL dari teks."""
        return re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    @staticmethod
    def remove_special_characters(text: str) -> str:
        """Menghapus karakter khusus dan angka."""
        return re.sub(r"[^a-zA-Z\s]", "", text)

    @staticmethod
    def remove_extra_spaces(text: str) -> str:
        """Menghapus spasi berlebihan."""
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def clean_text(text: str) -> str:
        """Melakukan pembersihan lengkap pada teks."""
        text = text.lower()  # Ubah ke huruf kecil
        text = CommentPreprocessing.remove_urls(text)
        text = CommentPreprocessing.remove_special_characters(text)
        text = CommentPreprocessing.remove_extra_spaces(text)
        return text
