import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class IndoBertSentiment:
    def __init__(self, model_name="indobenchmark/indobert-base-p1"):
        """Memuat model dan tokenizer IndoBERT."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)  # Positif, Netral, Negatif
        self.model.eval()

    def predict(self, text: str) -> str:
        """Memprediksi sentimen dari teks."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            label = torch.argmax(probabilities).item()

        sentiment_labels = {0: "Negatif", 1: "Netral", 2: "Positif"}
        return sentiment_labels[label]