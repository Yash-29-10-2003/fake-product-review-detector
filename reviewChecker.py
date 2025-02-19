import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import nltk
import streamlit as st

class FakeReviewDetector:
    def __init__(self, model_path="roberta_fake_review_model"):
        """Initialize the Fake Review Detector with a pre-trained RoBERTa model."""
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = RobertaForSequenceClassification.from_pretrained(model_path)
        self.model.eval()  # Set model to evaluation mode
    
    def preprocess(self, text):
        """Preprocess the input text (tokenization, truncation, padding)."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        return inputs

    def predict(self, text):
        """Predict whether a review is fake or genuine."""
        inputs = self.preprocess(text)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
        
        label = "Genuine Review" if prediction == 1 else "Fake Review"
        confidence = probabilities[0][prediction].item()
        return label, confidence

# Testing:
detector = FakeReviewDetector()
label, confidence = detector.predict("Compact and good build. Value for money, good connectivity and high transfer rate")
print(f"Prediction: {label}, Confidence: {confidence:.2f}")