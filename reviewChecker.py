import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import nltk
import streamlit as st

class FakeReviewDetector:
    def __init__(self, model_path="roberta_fake_review_model"):                        #Adress of the pre trained model
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = RobertaForSequenceClassification.from_pretrained(model_path)
        self.model.eval()  # Model to eval mode
    
    #Preprocessing the user input text to match the model requirements
    def preprocess(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        return inputs

    #Prediction for the review
    def predict(self, text):
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
#detector = FakeReviewDetector()
#label, confidence = detector.predict("Compact and good build. Value for money, good connectivity and high transfer rate")
#print(f"Prediction: {label}, Confidence: {confidence:.2f}")