#THIS IS A UPDATED BACKEND WHICH USES FAST API FOR THE BACKGROUND REQUESTS.

from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

app = FastAPI()

# Loading model and tokenizer
model_path = "roberta_fake_review_model"  
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)
model.eval()  # eval mode

# request body
class ReviewInput(BaseModel):
    text: str

def preprocess(text):
    """Tokenize input text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    return inputs

def predict(text):     #Predicting the fakeness of review
    inputs = preprocess(text)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
    
    label = "Genuine Review" if prediction == 1 else "Fake Review"
    confidence = probabilities[0][prediction].item()
    return {"label": label, "confidence": confidence}

# API Endpoint
@app.post("/predict/")
def predict_review(review: ReviewInput):
    return predict(review.text)
