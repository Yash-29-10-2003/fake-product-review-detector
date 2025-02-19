#Project14
#a) Concept and Vision: Fake Product Review Detector
#b) Train a RoBERTa model to classify fake vs. genuine reviews.
#c) Libraries: Hugging Face Transformers, LangChain, OpenAI API, TensorFlow/PyTorch, spaCy, NLTK
#d) Frameworks: Streamlit/Gradio for UI, FastAPI for deployment.
#e) Datasets: Kaggle, Hugging Face Hub, UCI ML Repository.
#f) Evaluation Criteria
    #I. Functionality: Does the project work as intended?
    #II. Code Quality: Readability, modularity, and documentation.
    #III. GenAI Integration: Effective use of generative models (GPT, diffusion models, etc.).
    #IV. Creativity: Unique problem-solving or UI design.
    #V. Scalability: Could the solution handle real-world data? If yes how


import streamlit as st
import requests

# Streamlit UI
st.title("Fake Product Review Detector")
st.write("This application detects whether a product review is genuine or fake.")

st.header("Target Review:")
user_input = st.text_area("Paste the review to be checked:")

# Button to check review
if st.button("Check Review"):
    if user_input:
        # Send request to FastAPI
        response = requests.post("http://127.0.0.1:8000/predict/", json={"text": user_input})
        
        if response.status_code == 200:
            result = response.json()
            st.write(f"**Prediction:** {result['label']} (Confidence: {result['confidence']:.2f})")
        else:
            st.error("Error: Unable to get a response from the server.")
    else:
        st.warning("Please enter a review before checking.")



