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


#Importing necessary classes
import streamlit as st
from reviewChecker import FakeReviewDetector 

detector = FakeReviewDetector()

#Main UI using streamlit
st.title("Fake Product Review Detector")
st.write("This application detects whether a product review is genuine or fake.")

st.header("Target Review:")
user_input = st.text_area("Paste the review to be checked:")

# Button to check review
if st.button("Check Review"):
    if user_input:
        label, confidence = detector.predict(user_input)
        st.write(f"Prediction: **{label}**")
        #st.write(f"Confidence: **{confidence:.2f}**")       #could be used to also display the confidence
    else:
        st.warning("Please enter a review before checking.")



