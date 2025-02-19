# fake-product-review-detector
This project was made as an assignment submission for cyfutre.

### Problem Statement :
###### Project14
  a) Concept and Vision: Fake Product Review Detector.
  b) Train a RoBERTa model to classify fake vs. genuine reviews.
  c) Libraries: Hugging Face Transformers, LangChain, OpenAI API, TensorFlow/PyTorch, spaCy, NLTK
  d) Frameworks: Streamlit/Gradio for UI, FastAPI for deployment.
  e) Datasets: Kaggle, Hugging Face Hub, UCI ML Repository.
  f) Evaluation Criteria
    I. Functionality: Does the project work as intended?
    II. Code Quality: Readability, modularity, and documentation.
    III. GenAI Integration: Effective use of generative models (GPT, diffusion models, etc.).
    IV. Creativity: Unique problem-solving or UI design.
    V. Scalability: Could the solution handle real-world data? If yes how.


### Proposed Solution :

The code is divided into 3 main parts:
- The Streamlit UI for user input and output generation. (main.py)
- Pretraining of the fake review dector RoBERTa model. (fake_review_detector.ipynb)
- Calling the pretrained model from colab into the local machine. (reviewChecker.py)

#### Streamlit UI

The streamlit UI is fairly simple with a breif description of the product, a text field and a button.
<img width="1407" alt="Screenshot 2025-02-19 at 7 20 24 PM" src="https://github.com/user-attachments/assets/54f92190-6afe-465e-97a8-0f1dd6ac105f" />

The user input in the text field mentioned sends the query to the reviewChecker.py class and its functions give out an output based on if the review is computer generated or an authentic human review.

