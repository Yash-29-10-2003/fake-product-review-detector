# fake-product-review-detector
This project was made as an assignment submission for cyfutre.

---
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
---

### Proposed Solution :

The code is divided into 3 main parts:
- The Streamlit UI for user input and output generation. (main.py)
- Pretraining of the fake review dector RoBERTa model. (fake_review_detector.ipynb)
- Calling the pretrained model from colab into the local machine. (reviewChecker.py)


---
#### Streamlit UI and Checking Reviews

The streamlit UI is fairly simple with a breif description of the product, a text field and a button.
<img width="1407" alt="Screenshot 2025-02-19 at 7 20 24 PM" src="https://github.com/user-attachments/assets/54f92190-6afe-465e-97a8-0f1dd6ac105f" />

The user input in the text field mentioned sends the query to the reviewChecker.py class and its functions give out an output based on if the review is computer generated or an authentic human review.

#### Fine tuning the RoBERTa model for Fake Review Detection 

I used the fake review detection dataset from kaggle. [https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset]

In the python notebook after installing the dependencies and importing required classes, I first did some dataset preprocessing where i dropped the missing values and extracted the necessary columns from the dataset. I also converted the label column to 0 and 1's for better accessibility and concatenated the category, rating and the review text into the text for better context.

For tokenization I used the RobertaTokenizer to preprocess text data. Which converts the input text into tokenized format with padding and truncation. This tokenized data was further converted into Hugging Face dataset format for training.

Then for training, after loading the pre trained roberta base model, and setting up basic training arguments I used the trainer API to train and evaluate the model and further saved the model and tokenizer for use in the project.

---

## Installation and Usage

Following are the steps to follow to use the application in any local machine.
To set up the project, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies using the provided requirements file.
3. Download the dataset and place it in the root directory of the project.

---

### Training the Model

1. Open the Jupyter Notebook provided in the project.
2. Run the notebook to preprocess the data, train the RoBERTa model, and save the trained model to a specified directory.

### Running the Streamlit App

1. Navigate to the project directory in your terminal.
2. Run the Streamlit application using the appropriate command.
3. Open the provided URL in your web browser to interact with the app.
4. Paste a product review into the text area and click "Check Review" to see the prediction.

---
