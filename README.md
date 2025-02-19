# Fake Product Review Detector  

This project was developed as an assignment submission for Cyfuture 2025 placement drive in Amity University, Noida.
Submission by - Yash Singh (A2305221635)

---

### Problem Statement  

###### Project 14  

- **Concept and Vision:** Fake Product Review Detector.  
- **Objective:** Train a RoBERTa model to classify fake vs. genuine reviews.  
- **Libraries:** Hugging Face Transformers, LangChain, OpenAI API, TensorFlow/PyTorch, spaCy, NLTK.  
- **Frameworks:** Streamlit/Gradio for UI, FastAPI for deployment.  
- **Datasets:** Kaggle, Hugging Face Hub, UCI ML Repository.  
- **Evaluation Criteria:**  
  1. **Functionality:** Does the project work as intended?  
  2. **Code Quality:** Readability, modularity, and documentation.  
  3. **GenAI Integration:** Effective use of generative models (GPT, diffusion models, etc.).  
  4. **Creativity:** Unique problem-solving or UI design.  
  5. **Scalability:** Can the solution handle real-world data? If so, how?  

---

### Proposed Solution  

The code is divided into three main parts:  
- **Streamlit UI** for user input and output generation (**main.py**).  
- **Pretraining the Fake Review Detector RoBERTa model** (**fake_review_detector.ipynb**).  
- **Loading the pretrained model from Colab into the local machine** (**reviewChecker.py**).  

---

### Streamlit UI and Review Checking  

The Streamlit UI is simple, featuring a brief description of the product, a text field, and a button.  

![Screenshot](https://github.com/user-attachments/assets/54f92190-6afe-465e-97a8-0f1dd6ac105f)  

When the user enters a review in the text field, the query is sent to the `api.py` class, which sends a request to the Fast API instance running in the background and it returns the status of the review and the confidence.

---

### Fine-Tuning the RoBERTa Model for Fake Review Detection  

The dataset used for fine-tuning was obtained from Kaggle: [Fake Reviews Dataset](https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset).  

In the Jupyter notebook, after installing dependencies and importing the required classes, the following preprocessing steps were performed:  
- Dropped missing values.  
- Extracted relevant columns.  
- Converted the label column into binary format (0 and 1) for better accessibility.  
- Concatenated the category, rating, and review text for better context.  

For tokenization, **RobertaTokenizer** was used to preprocess text data, converting input text into a tokenized format with padding and truncation. This tokenized data was then converted into Hugging Face's dataset format for training.  

During training:  
- The **pretrained RoBERTa base model** was loaded.  
- Basic training arguments were set.  
- The **Trainer API** was used for training and evaluation.  
- The trained model and tokenizer were saved for use in the project.  

Folowing were the recorded accuracies and loss after the training for the 3 epochs:

<img width="860" alt="Screenshot 2025-02-19 at 4 19 45 PM" src="https://github.com/user-attachments/assets/e41a2b2c-c8c4-4ac0-952e-cf897e9e80db" />

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
2. Run the Fast API backend.
3. Run the Streamlit application using the appropriate command.
4. Open the provided URL in your web browser to interact with the app.
5. Paste a product review into the text area and click "Check Review" to see the prediction.


---
