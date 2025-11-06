# 📩 Spam Classifier

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Libraries](https://img.shields.io/badge/Libraries-NLTK%20%7C%20Scikit--learn%20%7C%20Gensim-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A classic NLP project to classify SMS messages as "Spam" or "Ham." This repository details the entire pipeline, from data cleaning and preprocessing to feature extraction and model comparison. It explores and contrasts multiple techniques, including Bag-of-Words, TF-IDF, and Word2Vec embeddings.

## 📋 Table of Contents

-   [Project Overview](#-project-overview)
-   [Visualizations](#-visualizations)
-   [Models & Performance](#-models--performance)
-   [Technologies Used](#-technologies-used)
-   [Installation](#-installation)
-   [How to Use](#-how-to-use)
-   [Test with Your Own Message](#-test-with-your-own-message)

## 📖 Project Overview

The goal of this project is to build a machine learning model that can accurately distinguish spam (unwanted or unsolicited) messages from ham (legitimate) messages. The dataset used is the **SMSSpamCollection**, a public set of over 5,500 labeled SMS messages.

The notebook explores three primary feature extraction methods:
1.  **Bag-of-Words (BoW):** Treats each message as a collection of its words, disregarding grammar and word order but keeping track of frequency.
2.  **TF-IDF (Term Frequency-Inverse Document Frequency):** Builds on BoW by weighting words based on how important they are to a specific message, not just their frequency in the overall corpus.
3.  **Word2Vec:** A neural network-based technique that learns distributed vector representations of words (embeddings), capturing their semantic meaning and context.

These feature sets are then used to train and evaluate two types of classifiers: **Multinomial Naive Bayes** and **Random Forest**.

## 📊 Visualizations

Exploring the data reveals clear differences between spam and ham messages.

### Word Clouds
The most frequent words for spam and ham messages are starkly different. Spam is dominated by words like "free," "prize," "claim," and "urgent," while ham messages feature common conversational words.

*(<img width="950" height="324" alt="image" src="https://github.com/user-attachments/assets/cca51618-4182-471a-af0c-8ab86f1d945e" />
*

### Message Length Distribution
Spam messages, on average, tend to be significantly longer than ham messages, which are often short and conversational.

*<img width="859" height="547" alt="image" src="https://github.com/user-attachments/assets/f88d406d-99b1-426f-be3e-444aff906525" />
*

### Final Model: Word2Vec + Random Forest
The following plots show the performance of the final model implemented in the notebook (Random Forest classifier trained on average Word2Vec embeddings).

#### Confusion Matrix
The model shows strong performance, correctly identifying most ham and spam messages, though it has more difficulty with false negatives (spam classified as ham).

*<img width="510" height="393" alt="image" src="https://github.com/user-attachments/assets/97c0532a-90a7-4b70-b24a-7bd0025bc72e" />
*

#### ROC Curve & AUC
The model achieves an Area Under the Curve (AUC) of 0.96, indicating excellent separability between the two classes.

*<img width="702" height="547" alt="image" src="https://github.com/user-attachments/assets/96b735a5-182f-4639-8327-245aa8fdabb3" />
*

#### Feature Importance
Since Word2Vec creates 100-dimensional abstract features, the importance plot shows which of these embedding dimensions were most influential for the Random Forest model.

*<img width="822" height="547" alt="image" src="https://github.com/user-attachments/assets/3dfee8be-ee60-460a-88ab-a8dd12b6e551" />
*

## 🚀 Models & Performance

Several models were trained and compared. The classic **Bag-of-Words** model with **Multinomial Naive Bayes** achieved the highest accuracy, demonstrating the power of simple, robust methods on this dataset.

| Vectorization | Model | Test Accuracy |
| :--- | :--- | :---: |
| **Bag-of-Words (BoW)** | **Multinomial Naive Bayes** | **98.65%** |
| TF-IDF (n-grams 1,2) | Multinomial Naive Bayes | 98.12% |
| TF-IDF (n-grams 1,2) | Random Forest | 98.39% |
| Word2Vec (Avg) | Random Forest | 95.78% |

## 🛠️ Technologies Used

-   **Python 3.10**
-   **Pandas:** For data loading and manipulation.
-   **NLTK (Natural Language Toolkit):** For core NLP tasks like tokenization, stopwords, and stemming/lemmatization.
-   **Scikit-learn:** For feature extraction (CountVectorizer, TfidfVectorizer), model training (MultinomialNB, RandomForestClassifier), and evaluation (accuracy_score, classification_report, roc_curve, auc).
-   **Gensim:** For training the Word2Vec model.
-   **WordCloud:** For generating visualization.
-   **Matplotlib & Seaborn:** For plotting.
-   **Jupyter Notebook:** For interactive development.

## 📦 Installation

1.  Clone this repository:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/sms-spam-classifier.git](https://github.com/YOUR_USERNAME/sms-spam-classifier.git)
    cd sms-spam-classifier
    ```
2.  Create and activate a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  Install the required libraries:
    ```bash
    pip install pandas nltk scikit-learn gensim wordcloud matplotlib seaborn jupyter
    ```
4.  Run the notebook to download NLTK data (or run the `nltk.download()` cells within the notebook).

## ⚡ How to Use

1.  Ensure all dependencies are installed (see [Installation](#-installation)).
2.  Start the Jupyter Notebook server:
    ```bash
    jupyter notebook
    ```
3.  Open `Spam_NLP.ipynb` in your browser.
4.  Run the cells sequentially to load the data, preprocess it, and train the models.

## 🔬 Test with Your Own Message

You can easily test a custom message using the trained Word2Vec and Random Forest model (`clf`).

1.  Run all cells in the notebook up to **Cell 81** to train the model and define all necessary functions (like `avg_word2vec`).
2.  In a new cell, adapt the code from **Cell 82** to test your own message:

```python
import re
from gensim.utils import simple_preprocess

# --- Your message here ---
new_message = "Congratulations! You have won a free lottery. Click here to claim your prize."
# new_message = "Hey, are you free for dinner tonight?"

# Preprocess the message
review = re.sub('[^a-zA-Z0-9]', ' ', new_message)
review = review.lower()
tokens = simple_preprocess(review)
review = [lemmatizer.lemmatize(word) for word in tokens if not word in set(stopwords.words('english'))]

# Get average word2vec vector (ensuring correct shape)
email_vec = avg_word2vec(review).reshape(1, -1)

# Predict
prediction = clf.predict(email_vec)
print("Prediction:", "Spam" if prediction[0] == 1 else "Ham")
