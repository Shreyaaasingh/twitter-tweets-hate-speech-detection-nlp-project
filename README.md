# twitter-tweets-hate-speech-detection-nlp-project
An NLP project for detecting hate speech in tweets using machine learning models and TF-IDF features.


## Project Overview

This project implements a machine learning pipeline for detecting hate speech in tweets. The primary goal is to classify tweets as either containing hate speech (racist or sexist sentiment) or not. This is a critical task given the prevalence of harmful content online and the need for automated moderation systems.

The project addresses several key challenges inherent in NLP classification tasks, particularly dealing with imbalanced datasets, where hate speech (the minority class) is significantly less frequent than non-hate speech.

## Table of Contents

1.  [Problem Statement and Objective](#problem-statement-and-objective)
2.  [Dataset](#dataset)
3.  [Methodology](#methodology)
    *   [Data Loading and Initial Exploration](#data-loading-and-initial-exploration)
    *   [Data Preprocessing](#data-preprocessing)
    *   [Feature Extraction (TF-IDF)](#feature-extraction-tf-idf)
    *   [Model Training and Evaluation](#model-training-and-evaluation)
    *   [Hyperparameter Tuning](#hyperparameter-tuning)
4.  [Results and Conclusion](#results-and-conclusion)
5.  [Setup and Installation](#setup-and-installation)
6.  [Usage](#usage)
7.  [Future Work](#future-work)
8.  [License](#license)

## 1. Problem Statement and Objective

The objective is to build a classification model that can accurately identify hate speech in tweets. For this project, hate speech is defined as racist or sexist sentiment. Given a tweet, the model should predict whether it belongs to the 'hate speech' class (label 1) or 'not hate speech' class (label 0).

The main challenge is the highly imbalanced nature of the dataset, where hate speech examples are a small minority. The project focuses on developing models that perform well on the minority class, typically prioritizing recall and F1-score for hate speech detection.

## 2. Dataset

The dataset used for this project consists of `31,962` labeled tweets. Each entry includes a unique tweet ID, a label (0 for non-hate speech, 1 for hate speech), and the tweet text itself.

*   **Source:** The dataset is provided in this repository along with other files
* 
*   **File Name:** `train_E6oV3lV.csv`
*   **Content:** Contains tweet text and a binary label indicating hate speech presence.

## 3. Methodology

The project follows a standard machine learning pipeline for text classification:

### Data Loading and Initial Exploration

*   **Initial Data Loading:** Loaded the `train_E6oV3lV.csv` into a Pandas DataFrame.
*   **Class Distribution Analysis:** Performed initial EDA to understand the distribution of labels, revealing a significant class imbalance (approx. 7% hate speech).
*   **Tweet Length Analysis:** Examined the character length distribution of tweets for both classes.
*   **N-gram Analysis:** Investigated common bigrams in both hate speech and non-hate speech to identify frequent phrases.
*   **Word Cloud Generation:** Visualized the most frequent words across the entire corpus.
*   **Most Frequent Words by Class:** Analyzed and visualized the top 10 most frequent words in hate speech and non-hate speech tweets to highlight distinguishing vocabulary.
*   **Hashtag Analysis by Class:** Extracted and visualized the top 10 most frequent hashtags associated with each class, revealing thematic differences.

### Data Preprocessing

Text preprocessing is crucial for NLP tasks. The following steps were applied:

1.  **User Mention Removal:** `@user` mentions were removed from tweets.
2.  **Special Character Removal:** Non-alphabetic characters (except `#`) were removed.
3.  **Lowercasing:** All text was converted to lowercase.
4.  **Tokenization:** Tweets were split into individual words.
5.  **Stop Word Removal:** Common English stop words (e.g., 'the', 'is', 'a') were removed using `nltk`.
6.  **Stemming:** Words were reduced to their root form using `PorterStemmer` from `nltk`.

### Feature Extraction (TF-IDF)

*   **TF-IDF Vectorization:** The processed `tweet_text` was converted into numerical feature vectors using `TfidfVectorizer` from `sklearn`.
*   **Parameters:** `max_df=0.90`, `min_df=2`, `max_features=1000`, `stop_words='english'` were used to select the most relevant 1,000 features.

### Model Training and Evaluation

The dataset was split into 70% training and 30% testing sets. Several classification models were trained and evaluated:

*   **Logistic Regression:** Initial model trained without and with `class_weight='balanced'`.
*   **RandomForestClassifier:** Trained with `class_weight='balanced'`.
*   **XGBoostClassifier:** Initial model trained with `scale_pos_weight` to handle imbalance.

Evaluation metrics focused on **Accuracy, Precision, Recall, and F1-Score**, with a particular emphasis on the F1-score for the minority 'hate speech' class due to the class imbalance. Confusion matrices were also generated for visual analysis.

### Hyperparameter Tuning

*   **GridSearchCV for XGBoost:** Hyperparameter tuning was performed on the XGBoostClassifier using `GridSearchCV` to optimize for the F1-score of the positive (hate speech) class. Parameters tuned included `n_estimators`, `learning_rate`, and `max_depth`.

## 4. Results and Conclusion

After training and evaluating multiple models, including hyperparameter tuning, the **Tuned XGBoostClassifier** emerged as the best-performing model for this hate speech detection task.

*   **Best Parameters (XGBoost Tuned):** `learning_rate=0.2`, `max_depth=7`, `n_estimators=200`.
*   **Key Performance Metrics (Hate Speech Class - Label 1):**
    *   **Precision:** 0.431
    *   **Recall:** 0.640
    *   **F1-Score:** 0.515 (Highest among all models tested)

The tuned XGBoost model achieved the highest F1-score for the minority 'hate speech' class, demonstrating an effective balance between identifying actual hate speech instances (good recall) and minimizing false alarms (acceptable precision). This makes it the most robust choice for deployment in this scenario.

## 5. Setup and Installation

To run this notebook locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YourUsername/YourRepoName.git
    cd YourRepoName
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *   *(You'll need to generate a `requirements.txt` file first. In your Colab notebook, run `!pip freeze > requirements.txt` and download it, or manually list the libraries used: `pandas`, `numpy`, `scikit-learn`, `nltk`, `seaborn`, `matplotlib`, `xgboost`, `wordcloud`.)*

4.  **Download NLTK Data:**
    In a Python environment or at the beginning of your notebook, run:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt') # Often needed for tokenization
    ```

## 6. Usage

1.  **Data Placement:** Ensure the `train_E6oV3lV.csv` dataset is placed in the expected path (e.g., `/content/drive/MyDrive/NLP/` if running in Google Colab, or a designated `data/` folder if running locally).
2.  **Run the Notebook:** Open the `[Your-Notebook-Name].ipynb` file (e.g., `Hate_Speech_Detection.ipynb`) in a Jupyter environment (like Jupyter Lab, Jupyter Notebook, or Google Colab) and execute the cells sequentially.

## 7. Future Work

*   **Advanced Resampling:** Implement and evaluate advanced resampling techniques like SMOTE or ADASYN to potentially further improve minority class performance.
*   **Deep Learning Models:** Explore transformer-based models (e.g., BERT, RoBERTa) which often achieve state-of-the-art results in NLP tasks.
*   **Word Embeddings:** Incorporate pre-trained word embeddings (e.g., Word2Vec, GloVe) to capture semantic relationships between words.
*   **Error Analysis:** Conduct a deeper error analysis on misclassified tweets to identify patterns and further improve the model.
*   **Explainability:** Use techniques like SHAP or LIME to understand why the model makes certain predictions.
*   **Deployment:** Develop a simple web application or API to deploy the best-performing model.

## 8. License

This project is licensed under the [Your Chosen License, e.g., MIT License] - see the [LICENSE](LICENSE) file for details.
