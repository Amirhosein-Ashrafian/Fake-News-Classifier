# Fake News Classifier with NLP

This project is a simple fake news detection system built using Python, Natural Language Processing (NLP), and machine learning. The goal is to classify news articles as either real or fake based on their textual content.

I worked on this as part of my learning process in data science and NLP. It helped me understand how to clean text, extract features using TF-IDF and CountVectorizer, and train a classifier to make predictions.

## Dataset

The dataset used for this project is `train.csv`, which contains news headlines and labels indicating whether each news item is real or fake. It’s a basic dataset, but useful for learning purposes.

- Columns:
  - `id`: Unique ID for each news sample
  - `title`, `author`, `text`: Textual features
  - `label`: Target value (0 = Real, 1 = Fake)

## Features

- Text cleaning and preprocessing (punctuation removal, lowercasing, stopword removal, lemmatization)
- Feature extraction with CountVectorizer and TfidfVectorizer
- Logistic Regression model for binary classification
- Custom text preprocessing module (`nlp_utils.py`) for better code organization

## Libraries Used

- Python 3.13
- pandas
- sklearn (scikit-learn)
- nltk
- matplotlib
- seaborn
- tqdm

### NLTK Resources Required

Make sure you have the following NLTK resources downloaded:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


### Results

After training the model on the dataset, it was able to predict fake and real news with a good level of accuracy (around 90%). I also used a confusion matrix and a classification report to check how well it performed on both classes.

Although this is not a very complex model, I was able to see clearly how text preprocessing (like cleaning, lemmatization, and using TF-IDF) can help improve the final results. It was interesting to see how small improvements in the text cleaning steps affected the accuracy.

### How to Run

If you want to try this project yourself:

1. Make sure all three files — the Jupyter notebook (`.ipynb`), the text processing module (`nlp_utils.py`), and the dataset (`train.csv`) — are in the same folder.  
2. Open the notebook in Jupyter or any similar environment.  
3. Run the cells from top to bottom one by one. You may need to download NLTK resources when prompted.  
4. The results and evaluation are shown at the end of the notebook.

### Notes

This was one of my early projects when I started learning about machine learning and NLP. I wanted to keep it simple but meaningful. I wrote the text processing code separately to make it more reusable and to practice organizing code better.

There are still things that can be improved in this project — like testing with other models, doing hyperparameter tuning, or using more advanced NLP methods — but I think this version gives a good overview of how fake news classification works and how it can be implemented in Python.



---

> Developed by Amirhosein Ashrafiyan | July 2025
