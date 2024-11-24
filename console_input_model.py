'''
Data for model was taken from Kaggle repository:
https://www.kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification/overview
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import nltk
import string

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text
data = pd.read_csv('jigsaw-toxic-comment-train.csv')

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

data['comment_text'] = data['comment_text'].apply(preprocess_text)

X = data['comment_text']
y = data['obscene']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = make_pipeline(CountVectorizer(), MultinomialNB())

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f'Accuracy: {accuracy:.2f}')

def predict_offensiveness(input_text):
    processed_text = preprocess_text(input_text)
    prediction = model.predict([processed_text])
    return "Offensive" if prediction[0] == 1 else "Not Offensive"

while True:
    user_input = input("Enter a comment: ")
    result = predict_offensiveness(user_input)
    print(result)