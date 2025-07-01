import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle


data = pd.read_csv("data/spam.csv", encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']
data['label_num'] = data['label'].map({'ham': 0, 'spam': 1})


cv = CountVectorizer()
X = cv.fit_transform(data['message'])
y = data['label_num']


model = MultinomialNB()
model.fit(X, y)


pickle.dump(model, open("model/spam_model.pkl", "wb"))
pickle.dump(cv, open("model/vectorizer.pkl", "wb"))