import re
import pandas as pd
import pickle
import string
from create_dataset import df
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn import preprocessing


# Pre-Processing : remove punctations and digits , stop words and lower case to be done in Tfidf
df['data']= df['data'].apply(lambda x: x.translate(None, string.punctuation))
df['data']= df['data'].apply(lambda x: x.translate(None, string.digits))
# df['data'] = df['data'].str.lower()
X= df['data']
y = df['labels']

# test_sample = df['data'].head(2)
# for w in test_sample:
#     print(w)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=2)


pl = Pipeline([
    ('vectorizer',TfidfVectorizer(stop_words='english',ngram_range=(1,3),min_df=3,max_df=100,max_features=None)),
    ('clf',svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1))
])

pl.fit(X_train,y_train)

preds = pl.predict(X_train)
print(" train accuracy: ", accuracy_score(y_train, preds))
preds_test = pl.predict(X_test)
print(" test accuracy: ", accuracy_score(y_test, preds_test))


pipe = Pipeline([
    ('vectorizer',TfidfVectorizer(stop_words='english',ngram_range=(1,3),min_df=3,max_df=100,max_features=None)),
    ('clf',LogisticRegression())
])

pipe.fit(X_train,y_train)

p = pipe.predict(X_train)
print(" train accuracy: ", accuracy_score(y_train, p))
p_t = pipe.predict(X_test)
print(" test accuracy: ", accuracy_score(y_test, p_t))


with open('oneclass.pickle', 'wb') as fo:
    pickle.dump(pl,fo)

with open('log.pickle', 'wb') as fo:
    pickle.dump(pl,fo)
