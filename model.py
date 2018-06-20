import pickle
from create_dataset import df
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn import metrics

X= df['data']
y = df['labels']

# X_train,X_test,y_train,y_test = train_test_split(X,pd.get_dummies(y),random_state=2)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=2)

pl = Pipeline([
    ('vectorizer',TfidfVectorizer()),
    ('clf',svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.5))
])

pl.fit(X_train,y_train)

preds = pl.predict(X_train)
targs = y_train
print(" train accuracy: ", metrics.accuracy_score(targs, preds))
# preds = pl.predict(X_test)
# targs = y_test
# print("accuracy: ", metrics.accuracy_score(targs, preds))

with open('oneclass.pickle', 'wb') as fo:
    pickle.dump(pl,fo)
