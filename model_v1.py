import pandas as pd
import pickle
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

colnames=['query','label']
df = pd.read_csv('FoodBot Intents - Examples.csv', names=colnames)
# print(df.columns)

book_table = df[df['label'] == 'Book Table']
checkout = df[df['label'] == 'Checkout']
complaint = df[df['label'] == 'Complaint']
compliment = df[df['label'] == 'Compliment']
connect = df[df['label'] == 'Connect']
cost = df[df['label'] == 'Cost']
delivery = df[df['label'] == 'Delivery']
food_info = df[df['label'] == 'Food Info']
location = df[df['label'] == 'Location']
modify = df[df['label'] == 'Modify']
nutrition = df[df['label'] == 'Nutrition']
order = df[df['label'] == 'Order']
payment = df[df['label'] == 'Payment']
preparation_time = df[df['label'] == 'Preparation Time']
promotions = df[df['label'] == 'Promotions']
restaurant_info = df[df['label'] == 'Restaurant Info']
show_menu = df[df['label'] == 'Show Menu']
status = df[df['label'] == 'Status']
suggest = df[df['label'] == 'Suggest']
suggest_drink = df[df['label'] == 'Suggest Drink']
suggest_food = df[df['label'] == 'Suggest Food']


# df['data']= df['data'].apply(lambda x: x.translate(None, string.punctuation))
# df['data']= df['data'].apply(lambda x: x.translate(None, string.digits))

X = order['query']
y = order['label']

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

with open('oneclass.pickle', 'wb') as fo:
    pickle.dump(pl,fo)
