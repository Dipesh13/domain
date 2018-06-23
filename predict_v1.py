import pickle

data_set = ["i would like to book a table for 2 people at 2 am this Monday","I want to have a pizza"]

with open('oneclass.pickle', 'rb') as fi:
    model = pickle.load(fi)

for data in data_set:
    print(model.predict([data]))