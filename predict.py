import pickle

data = """Around Rs 25,000 crore refund is pending for the exporters while more than 3 lakhs applications seeking refunds have piled up with the central government, said West Bengal Finance Minister and former chairman of the empowered committee on GST, Amit Mitra.

Adding that he would raise the issue in the next GST council meeting, Mitra said, the exporters are suffering hugely for this.

The exporters are not getting refund and that is why they are losing out on their working capital which adversely aff.

"""

with open('oneclass.pickle', 'rb') as fi:
    model = pickle.load(fi)

with open('log.pickle', 'rb') as f:
    model2 = pickle.load(f)

# foldername = './dataset'
# filepath = os.path.join(os.getcwd(), foldername)
# for file in os.listdir(filepath):
#     with open(os.path.join(filepath, file), 'rb') as f:
#         data = [f.read()]
#     label = model.predict(data)
#     label2 = model2.predict(data)
#     print (file, label[0])
#     print (file, label2[0])

label = model.predict(data)
label2 = model2.predict(data)
print (file, label[0])
print (file, label2[0])