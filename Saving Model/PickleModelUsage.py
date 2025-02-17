import pickle

with open('model', 'rb') as f:
    model = pickle.load(f)

output = model.predict([[3000, 3, 40]])
print(output)