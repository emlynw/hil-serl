import pickle

# read the pickle file
with open('first_run/buffer/transitions_1000.pkl', 'rb') as f:
    data = pickle.load(f)


for key, value in data[0].items():
    print(key)

print(data[0]['observations']['wrist2'].shape)
