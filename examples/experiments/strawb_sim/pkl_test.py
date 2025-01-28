import pickle

# read the pickle file
with open('/home/emlyn/rl_franka/hil-serl/examples/classifier_data/strawb_real_100_success_images_2025-01-28_09-20-02.pkl', 'rb') as f:
    data = pickle.load(f)


for key, value in data[0].items():
    print(key)

print(f"obs keys: {data[0]['observations'].keys()}")