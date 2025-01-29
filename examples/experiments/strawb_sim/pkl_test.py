import pickle

# read the pickle file
with open('/home/emlyn/rl_franka/hil-serl/examples/classifier_data/strawb_real_100_success_images_2025-01-29_11-26-09.pkl', 'rb') as f:
    data = pickle.load(f)


for key, value in data[0].items():
    print(key)

print(f"obs keys: {data[0]['observations'].keys()}")
print(f"next obs keys: {data[0]['next_observations'].keys()}")
print(f"action shape: {data[0]['actions'].shape}")
print(f"reward: {data[0]['rewards']}")
print(f"mask: {data[0]['masks']}")
print(f"dones: {data[0]['dones']}")