import pickle
import cv2

# read the pickle file
with open('/home/emlyn/rl_franka/hil-serl/examples/experiments/strawb_real/learned_gripper/demo_buffer/transitions_1000.pkl', 'rb') as f:
    data = pickle.load(f)


for key, value in data[0].items():
    print(key)

print(f"len data: {len(data)}")
print(f"obs keys: {data[0]['observations'].keys()}")
print(f"next obs keys: {data[0]['next_observations'].keys()}")
print(f"action shape: {data[0]['actions'].shape}")
print(f"reward: {data[0]['rewards']}")
print(f"mask: {data[0]['masks']}")
print(f"dones: {data[0]['dones']}")

print(f"obs shape: {data[0]['observations']['wrist1'].shape}")

for transition in data:
    wrist1_image = transition['observations']['wrist1'][0]
    wrist1_image_bgr = cv2.cvtColor(wrist1_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Wrist1 Image Obs', wrist1_image_bgr)
    wrist1_next_image = transition['next_observations']['wrist1'][0]
    wrist1_next_image_bgr = cv2.cvtColor(wrist1_next_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Wrist1 Image Next Obs', wrist1_next_image_bgr)
    cv2.waitKey(0)