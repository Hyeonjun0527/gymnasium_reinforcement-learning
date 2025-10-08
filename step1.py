import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

CUSTOM_MAP = [
    "SFFFFFFH",
    "FFHFFFFF",
    "FFFHFFFF",
    "FFFFFHFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
]
CUSTOM_MAP = np.array(CUSTOM_MAP, dtype='c')

env = gym.make(
    id="FrozenLake-v1",
    render_mode="rgb_array", # or "rgb_array"
    desc=CUSTOM_MAP,
    is_slippery=False
)

obs, info = env.reset()

# 0: 좌, 1: 하, 2: 우, 3: 상
obs, reward, terminated, truncated, info = env.step(2)

print(f'# Observation: {obs}')
print(f'# Reward: {reward}')
print(f'# Terminated: {terminated}')
print(f'# Truncated: {truncated}')
print(f'# Info: {info}')

image = env.render()

print(f"Image Type: {type(image)}")
print(f"Image Shape: {image.shape}")

plt.imshow(image)
plt.axis('off')
plt.show()