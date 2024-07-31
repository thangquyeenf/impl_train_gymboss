import GymBoss
import os
import gym
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from tqdm import tqdm
import random
import csv
import wandb
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

priority_list = [(3, 3), (2, 3), (1, 3), (0, 3), 
                 (0, 2), (1, 2), (2, 2), (3, 2),
                 (3, 1), (2, 1), (3, 0), (2, 0),
                 (1, 0), (1, 1), (0, 1), (0, 0)]

def get_priority(coord):
    return priority_list.index(coord)

def compare_coordinates(coord1, coord2) -> bool:
    priority1 = get_priority(coord1)
    priority2 = get_priority(coord2)
    return priority1 < priority2

def get_label(obs_1, obs_2, ep_reward_1, ep_reward_2, step_earn_1, step_earn_2):
    if ep_reward_1 > ep_reward_2:
        return 1.0
    elif ep_reward_1 < ep_reward_2:
        return 0.0
    else:
        coord_1 = (obs_1[0], obs_1[1])
        coord_2 = (obs_2[0], obs_2[1])

        if coord_1 == coord_2 and step_earn_1 < step_earn_2:
            return 1.0
        
        if coord_1 == coord_2 and step_earn_1 > step_earn_2:
            return 0.0

        if coord_1 == coord_2 and step_earn_1 == step_earn_2:
            return 0.5
        
        if compare_coordinates(coord1=coord_1, coord2=coord_2):
            return 1.0
        else: 
            return 0.0

def get_preference_data(env, num_samples=100, expert=None):
    preference_data = []
    print("use expert: ", expert is not None)
    for num in range(num_samples):
        if num%1000==0 :print(num )
        observations_1 = []
        actions_1 = []
        observations_2 = []
        actions_2 = []
        obs_1, _ = env.reset()
        ep_reward_1 = 0
        ep_reward_2 = 0
        step_earn_1 = 0
        step_earn_2 = 0
        for step in range(200):
            if expert is not None:
                action, _ = expert.predict(obs_1, deterministic=True)
            else:
                action = env.action_space.sample()
            next_observation, reward, done, _, _ = env.step(action)
            obs_1 = np.array(obs_1, dtype=np.float32)
            if len(obs_1.shape) == 1:
                obs_1 = np.expand_dims(obs_1, axis=0)
            observations_1.append(torch.tensor(obs_1).to(device))
            actions_1.append(torch.tensor(action, dtype=torch.int64).to(device))
            ep_reward_1 += reward
            obs_1 = next_observation
            if done: 
                obs_1, _ = env.reset()
                step_earn_1 = step

        obs_2, _ = env.reset()
        for step in range(200):
            if expert is not None:
                action, _ = expert.predict(obs_2, deterministic=True)
            else:
                action = env.action_space.sample()
            next_observation, reward, done, _, _ = env.step(action)
            obs_2 = np.array(obs_2, dtype=np.float32)
            if len(obs_2.shape) == 1:
                obs_2 = np.expand_dims(obs_2, axis=0)
            observations_2.append(torch.tensor(obs_2).to(device))
            actions_2.append(torch.tensor(action, dtype=torch.int64).to(device))
            ep_reward_2 += reward
            obs_2 = next_observation
            if done: 
                obs_2, _ = env.reset()
                step_earn_2 = step

        label = get_label(obs_1=obs_1, obs_2=obs_2,
                          ep_reward_1=ep_reward_1, ep_reward_2=ep_reward_2,
                          step_earn_1=step_earn_1, step_earn_2=step_earn_2)
        
        segment1 = list(zip(observations_1, actions_1, observations_1[1:]))
        segment2 = list(zip(observations_2, actions_2, observations_2[1:]))
        
        preference_data.append((segment1, segment2, label))
    return preference_data

env = gym.make('GymBoss/BossGame-v0')

# Initialize the DQN model
model = DQN('MlpPolicy', env, verbose=1, gamma=0.99)

expert = DQN('MlpPolicy', env, verbose=1)
policy_path = "./dqn_logs/best_model_policy.pt"
optimizer_path = "./dqn_logs/best_model_optimizer.pt"
expert.policy.load_state_dict(torch.load(policy_path))
expert.policy.optimizer.load_state_dict(torch.load(optimizer_path))
expert.policy.to(device)
# Get preference and offline data

def gen_off_data(env, expert, num_off_data):
    off_data = []
    obs, _ = env.reset()
    for num in range(num_off_data):
        if num%1000==0 :print(num )
        if expert is not None:
            action, _ = expert.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
        next_observation, _, _, _, _ = env.step(action)
        off_data.append((torch.tensor(obs, dtype=torch.float32).to(device), 
                         torch.tensor(action, dtype=torch.int64).to(device),
                         torch.tensor(next_observation, dtype=torch.float32).to(device)))
        obs = next_observation
    return off_data

preference_data = get_preference_data(env, 10000, expert)
offline_data = gen_off_data(env, expert=expert, num_off_data=10000)

# Chuyển đổi dữ liệu từ CUDA sang numpy và lưu lại
preference_data_np = []
for segment1, segment2, label in preference_data:
    segment1_np = [(obs.cpu().numpy(), action.cpu().numpy(), next_obs.cpu().numpy()) for obs, action, next_obs in segment1]
    segment2_np = [(obs.cpu().numpy(), action.cpu().numpy(), next_obs.cpu().numpy()) for obs, action, next_obs in segment2]
    preference_data_np.append((segment1_np, segment2_np, label))

offline_data_np = [(obs.cpu().numpy(), action.cpu().numpy(), next_obs.cpu().numpy()) for obs, action, next_obs in offline_data]

# Save to pickle files
os.makedirs('dataset', exist_ok=True)
with open('dataset/preference_data.pkl', 'wb') as f:
    pickle.dump(preference_data_np, f)

with open('dataset/offline_data.pkl', 'wb') as f:
    pickle.dump(offline_data_np, f)

# # To load the data
# with open('dataset/preference_data.pkl', 'rb') as f:
#     loaded_preference_data = pickle.load(f)

# with open('dataset/offline_data.pkl', 'rb') as f:
#     loaded_offline_data = pickle.load(f)

# print(loaded_offline_data)
