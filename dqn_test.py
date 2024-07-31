import GymBoss
import gym
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
# Define the policy class to match the one used in training
# class CustomPolicy(DQN.policy_class):
#     def __init__(self, *args, **kwargs):
#         super(CustomPolicy, self).__init__(*args, **kwargs)

# Load the environment
env = gym.make('GymBoss/BossGame-v0')

# Load the saved model's state_dict
model = DQN('MlpPolicy', env, verbose=1)
policy_path = "./dqn_logs/best_model_policy.pt"
optimizer_path = "./dqn_logs/best_model_optimizer.pt"
model.policy.load_state_dict(torch.load(policy_path))
model.policy.optimizer.load_state_dict(torch.load(optimizer_path))
# Evaluate the loaded model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Detailed evaluation function
def evaluate_model(env, model, num_episodes=100):
    all_rewards = []
    all_lengths = []
    all_successes = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        length = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            length += 1

            if length > 2000:
                all_rewards.append(total_reward)
                all_lengths.append(length)
                all_successes.append(1 if reward == 100 else 0)
                break

            if done:
                all_rewards.append(total_reward)
                all_lengths.append(length)
                all_successes.append(1 if reward == 100 else 0)

    avg_reward = np.mean(all_rewards)
    avg_length = np.mean(all_lengths)
    success_rate = np.mean(all_successes)

    return {
        'Average Reward': avg_reward,
        'Average Length': avg_length,
        'Success Rate': success_rate
    }

# Perform detailed evaluation
evaluation_results = evaluate_model(env, model, num_episodes=100)
print(f"Evaluation Results: {evaluation_results}")
