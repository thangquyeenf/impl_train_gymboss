import GymBoss
import os
import gym
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import wandb
from tqdm import tqdm
import torch

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=10000, log_dir="./logs", verbose=1):
        super(CustomCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.log_dir = log_dir
        self.best_mean_reward = -np.inf
        self.logs = []
        self.train_losses = []
        self.train_rewards = []
        self.train_lengths = []

    def _init_callback(self):
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
        wandb.init(project="dqn-evaluation")

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            mean_reward, std_reward, eval_metrics = self.evaluate_model()
            train_metrics = self._log_training_metrics()

            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.save_model(f"{self.log_dir}/best_model")

            # Log metrics
            log_data = {
                "timesteps": self.num_timesteps,
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "eval_length": eval_metrics["avg_length"],
                "success_rate": eval_metrics["success_rate"],
                "train_loss": train_metrics["loss"],
                "train_reward": train_metrics["reward"],
                "train_length": train_metrics["length"]
            }
            self.logs.append(log_data)
            wandb.log(log_data)

            # Save logs to CSV
            self._save_logs_to_csv()

        return True

    def _log_training_metrics(self):
        # Calculate average training metrics
        loss = np.mean(self.train_losses) if self.train_losses else 0
        reward = np.mean(self.train_rewards) if self.train_rewards else 0
        length = np.mean(self.train_lengths) if self.train_lengths else 0

        # Clear the lists after logging
        self.train_losses.clear()
        self.train_rewards.clear()
        self.train_lengths.clear()

        return {
            "loss": loss,
            "reward": reward,
            "length": length
        }

    def evaluate_model(self):
        all_rewards = []
        all_lengths = []
        all_successes = []

        for _ in tqdm(range(100)):  # Evaluate over 100 episodes
            obs, _ = self.eval_env.reset()
            total_reward = 0
            done = False
            length = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, _ = self.eval_env.step(action)
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
        std_reward = np.std(all_rewards)
        avg_length = np.mean(all_lengths)
        success_rate = np.mean(all_successes)

        eval_metrics = {
            'avg_length': avg_length,
            'success_rate': success_rate
        }

        return avg_reward, std_reward, eval_metrics

    def _save_logs_to_csv(self):
        df = pd.DataFrame(self.logs)
        df.to_csv(f"{self.log_dir}/log.csv", index=False)
    
    def save_model(self, path):
        torch.save(self.model.policy.state_dict(), f"{path}_policy.pt")
        torch.save(self.model.policy.optimizer.state_dict(), f"{path}_optimizer.pt")

# Ensure the environment is registered
env = gym.make('GymBoss/BossGame-v0')
eval_env = gym.make('GymBoss/BossGame-v0')

# Create log directory
log_dir = "./dqn_logs"
log_dir_1 = "./logs"

# Create the callback
callback = CustomCallback(eval_env=eval_env, eval_freq=50000, log_dir=log_dir)

# Track and log losses during training
class CustomDQN(DQN):
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        for gradient_step in range(gradient_steps):
            # Perform a gradient step
            super().train(1, batch_size)

            # Assuming the replay buffer holds the required information
            # and loss can be computed/accessed from the model's internal state.
            # This is just a placeholder logic.
            if hasattr(self, 'logger'):
                for key, value in self.logger.name_to_value.items():
                    if 'loss' in key:
                        callback.train_losses.append(value)
                    if 'reward' in key:
                        callback.train_rewards.append(value)
                    if 'length' in key:
                        callback.train_lengths.append(value)

# Initialize the custom DQN model
model = CustomDQN('MlpPolicy', env, verbose=1, tensorboard_log=log_dir_1, gamma=0.99)

# Move model to CUDA
model.policy.to(device)

# Train the model
total_timesteps = 1000000
model.learn(total_timesteps=total_timesteps, callback=callback)

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Log final evaluation results to wandb
wandb.log({"final_mean_reward": mean_reward, "final_std_reward": std_reward})
wandb.finish()
