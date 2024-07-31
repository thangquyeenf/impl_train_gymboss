import GymBoss
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


# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize wandb
wandb.login()
wandb.init(project="ipl_algorithm")

gamma = 0.1
batch_size = 256

def V_pi(Q, state, gamma):
    with torch.no_grad():
        state = state.to(device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)  # Reshape to 2D tensor
        q_values = Q(state)
        v_pi = torch.max(q_values, dim=1)[0]
    return v_pi

def preference_probability(Q, segment1, segment2, gamma):
    def sum_T_pi_Q(segment):
        T_pi_Q = lambda s, a, s_prime: Q(s.to(device)).gather(1, a.view(-1, 1).to(device)).squeeze() - gamma * V_pi(Q, s_prime, gamma)
        return sum([T_pi_Q(s, a, s_prime) for s, a, s_prime in segment])

    exp1 = torch.exp(sum_T_pi_Q(segment1))
    exp2 = torch.exp(sum_T_pi_Q(segment2))
    return exp1 / (exp1 + exp2)

def regularized_preference_loss(Q, preference_data, lambda_psi=0.1, gamma=0.99):
    loss = 0
    for segment1, segment2, y in preference_data:
        p = preference_probability(Q, segment1, segment2, gamma)
        loss += y * torch.log(p) + (1 - y) * torch.log(1 - p)
    loss = -loss / len(preference_data)

    # Regularization term on Q network parameters
    reg_term = lambda_psi * sum(torch.sum(param ** 2) for param in Q.parameters())
    
    return loss + reg_term

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
    for _ in range(num_samples):
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
            observations_1.append(torch.tensor(obs_1))
            actions_1.append(torch.tensor(action, dtype=torch.int64))
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
            observations_2.append(torch.tensor(obs_2))
            actions_2.append(torch.tensor(action, dtype=torch.int64))
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

def IPL_algorithm(env, model, preference_data, offline_data, total_timesteps=10000, lambda_psi=0.1, alpha=0.2, gamma=0.99):
    model.q_net.to(device)
    optimizer_Q = torch.optim.Adam(model.q_net.parameters())
    optimizer_V = torch.optim.Adam(model.q_net.parameters())  # Replace with actual V parameters if separate
    
    log_data = []

    eval_callback = EvalCallback(env, best_model_save_path='./logs/',
                                 log_path='./logs/', eval_freq=1000,
                                 deterministic=True, render=False)
    eval_callback.init_callback(model)

    for timestep in range(total_timesteps):
        print("Current timestep: ", timestep)
        # Sample batches
        Bp = random.sample(preference_data, batch_size)
        Bo = random.sample(offline_data, batch_size)

        # Update Q
        Q_loss = regularized_preference_loss(model.q_net, Bp, lambda_psi, gamma)
        optimizer_Q.zero_grad()
        Q_loss.backward()
        optimizer_Q.step()

        # Update V
        V_loss = 0
        for (s, a, s_prime) in Bo:
            V_s = V_pi(model.q_net, s_prime, gamma)
            if len(s.shape) == 1:
                s = s.unsqueeze(0)  # Reshape to 2D tensor
            Q_sa = model.q_net(s).gather(1, a.view(-1, 1)).squeeze()
            V_loss += torch.mean((Q_sa - V_s) ** 2)
        V_loss /= len(Bo)
        optimizer_V.zero_grad()
        V_loss.backward()
        optimizer_V.step()

        # Update π
        pi_loss = 0
        for (s, a, s_prime) in Bo:
            if len(s.shape) == 1:
                s = s.unsqueeze(0)  # Reshape to 2D tensor
            Q_sa = model.q_net(s).gather(1, a.view(-1, 1)).squeeze()
            V_s = V_pi(model.q_net, s_prime, gamma)
            pi_loss += torch.mean(torch.exp((Q_sa - V_s) / alpha) * torch.log(model.policy(s)))
        pi_loss /= len(Bo)
        model.policy.optimizer.zero_grad()
        pi_loss.backward()
        model.policy.optimizer.step()

        # Log losses
        log_data.append({'timestep': timestep, 'Q_loss': Q_loss.item(), 'V_loss': V_loss.item(), 'pi_loss': pi_loss.item()})
        wandb.log({'timestep': timestep, 'Q_loss': Q_loss.item(), 'V_loss': V_loss.item(), 'pi_loss': pi_loss.item()})

        # Evaluate every 1000 timesteps
        if timestep % 1000 == 0:
            eval_results = evaluate_model(env, model, num_episodes=100)
            log_data[-1].update(eval_results)
        wandb.log({'timestep': timestep, 'Q_loss': Q_loss.item(), 'V_loss': V_loss.item(), 'pi_loss': pi_loss.item()})


        # Evaluate and save the best model using callback
        eval_callback.on_step()
    
    # Write log data to CSV
    with open('training_log.csv', 'w', newline='') as csvfile:
        fieldnames = ['timestep', 'Q_loss', 'V_loss', 'pi_loss', 'Average Reward', 'Average Length', 'Success Rate']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for data in log_data:
            writer.writerow(data)

def evaluate_model(env, model, num_episodes=100):
    model.q_net.to(device)
    all_rewards = []
    all_lengths = []
    all_successes = []

    for episode in tqdm(range(num_episodes)):
        print("evalute: ", episode)
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

# Ensure the environment is registered
env = gym.make('GymBoss/BossGame-v0')

# Initialize the SAC model
model = DQN('MlpPolicy', env, verbose=1, gamma=0.99)

expert = DQN('MlpPolicy', env, verbose=1)
policy_path = "./dqn_logs/best_model_policy.pt"
optimizer_path = "./dqn_logs/best_model_optimizer.pt"
expert.policy.load_state_dict(torch.load(policy_path))
expert.policy.optimizer.load_state_dict(torch.load(optimizer_path))
expert.policy.to(device)
# Get preference and offline data
preference_data = get_preference_data(env, 500, expert)

def gen_off_data(env, expert):
    off_data = []
    obs, _ = env.reset()
    for _ in range(500):
        if expert is not None:
            action, _ = expert.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
        next_observation, _, _, _, _ = env.step(action)
        off_data.append((torch.tensor(obs, dtype=torch.float32), 
                         torch.tensor(action, dtype=torch.int64),
                         torch.tensor(next_observation, dtype=torch.float32)))
        obs = next_observation
    return off_data


offline_data = gen_off_data(env, expert=expert)  # Dummy offline data

# Execute the IPL algorithm
IPL_algorithm(env, model, preference_data, offline_data, total_timesteps=10, lambda_psi=0.1, alpha=0.2, gamma=0.1)

# Evaluate the model
eval = evaluate_model(env, model, num_episodes=100)
print(eval)
