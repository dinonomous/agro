import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from env.agro_env import AgroEnv
import os

def run_fixed_rotation(env, episodes=10):
    rotation = [1, 0, 2, 3] # Rice, Wheat, Maize, Soybean
    results = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        total_reward = 0
        while not done:
            action = rotation[step % len(rotation)]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            results.append({
                "Strategy": "Fixed Rotation",
                "Step": step,
                "Reward": reward,
                "Yield": info["yield"],
                "SoilHealth": info["soil_health"],
                "Action": action
            })
            step += 1
            done = terminated or truncated
    return results

def run_greedy_yield(env, episodes=10):
    results = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        while not done:
            # Simulate yield for all actions and pick best
            best_yield = -1
            best_action = 0
            # Note: In real scenarios, this needs a model of the env. 
            # Here it acts as a "perfect" greedy oracle for demonstration.
            for a in range(5):
                 # Temporarily copy env state or use simplified logic
                 # This is a simplification
                 req = env.crop_requirements[a]
                 n, p, k = obs[0], obs[1], obs[2]
                 nutrient_ratio = min(n/req[0], p/req[1], k/req[2])
                 if nutrient_ratio > best_yield:
                     best_yield = nutrient_ratio
                     best_action = a
            
            obs, reward, terminated, truncated, info = env.step(best_action)
            results.append({
                "Strategy": "Greedy Yield",
                "Step": step,
                "Reward": reward,
                "Yield": info["yield"],
                "SoilHealth": info["soil_health"],
                "Action": best_action
            })
            step += 1
            done = terminated or truncated
    return results

def run_ppo(env, model_path, episodes=10):
    model = PPO.load(model_path)
    results = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            results.append({
                "Strategy": "PPO (Optimized)",
                "Step": step,
                "Reward": reward,
                "Yield": info["yield"],
                "SoilHealth": info["soil_health"],
                "Action": int(action)
            })
            step += 1
            done = terminated or truncated
    return results

def evaluate_and_plot():
    env = AgroEnv()
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, 'models', 'ppo_agro_final')
    plots_dir = os.path.join(base_dir, 'results', 'plots')
    
    if not os.path.exists(model_path + ".zip"):
        print("PPO model not found. Please run train.py first.")
        return

    print("Evaluating strategies...")
    results_fixed = run_fixed_rotation(env, episodes=20)
    results_greedy = run_greedy_yield(env, episodes=20)
    results_ppo = run_ppo(env, model_path, episodes=20)
    
    df = pd.DataFrame(results_fixed + results_greedy + results_ppo)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot 1: Cumulative Reward Comparison
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Step", y="Reward", hue="Strategy")
    plt.title("Reward per Season Comparison")
    plt.savefig(os.path.join(plots_dir, "reward_comparison.png"))
    
    # Plot 2: Soil Health Trends
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Step", y="SoilHealth", hue="Strategy")
    plt.title("Soil Health Sustainability Trends")
    plt.savefig(os.path.join(plots_dir, "soil_health_trends.png"))
    
    # Plot 3: Yield Comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Strategy", y="Yield")
    plt.title("Average Yield Performance")
    plt.savefig(os.path.join(plots_dir, "yield_comparison.png"))

    # Plot 4: Policy Behavior (Action Distribution)
    plt.figure(figsize=(10, 6))
    ppo_df = df[df["Strategy"] == "PPO (Optimized)"]
    action_counts = ppo_df["Action"].value_counts().sort_index()
    crop_names = [env.crop_names[i] for i in action_counts.index]
    plt.pie(action_counts, labels=crop_names, autopct='%1.1f%%')
    plt.title("PPO Crop Selection Distribution")
    plt.savefig(os.path.join(plots_dir, "policy_distribution.png"))

    print(f"Plots generated in {plots_dir}")
    
    # Summary Table
    summary = df.groupby("Strategy")[["Reward", "Yield", "SoilHealth"]].mean()
    print("\nPerformance Summary:")
    print(summary)
    summary.to_csv(os.path.join(base_dir, "results", "summary_metrics.csv"))

if __name__ == "__main__":
    evaluate_and_plot()
