import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from env.agro_env import AgroEnv

def train():
    # Dynamic Paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(base_dir, 'models')
    logs_dir = os.path.join(base_dir, 'results', 'logs')
    model_path = os.path.join(models_dir, 'ppo_agro_final')
    
    # Create directories
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Initialize environment
    env = make_vec_env(lambda: AgroEnv(), n_envs=4)

    # Initialize PPO model
    # Hyperparameters tuned for stability in agricultural simulation
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=logs_dir
    )

    # Callbacks
    eval_env = AgroEnv()
    eval_callback = EvalCallback(eval_env, best_model_save_path=models_dir,
                             log_path=logs_dir, eval_freq=5000,
                             deterministic=True, render=False)

    # Train
    print("Starting training...")
    model.learn(total_timesteps=100000, callback=eval_callback)

    # Save final model
    model.save(model_path)
    print("Training completed and model saved.")

if __name__ == "__main__":
    train()
