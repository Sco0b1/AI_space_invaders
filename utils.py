import os
import matplotlib.pyplot as plt
import numpy as np
from config import (
    MODEL_SAVE_PATH, 
    LOG_PATH,
    EPISODES,
    BATCH_SIZE,
    LEARNING_RATE,
    GAMMA,
    USE_DOUBLE_DQN,
    USE_DUELING,
    USE_PRIORITIZED_REPLAY,
    USE_MULTI_STEP,
    N_STEP,
    USE_DISTRIBUTIONAL,
    N_ATOMS,
    USE_NOISY_NETS
)


def create_directories():
    """Create directories for models and logs"""
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)
    print(f"Created directories: {MODEL_SAVE_PATH}, {LOG_PATH}")


def plot_training_results(episodes, episode_rewards, losses, episode_lengths, save_path=None, best_ep=None):
    """Plot training metrics"""
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=False)
    axes[1].sharex(axes[0])
    
    #Plot episode rewards
    axes[0].plot(episode_rewards, alpha=0.3, label='Episode Reward', color='blue')
    warmup_period = min(100, episodes // 2)
    if len(episode_rewards) >= warmup_period:
        window = min(100, len(episode_rewards))
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        x_ma = range(window - 1, len(episode_rewards))
        axes[0].plot(
            x_ma,
            moving_avg,
            label=f'Moving Average ({window} episodes)', 
            linewidth=2,
            color='red'
        )
        best_so_far = np.maximum.accumulate(moving_avg)
        axes[0].plot(
            x_ma,
            best_so_far,
            label='Best Moving Average So Far',
            linewidth=2,
            linestyle='--',
            color='orange'
        )

    if 'best_ep' in locals() or 'best_ep' in globals():
        try:
            axes[0].axvline(
                best_ep,
                label=f'Best Model (episode {best_ep})',
                color='gold',
                linestyle='--',
                linewidth=1.8,
            )
        except Exception:
            pass

    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Episode Rewards Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    #Plot episode lengths
    axes[1].plot(episode_lengths, alpha=0.6, label='Episode Length', color='purple')
    if len(episode_lengths) >= 2:
        len_window = min(100, len(episode_lengths))
        moving_len = np.convolve(episode_lengths, np.ones(len_window)/len_window, mode='valid')
        x_ma = range(len_window - 1, len(episode_lengths))
        axes[1].plot(
            x_ma,
            moving_len, 
            label=f'Moving Average ({len_window} episodes)',
            linewidth=2,
            color='black'
            )
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Length (steps)')
    axes[1].set_title('Episode Lengths Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    #Plot training loss
    if losses:
        axes[2].plot(losses, alpha=0.6, label='Loss', color='green')
        if len(losses) >= 100:
            moving_avg = np.convolve(losses, np.ones(100)/100, mode='valid')
            axes[2].plot(range(99, len(losses)), moving_avg, 
                        label='Moving Average (100 steps)', linewidth=2, color='orange')
        axes[2].set_xlabel('Training Step')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('Training Loss Over Time')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")


def print_training_progress(episode, episodes, avg_reward, avg_loss, epsilon, steps, use_rainbow=False):
    """
    Print formatted training progress
    
    Args:
        episode: Current episode number
        episodes: Total episodes
        avg_reward: Average reward
        avg_loss: Average loss
        epsilon: Current epsilon value (ignored if use_rainbow=True)
        steps: Total steps taken
        use_rainbow: Whether using Rainbow (noisy nets instead of epsilon)
    """
    if use_rainbow:
        print(f"Episode {episode:4d}/{episodes} | "
              f"Steps: {steps:7d} | "
              f"Avg Reward: {avg_reward:7.2f} | "
              f"Avg Loss: {avg_loss:7.4f} | "
              f"Exploration: Noisy Nets")
    else:
        print(f"Episode {episode:4d}/{episodes} | "
              f"Steps: {steps:7d} | "
              f"Avg Reward: {avg_reward:7.2f} | "
              f"Avg Loss: {avg_loss:7.4f} | "
              f"Epsilon: {epsilon:.3f}")


def save_config_summary(filepath="logs/config_summary.txt"):
    """Save configuration summary to file"""
    with open(filepath, 'w') as f:
        f.write("=== Rainbow DQN Configuration ===\n\n")
        f.write(f"Episodes: {EPISODES}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Gamma: {GAMMA}\n\n")
        
        f.write("=== Rainbow Components ===\n")
        f.write(f"1. Double DQN: {USE_DOUBLE_DQN}\n")
        f.write(f"2. Dueling Networks: {USE_DUELING}\n")
        f.write(f"3. Prioritized Replay: {USE_PRIORITIZED_REPLAY}\n")
        f.write(f"4. Multi-step Learning: {USE_MULTI_STEP} (n={N_STEP})\n")
        f.write(f"5. Distributional RL: {USE_DISTRIBUTIONAL} (atoms={N_ATOMS})\n")
        f.write(f"6. Noisy Networks: {USE_NOISY_NETS}\n")
    
    print(f"Configuration saved to {filepath}")