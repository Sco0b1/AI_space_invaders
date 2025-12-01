import gymnasium as gym
import ale_py
import torch
import numpy as np
import os
import argparse
import math
import matplotlib.pyplot as plt
import imageio.v3 as iio
from datetime import datetime
from agent import Agent
from preprocessing import FramePreprocessor
from utils import create_directories, plot_training_results, print_training_progress, save_config_summary
from config import *


def run_deterministic_evaluation(env, agent, preprocessor, num_episodes=20):
    """Run deterministic evaluation without exploration noise"""
    eval_rewards = []
    
    for _ in range(num_episodes):
        observation, _ = env.reset()
        state = preprocessor.reset(observation)
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, training=False)
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = preprocessor.add_frame(observation)
            episode_reward += reward
        
        eval_rewards.append(episode_reward)
    
    return np.mean(eval_rewards), np.std(eval_rewards), eval_rewards


def train(episodes=EPISODES, use_rainbow=True, render=False, resume_path=None):
    """Train Rainbow or basic DQN agent"""
    create_directories()
    save_config_summary()
    
    #Setup environment
    gym.register_envs(ale_py)
    render_mode = "rgb_array" if render else None
    env = gym.make("ALE/SpaceInvaders-v5", 
                   render_mode=render_mode,
                   repeat_action_probability=0)
    
    #Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"Training Configuration")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Action space: {env.action_space.n} actions")
    print(f"Mode: {'Rainbow DQN' if use_rainbow else 'Basic DQN'}")
    if use_rainbow:
        print(f"  - Double DQN: {USE_DOUBLE_DQN}")
        print(f"  - Dueling Networks: {USE_DUELING}")
        print(f"  - Prioritized Replay: {USE_PRIORITIZED_REPLAY}")
        print(f"  - Multi-step (n={N_STEP}): {USE_MULTI_STEP}")
        print(f"  - Distributional C51: {USE_DISTRIBUTIONAL}")
        print(f"  - Noisy Networks: {USE_NOISY_NETS}")
    print(f"{'='*70}\n")
    
    #Initialize agent and preprocessor
    preprocessor = FramePreprocessor()
    agent = Agent(n_actions=env.action_space.n, device=device, use_rainbow=use_rainbow)
    
    #Load checkpoint if resuming
    start_episode = 0
    if resume_path:
        print(f"Loading checkpoint from: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        agent.load(resume_path)
        
        # Try to extract episode number from filename (e.g., rainbow_best.pth, rainbow_checkpoint_ep283.pth)
        import re
        match = re.search(r'ep(\d+)', resume_path)
        if match:
            start_episode = int(match.group(1)) + 1
            print(f"Resuming from episode {start_episode}")
        else:
            print("Checkpoint loaded, starting from episode 0 (episode counter not found in filename)")
        print(f"{'='*70}\n")
    
    #Training metrics
    episode_rewards = []
    episode_lengths = []
    losses = []
    total_steps = 0
    
    #Best model tracking
    best_avg_reward = -float('inf')
    best_episode = 0
    episodes_without_improvement = 0
    
    print(f"Starting training for {episodes} episodes...")
    print(f"{'='*70}\n")
    
    try:
        for episode in range(start_episode, start_episode + episodes):
            observation, _ = env.reset()
            state = preprocessor.reset(observation)
            
            if USE_NOISY_NETS and use_rainbow:
                agent.policy_net.reset_noise()
            
            episode_reward = 0
            episode_steps = 0
            episode_losses = []
            done = False
            
            while not done:
                if USE_NOISY_NETS and use_rainbow:
                    agent.policy_net.reset_noise()
                
                action = agent.select_action(state, training=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                #Reward clipping for training stability
                original_reward = reward
                clipped_reward = max(min(reward, 1), -1)
                
                next_state = preprocessor.add_frame(observation)
                agent.push_experience(state, action, clipped_reward, next_state, done)
                
                loss = agent.train_step()
                if loss is not None:
                    if math.isnan(loss) or math.isinf(loss) or loss > 100:
                        print(f"\nLoss explosion detected!")
                        print(f"   Episode: {episode}, Loss: {loss}")
                        print(f"   Saving emergency checkpoint...")
                        emergency_path = os.path.join(MODEL_SAVE_PATH, 
                                                     f"{'rainbow' if use_rainbow else 'dqn'}_emergency_ep{episode}.pth")
                        agent.save(emergency_path)
                        print(f"   Emergency model saved: {emergency_path}")
                        print(f"   Training stopped.")
                        env.close()
                        return
                    
                    episode_losses.append(loss)
                    losses.append(loss)
                
                state = next_state
                episode_reward += original_reward
                total_steps += 1
                episode_steps += 1

            episode_lengths.append(episode_steps)
            
            agent.decay_epsilon()
            episode_rewards.append(episode_reward)
            
            if episode % PRINT_FREQUENCY == 0:
                avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else episode_rewards[-1]
                avg_loss = np.mean(episode_losses) if episode_losses else 0
                
                if len(episode_rewards) >= 10:
                    print(f"Episode {episode:4d}/{episodes} | Steps: {total_steps:7d} | "
                          f"Reward: {episode_reward:6.2f} | Avg(10): {avg_reward:6.2f} | "
                          f"Loss: {avg_loss:7.4f} | Exploration: {'Noisy Nets' if use_rainbow else f'ε={agent.epsilon:.4f}'}")
                else:
                    print_training_progress(episode, episodes, avg_reward, avg_loss, 
                                           agent.epsilon, total_steps, use_rainbow=use_rainbow)
            
            #Save best model
            warmup_period = min(100, episodes // 2)
            if episode >= warmup_period:
                recent_avg = np.mean(episode_rewards[-100:])
                if recent_avg > best_avg_reward + 0.5:
                    improvement = recent_avg - best_avg_reward
                    best_avg_reward = recent_avg
                    best_episode = episode
                    episodes_without_improvement = 0
                    
                    best_path = os.path.join(MODEL_SAVE_PATH, 
                                            f"{'rainbow' if use_rainbow else 'dqn'}_best.pth")
                    agent.save(best_path)
                    if episode % PRINT_FREQUENCY == 0:
                        print(f"  New best model! Avg reward: {recent_avg:.2f} (last 100 ep, +{improvement:.2f})")
                else:
                    episodes_without_improvement += 1

            #Informational monitoring every 1000 episodes
            if episode > 0 and episode % 1000 == 0:
                recent_500 = np.mean(episode_rewards[-500:]) if len(episode_rewards) >= 500 else np.mean(episode_rewards)
                recent_loss = np.mean(losses[-500:]) if len(losses) >= 500 else np.mean(losses) if losses else 0
                
                print(f"\n{'='*70}")
                print(f"Progress Checkpoint (Episode {episode}/{episodes})")
                print(f"{'='*70}")
                print(f"   Last 500 episodes avg reward: {recent_500:.2f}")
                print(f"   Last 500 steps avg loss: {recent_loss:.4f}")
                print(f"   Best avg reward so far: {best_avg_reward:.2f} (ep {best_episode})")
                print(f"   Episodes without improvement: {episodes_without_improvement}")
                print(f"   Total steps taken: {total_steps:,}")
                print(f"{'='*70}\n")
            
            if episode % SAVE_FREQUENCY == 0 and episode > 0:
                if episode % 1000 == 0:
                    model_name = f"{'rainbow' if use_rainbow else 'dqn'}_checkpoint_ep{episode}.pth"
                    save_path = os.path.join(MODEL_SAVE_PATH, model_name)
                    agent.save(save_path)
                    print(f"  → Milestone saved: {save_path}")
                
                latest_name = f"{'rainbow' if use_rainbow else 'dqn'}_latest.pth"
                latest_path = os.path.join(MODEL_SAVE_PATH, latest_name)
                agent.save(latest_path)
                print(f"  → Latest checkpoint: {latest_path} (ep {episode})")

        print(f"\n{'='*70}")
        print("Training complete!")
        print(f"{'='*70}")
        print(f"Total episodes: {episodes}")
        print(f"Total steps: {total_steps:,}")
        print(f"Best avg reward: {best_avg_reward:.2f} (episode {best_episode})")
        print(f"{'='*70}\n")
        
        final_model_name = f"{'rainbow' if use_rainbow else 'dqn'}_final.pth"
        final_path = os.path.join(MODEL_SAVE_PATH, final_model_name)
        agent.save(final_path)
        print(f"Final model saved: {final_path}")
        model_name = f"{'rainbow' if use_rainbow else 'dqn'}_best.pth"
        print(f"Best model saved: {os.path.join(MODEL_SAVE_PATH, model_name)}")
        
        plot_path = os.path.join(LOG_PATH, "training_results.png")
        plot_training_results(episodes, episode_rewards, losses, episode_lengths, save_path=plot_path, best_ep=best_episode)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        interrupt_path = os.path.join(MODEL_SAVE_PATH, f"{'rainbow' if use_rainbow else 'dqn'}_interrupted_ep{episode}.pth")
        agent.save(interrupt_path)
        print(f"Model saved before exit: {interrupt_path}")
        if best_avg_reward > -float('inf'):
            print(f"Best model (avg reward {best_avg_reward:.2f}) was saved at episode {best_episode}")
        
        print("\nGenerating training results plot...")
        plot_path = os.path.join(LOG_PATH, "training_results.png")
        plot_training_results(episode_rewards, losses, save_path=plot_path)
        print(f"Training plot saved: {plot_path}")
    
    finally:
        env.close()

def save_video(frames, path, fps = 60):
    """Save frames as video using imageio."""
    if path.endswith(".gif"):
        iio.imwrite(path, frames, duration=1/fps, loop=0)
    else:
        iio.imwrite(path, frames, fps=fps, macro_block_size=1)


def evaluate(model_path, episodes=10, render=True, record=False, video_path=None, fps=60):
    """Evaluate trained agent"""
    gym.register_envs(ale_py)
    want_pixels = render or record
    env = gym.make("ALE/SpaceInvaders-v5", 
                   render_mode="rgb_array" if want_pixels else None,
                   repeat_action_probability=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #Determine if Rainbow or basic DQN from filename
    use_rainbow = 'rainbow' in model_path.lower()
    
    #Initialize agent
    preprocessor = FramePreprocessor()
    agent = Agent(n_actions=env.action_space.n, device=device, use_rainbow=use_rainbow)

    #Load model
    agent.load(model_path)
    print(f"Loaded model from: {model_path}")
    print(f"Mode: {'Rainbow DQN' if use_rainbow else 'Basic DQN'}")
    
    #Initialize pygame if rendering
    if render:
        import pygame
        pygame.init()
        WINDOW_WIDTH = 800
        WINDOW_HEIGHT = 900
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Space Invaders - Rainbow DQN Evaluation")
        clock = pygame.time.Clock()
        print("\nRendering with pygame window (800x900)")
    
    #Evaluation
    episode_rewards = []
    recorded_frames = []
    
    print(f"\nEvaluating for {episodes} episodes...")
    
    with torch.no_grad():
        for episode in range(episodes):
            observation, _ = env.reset()
            state = preprocessor.reset(observation)
            
            episode_reward = 0
            done = False
            
            while not done:
                #Handle pygame events if rendering
                if render:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            print("\nWindow closed by user")
                            done = True
                            break
                    
                    if done:
                        break
                
                #Select action (no exploration)
                action = agent.select_action(state, training=False)
                
                #Take step
                observation, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                #Display if rendering
                if want_pixels:
                    frame = env.render()

                    if record:
                        recorded_frames.append(frame)

                    if render:
                        frame = np.transpose(frame, (1, 0, 2))
                        surface = pygame.surfarray.make_surface(frame)
                        surface = pygame.transform.scale(surface, (WINDOW_WIDTH, WINDOW_HEIGHT))
                        screen.blit(surface, (0, 0))
                        pygame.display.flip()
                        clock.tick(60)

                #Preprocess
                state = preprocessor.add_frame(observation)
                episode_reward += reward
        
            episode_rewards.append(episode_reward)
            print(f"Episode {episode + 1}/{episodes} - Reward: {episode_reward}")
    
    #Save video if recording
    if record and recorded_frames:
        out = video_path or os.path.join(
            LOG_PATH, f"eval_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        )
        save_video(recorded_frames, out, fps=fps)

    #Cleanup
    if render:
        pygame.quit()
    env.close()
    
    #Statistics
    print(f"\n{'='*50}")
    print("Evaluation Results:")
    print(f"{'='*50}")
    print(f"Mean Reward: {np.mean(episode_rewards):.2f}")
    print(f"Std Reward: {np.std(episode_rewards):.2f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")

    p50 = np.percentile(episode_rewards, 50)
    p75 = np.percentile(episode_rewards, 75)
    p90 = np.percentile(episode_rewards, 90)

    print(f"50th Percentile: {p50:.2f}")
    print(f"75th Percentile: {p75:.2f}")
    print(f"90th Percentile: {p90:.2f}")
    
    print(f"{'='*50}")
    
    if record and recorded_frames:
        print(f"Recorded video saved to: {out}")

    #Plot
    fig, axes = plt.subplots(2,1, figsize=(10, 10), sharex=False, constrained_layout=True)
    axes[0].bar(range(len(episode_rewards)), episode_rewards)
    axes[0].axhline(y=np.mean(episode_rewards), color='r', linestyle='--', 
                label=f'Mean: {np.mean(episode_rewards):.2f}')
    axes[0].set_xticks(range(len(episode_rewards)))
    for i, r in enumerate(episode_rewards):
        axes[0].text(i, r + 5, f"{r:.0f}", ha='center', va='bottom', fontsize=8)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Evaluation Rewards per Episode')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(episode_rewards, bins=min(10, len(episode_rewards)), color='skyblue', edgecolor='black')
    axes[1].set_xlabel('Reward')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Histogram of Evaluation Rewards')
    axes[1].grid(True, alpha=0.3)

    summary_text = (
        f"Evaluation Summary:\n"
        f"Mean: {np.mean(episode_rewards):.2f} | "
        f"Std: {np.std(episode_rewards):.2f} | "
        f"Min: {np.min(episode_rewards):.2f} | "
        f"Max: {np.max(episode_rewards):.2f}"
    )
    plt.figtext(0.5, 0.02, summary_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    eval_plot_path = os.path.join(LOG_PATH, 'evaluation_results.png')
    plt.savefig(eval_plot_path)
    print(f"Plot saved to {eval_plot_path}")
    plt.show()


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Space Invaders Deep Reinforcement Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train Rainbow DQN
  python space_invaders.py train --episodes 1000
  
  # Train basic DQN
  python space_invaders.py train --basic --episodes 500
  
  # Train with rendering (slower)
  python space_invaders.py train --episodes 100 --render
  
  # Evaluate trained model (with pygame window)
  python space_invaders.py evaluate models/rainbow_final.pth
  
  # Evaluate best model
  python space_invaders.py evaluate models/rainbow_best.pth
  
  # Evaluate without rendering (faster)
  python space_invaders.py evaluate models/rainbow_final.pth --no-render --episodes 20
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    #Train command
    train_parser = subparsers.add_parser('train', help='Train an agent')
    train_parser.add_argument('--episodes', type=int, default=EPISODES, 
                            help=f'Number of episodes (default: {EPISODES})')
    train_parser.add_argument('--basic', action='store_true', 
                            help='Use basic DQN instead of Rainbow')
    train_parser.add_argument('--render', action='store_true', 
                            help='Render environment during training')
    train_parser.add_argument('--resume', type=str, default=None,
                            help='Resume training from checkpoint (path to .pth file)')
    
    #Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained agent')
    eval_parser.add_argument('model', type=str, help='Path to saved model')
    eval_parser.add_argument('--episodes', type=int, default=10, 
                           help='Number of episodes (default: 10)')
    eval_parser.add_argument('--no-render', action='store_true', 
                           help='Disable rendering')
    eval_parser.add_argument('--record', action='store_true', help='Record evaluation to video')
    eval_parser.add_argument('--video', type=str, default=None, help='Path to save recorded video')
    eval_parser.add_argument('--fps', type=int, default=60, help='FPS for recorded video (default: 60)')
    
    args = parser.parse_args()
    
    #Execute command
    if args.command == 'train':
        train(episodes=args.episodes, use_rainbow=not args.basic, render=args.render, resume_path=args.resume)
    
    elif args.command == 'evaluate':
        evaluate(args.model, episodes=args.episodes, render=not args.no_render, record=args.record, video_path=args.video, fps=args.fps)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()